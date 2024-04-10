# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)
import time
from torch.autograd import profiler
from .chatuniviModel import ChatUniViMetaForCausalLM

class CompressNet(nn.Module):
    def __init__(self, input_size):
        super(CompressNet, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((input_size[1], input_size[2]))

    def forward(self, x):
        # 假设 x 的形状是 [N, 20, A, B]
        # 在第一个维度上应用全局平均池化
        x = self.global_avg_pool(x)
        # 输出的形状是 [N, 1, A, B]
        return x

class LLaVAModel(BaseModel):

    def __init__(self,
                 llm,
                 visual_encoder,
                 video_frames,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=2,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 mode='pretrain',
                 enable_compress_tokens=False):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.visual_encoder = self._build_from_cfg_or_module(
                visual_encoder)
        self.llm.config.use_cache = False
        self.video_frames = video_frames
        dispatch_modules(self.llm)

        projector_config = ProjectorConfig(
            visual_hidden_size=self.visual_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth)
        self.projector = ProjectorModel(projector_config).to(
            self.visual_encoder.dtype)

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

        self.model_args = {
            'pretrain': {
                
                "use_cluster": True,
                "freeze": False,
                "vision_tune": False,

                "spatial_cluster_rate0": 64,  # 0.25
                "spatial_cluster_rate1": 32,  # 0.5
                "spatial_cluster_rate2": 16,  # 0.5

                "temporal_cluster_rate": 1/16,
            },

            'finetune':{

                "use_cluster": True,
                "freeze": False,
                "mm_tune": True,
                "vision_tune": False,

                "spatial_cluster_rate0": 64,  # 0.25
                "spatial_cluster_rate1": 32,  # 0.5
                "spatial_cluster_rate2": 16,  # 0.5

                "temporal_cluster_rate": 1/16,

            }
        }

        self.config = {'mm_hidden_size': 256}
        if enable_compress_tokens:
            self.chat_univi_model = ChatUniViMetaForCausalLM(model_args=self.model_args[mode], config=self.config)
        else:
            self.chat_univi_model = None
        # self.video_compress = CompressNet([20,336,336])

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()
        print('activate gradient')

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.visual_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):
        # self.train()
        if 'pixel_values' in data:

            # print(data['pixel_values'].shape, data['instance_type'])
            # start_time = time.perf_counter()
            visual_outputs = self.visual_encoder(
                data['pixel_values'], output_hidden_states=True)
            # print(visual_outputs.requires_grad)
            # end_encoder_time = time.perf_counter()
            
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            # end_projector_time = time.perf_counter()
            instance_type = data['instance_type']
            
            # print(f'after projection, pixel value shape: {pixel_values.shape}, instance list {instance_type}')

            data['pixel_values'] = pixel_values
            # for item in data['instance_type']:
            del data['instance_type']
            data = prepare_inputs_labels_for_multimodal(chatunivimodel = self.chat_univi_model, llm=self.llm, instance_list=instance_type, video_frames=self.video_frames, **data)
            
            # end_prepare_inputs_labels_for_multimodal_time = time.perf_counter()

            # print(f'visual_encoder:{end_encoder_time - start_time}\
            #         projector:{end_projector_time - end_encoder_time}\
            #         prepare_inputs_labels_for_multimodal: {end_prepare_inputs_labels_for_multimodal_time - end_projector_time}')
        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        # start_time = time.perf_counter()
        # with profiler.profile(use_cuda=True) as prof:
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        # prof.export_chrome_trace("trace.json")
        # end_time = time.perf_counter()
        # print(f'LLM time:{end_time - start_time}')
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
