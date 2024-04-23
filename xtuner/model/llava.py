# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig

from xtuner.registry import BUILDER
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)
import time
from torch.autograd import profiler
from .chatuniviModel import ChatUniViMetaForCausalLM

from typing import List, Optional

import torch
from mmengine import print_log
from mmengine.utils.misc import get_object_from_string
from mmengine.model import BaseModel
from peft import PeftType
from torch import nn
from transformers import PreTrainedModel

from xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX


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
                 enable_compress_tokens=False,
                 max_position_embeddings=None):
        
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

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

        self.config = {'mm_hidden_size': 1024}
        self.enable_compress_tokens = enable_compress_tokens
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

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        # hardcode for internlm2
        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config',
                             'Starcoder2Config', 'Starcoder2Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Starcoder2Config', 'Starcoder2Config')

        if SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch.bfloat16 \
                if torch.cuda.is_bf16_supported() else torch.float16
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

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
                data['pixel_values'].to(self.visual_encoder.dtype), output_hidden_states=True)
            # end_encoder_time = time.perf_counter()
            
            # if self.enable_compress_tokens:
            #     pixel_values = visual_outputs.hidden_states[self.visual_select_layer][:, 1:]
            # else:
            pixel_values = self.projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
            # end_projector_time = time.perf_counter()
            
            # print(f'after projection, pixel value shape: {pixel_values.shape}, instance list {instance_type}')
            data['pixel_values'] = pixel_values

            # if self.enable_compress_tokens:
            #     data = self.prepare_inputs_labels_for_multimodal(chatunivimodel = self.chat_univi_model, llm=self.llm, instance_list=instance_type, video_frames=self.video_frames, **data)
            # else:
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, video_frames=self.video_frames, **data)
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
        

    # def prepare_inputs_labels_for_multimodal(
    #     self,
    #     llm: PreTrainedModel,
    #     video_frames: Optional[int] = 10,
    #     instance_list: List[str] = ['image'],
    #     input_ids: torch.LongTensor = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     chatunivimodel: BaseModel = None):
    #     # projector: Optional[nn.Module] = None):
    #     if pixel_values is None:
    #         return {
    #             'input_ids': input_ids,
    #             'position_ids': position_ids,
    #             'attention_mask': attention_mask,
    #             'past_key_values': past_key_values,
    #             'inputs_embeds': None,
    #             'labels': labels
    #         }

    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(
    #             0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # remove the padding using attention_mask -- TODO: double check
    #     input_ids = [
    #         cur_input_ids[cur_attention_mask]
    #         for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    #     ]
    #     labels = [
    #         cur_labels[cur_attention_mask]
    #         for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    #     ]

    #     new_inputs_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0

    #     split_sizes_overall = []
    #     vision_feature_overall = []
    #     overall_feat_after_proj_split_list = []
    #     if chatunivimodel is not None:
    #         for batch_idx, (cur_input_ids, instance) in enumerate(zip(input_ids, instance_list)):
    #             num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #             num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()
    #             split_sizes_perinstance = []
    #             vision_feat_perinstance = []
    #             if num_videos > 0:
    #                 for i in range(num_videos + 1):
    #                     if i < num_videos:
    #                         cur_pixel_values = pixel_values[cur_image_idx: cur_image_idx + video_frames]
    #                         cur_image_idx += video_frames
    #                         cur_image_features = chatunivimodel(cur_pixel_values, input_type="video").squeeze(0)
    #                         # cur_image_features = cur_image_features.to(cur_inputs_embeds.dtype)
    #                         # cur_image_features = cur_image_features.to(cur_inputs_embeds.device)
    #                         split_sizes_perinstance.append(cur_image_features.shape[0])
    #                         vision_feat_perinstance.append(cur_image_features)

    #             if num_images > 0:
    #                 for i in range(num_images + 1):
    #                     if i < num_images:
    #                         cur_pixel_values = pixel_values[cur_image_idx]
    #                         cur_image_idx += 1
    #                         cur_image_features = chatunivimodel(cur_pixel_values.unsqueeze(0), input_type="image").squeeze(0)
    #                         # cur_image_features = cur_image_features.to(cur_inputs_embeds.dtype)
    #                         # cur_image_features = cur_image_features.to(cur_inputs_embeds.device)
    #                         split_sizes_perinstance.append(cur_image_features.shape[0])
    #                         vision_feat_perinstance.append(cur_image_features)
    #                         # print('cur_image_features.shape:',cur_image_features.shape)

    #             if num_images == 0 and num_videos == 0:
    #                 cur_pixel_values = pixel_values[cur_image_idx]
    #                 cur_image_idx += 1
    #                 # print(f'empty image:{num_images} video:{num_videos} instance {instance} cur_input_ids {cur_input_ids} pixel')
    #                 ZERO_VISION_FEAT = torch.zeros(96, 1024).to(self.visual_encoder.device).to(self.visual_encoder.dtype)
    #                 split_sizes_perinstance.append(ZERO_VISION_FEAT.shape[0])
    #                 vision_feat_perinstance.append(ZERO_VISION_FEAT)
                               
    #             split_sizes_overall.append(split_sizes_perinstance)
    #             vision_feature_overall.append(torch.cat(vision_feat_perinstance))

    #         overall_feat_before_proj = torch.cat(vision_feature_overall)
    #         overall_feat_after_proj = self.projector(overall_feat_before_proj)
    #         overall_feat_after_proj_split = torch.split(overall_feat_after_proj, [lis for lists in split_sizes_overall for lis in lists], dim=0)
    #         i = 0
            
    #         for split_sizes_perinstance in split_sizes_overall:
    #             overall_feat_after_proj_split_list.append(overall_feat_after_proj_split[i:i+len(split_sizes_perinstance)])
    #             i = i + len(split_sizes_perinstance)

    #     cur_image_idx = 0
    #     # print(chatunivimodel)

    #     if len(overall_feat_after_proj_split_list) == 0:
    #         overall_feat_after_proj_split_list = [torch.zeros(96, 1)] * len(instance_list)
    #     for batch_idx, (cur_input_ids, instance, vision_feat_perinstance) in enumerate(zip(input_ids, instance_list, overall_feat_after_proj_split_list)):
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()

    #         if num_images == 0 and num_videos == 0:
    #             cur_pixel_values = pixel_values[cur_image_idx]
    #             # print(cur_pixel_values[0:0].shape)
    #             cur_inputs_embeds_1 = llm.get_input_embeddings()(cur_input_ids)
    #             cur_inputs_embeds = torch.cat(
    #                 [cur_inputs_embeds_1], dim=0)
    #             new_inputs_embeds.append(cur_inputs_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

            
    #         if num_videos > 0:
    #             video_token_indices =  [-1] + torch.where(       #[-1, 4, cur_input_ids.shape[0]]
    #                 cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist() + [
    #                     cur_input_ids.shape[0]
    #                 ]

    #             cur_input_ids_noim = []
    #             cur_labels = labels[batch_idx]
    #             cur_labels_noim = []
    #             for i in range(len(video_token_indices) - 1):
    #                 cur_input_ids_noim.append(cur_input_ids[video_token_indices[i] +
    #                                                         1:video_token_indices[i +
    #                                                                             1]])
    #                 cur_labels_noim.append(cur_labels[video_token_indices[i] +
    #                                                 1:video_token_indices[i + 1]])
                
    #             split_sizes = [x.shape[0] for x in cur_labels_noim]
    #             cur_inputs_embeds = llm.get_input_embeddings()(
    #                 torch.cat(cur_input_ids_noim))
    #             cur_inputs_embeds_no_im = torch.split(
    #                 cur_inputs_embeds, split_sizes, dim=0)
    #             cur_new_inputs_embeds = []
    #             cur_new_labels = []


    #             for i in range(num_videos + 1):
    #                 cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
    #                 cur_new_labels.append(cur_labels_noim[i])
    #                 if i < num_videos:
    #                     cur_pixel_values = pixel_values[cur_image_idx: cur_image_idx + video_frames]
    #                     cur_image_idx += video_frames
    #                     # merge video frames
    #                     if chatunivimodel is None:
    #                         for item in cur_pixel_values:
    #                         #VIDEO as N * IMAGE, NOT COMPRSEE
    #                             cur_new_inputs_embeds.append(item) #check here
    #                             cur_new_labels.append(
    #                                 torch.full((item.shape[0], ),#report error here
    #                                         IGNORE_INDEX,
    #                                         device=cur_labels.device,
    #                                         dtype=cur_labels.dtype))
    #                     else:
    #                         cur_new_inputs_embeds.append(vision_feat_perinstance[i])
    #                         cur_new_labels.append(
    #                                 torch.full((vision_feat_perinstance[i].shape[0], ),#report error here
    #                                         IGNORE_INDEX,
    #                                         device=cur_labels.device,
    #                                         dtype=cur_labels.dtype))
                        

            
    #         elif num_images > 0:
    #             image_token_indices = [-1] + torch.where(
    #                 cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
    #                     cur_input_ids.shape[0]
    #                 ]
    #             # print(f'image token indices: {image_token_indices}, {cur_input_ids}')
    #             cur_input_ids_noim = []
    #             cur_labels = labels[batch_idx]
    #             cur_labels_noim = []
    #             for i in range(len(image_token_indices) - 1):
    #                 cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
    #                                                         1:image_token_indices[i +
    #                                                                             1]])
    #                 cur_labels_noim.append(cur_labels[image_token_indices[i] +
    #                                                 1:image_token_indices[i + 1]])
    #             split_sizes = [x.shape[0] for x in cur_labels_noim]
    #             cur_inputs_embeds = llm.get_input_embeddings()(
    #                 torch.cat(cur_input_ids_noim))
    #             cur_inputs_embeds_no_im = torch.split(
    #                 cur_inputs_embeds, split_sizes, dim=0)
    #             cur_new_inputs_embeds = []
    #             cur_new_labels = []

    #             for i in range(num_images + 1):
    #                 cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
    #                 cur_new_labels.append(cur_labels_noim[i])
    #                 if i < num_images:
    #                     cur_pixel_values = pixel_values[cur_image_idx]
    #                     cur_image_idx += 1
    #                     if chatunivimodel is None:
    #                         cur_new_inputs_embeds.append(cur_pixel_values)
    #                         cur_new_labels.append(
    #                             torch.full((cur_pixel_values.shape[0], ),
    #                                     IGNORE_INDEX,
    #                                     device=cur_labels.device,
    #                                     dtype=cur_labels.dtype))
    #                     else:
    #                         # cur_image_features = chatunivimodel(cur_pixel_values.unsqueeze(0), input_type="image").squeeze(0)
    #                         # cur_image_features = cur_image_features.to(cur_inputs_embeds.dtype)
    #                         # cur_image_features = cur_image_features.to(cur_inputs_embeds.device)
    #                         # cur_image_features = self.projector(cur_image_features)
    #                         cur_new_inputs_embeds.append(vision_feat_perinstance[i])
    #                         cur_new_labels.append(
    #                             torch.full((vision_feat_perinstance[i].shape[0], ),
    #                                     IGNORE_INDEX,
    #                                     device=cur_labels.device,
    #                                     dtype=cur_labels.dtype))


    #         cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_inputs_embeds.append(cur_new_inputs_embeds)
    #         new_labels.append(cur_new_labels)

    #     # Combine them
    #     max_len = max(x.shape[0] for x in new_inputs_embeds)
    #     batch_size = len(new_inputs_embeds)

    #     new_inputs_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len),
    #                                 IGNORE_INDEX,
    #                                 dtype=new_labels[0].dtype,
    #                                 device=new_labels[0].device)
    #     attention_mask = torch.zeros((batch_size, max_len),
    #                                 dtype=attention_mask.dtype,
    #                                 device=attention_mask.device)
    #     position_ids = torch.zeros((batch_size, max_len),
    #                             dtype=position_ids.dtype,
    #                             device=position_ids.device)

    #     for i, (cur_new_embed,
    #             cur_new_labels) in enumerate(zip(new_inputs_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
    #         new_inputs_embeds_padded.append(
    #             torch.cat((cur_new_embed,
    #                     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
    #                                 dtype=cur_new_embed.dtype,
    #                                 device=cur_new_embed.device)),
    #                     dim=0))
    #         if cur_len > 0:
    #             new_labels_padded[i, :cur_len] = cur_new_labels
    #             attention_mask[i, :cur_len] = True
    #             position_ids[i, :cur_len] = torch.arange(
    #                 0,
    #                 cur_len,
    #                 dtype=position_ids.dtype,
    #                 device=position_ids.device)

    #     new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None
    #     # try:
    #     #     print(new_inputs_embeds.shape, new_labels.shape)
    #     # except:
    #     #     print('get failure')
    #     # print(new_inputs_embeds.shape)
    #     # print(position_ids.shape)
    #     # print(new_inputs_embeds.requires_grad)
    #     return {
    #         'input_ids': None,
    #         'position_ids': position_ids,
    #         'attention_mask': attention_mask,
    #         'past_key_values': past_key_values,
    #         'inputs_embeds': new_inputs_embeds,
    #         'labels': new_labels
    #     }
