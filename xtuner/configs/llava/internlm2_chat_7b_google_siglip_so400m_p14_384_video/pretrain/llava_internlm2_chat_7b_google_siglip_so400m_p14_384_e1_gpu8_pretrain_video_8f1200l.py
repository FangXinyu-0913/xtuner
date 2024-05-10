# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, AutoModel, AutoProcessor)

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn, video_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import VideoLengthGroupedSampler, VideoImageSeperateBatchSampler
from xtuner.engine import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = '/cpfs01/shared/llmeval/dhd/hub/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f'
visual_encoder_name_or_path = 'google/siglip-so400m-patch14-384' #'/cpfs01/shared/llmeval/dhd/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'

# Data
data_path = '/cpfs01/user/fangxinyu/Video-LLaVA/data/llava_image_tune/llava_v1_5_mix665k.json' #image_path
image_folder = '/cpfs01/user/fangxinyu/Video-LLaVA/data'
video_data_path = '/cpfs01/user/fangxinyu/Video-LLaVA/data/train_json/valley_llavaimage_pretrain_modify_shuffle.json'
video_folder = '/cpfs01/user/fangxinyu/Video-LLaVA/data'
# offline_data_folder_full='/cpfs01/shared/llmeval/fangxinyu/video_related_data/xtuner_offline_data/valley_llavaimage_pretrain_modify_shuffle_internlm2_7b_1536l_llava_map_fn_offline'
prompt_template = PROMPT_TEMPLATE.internlm2_chat

video_frames = 8
video_batch_size = 3
image_batch_size = 18
frame_size = 384
pixel_size = 14
max_length = int(1200) #text max length, the same with previous situation

# Scheduler & Optimizer
batch_size = image_batch_size  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 1e-3
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']
evaluation_videos = '/cpfs01/user/fangxinyu/xtuner/xtuner/dataset/sample_demo_1.mp4'
evaluation_inputs_video = ['为什么这段视频很有趣', 'Why is this video funny']

#######################################################################
#            PART 2  Model & Tokenizer & Video Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=AutoProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path)

model = dict(
    type=LLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    video_frames=video_frames,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        torch_dtype = torch.bfloat16))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVADataset,
    # offline_processed_text_folder=offline_data_folder_full,
    data_path=data_path,
    image_folder=image_folder,
    video_data_path=video_data_path,
    video_folder=video_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    video_batch_size=video_batch_size,
    image_batch_size=image_batch_size,
    max_length=max_length,
    frame_size=frame_size,
    video_frames=video_frames,
    pad_image_to_square=False)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(
        type=VideoLengthGroupedSampler, 
        length_property='modality_length',
        per_device_batch_size = batch_size * accumulative_counts,
    ),
    batch_sampler=dict(
        type=VideoImageSeperateBatchSampler, video_batch_size=video_batch_size, drop_last=True
    ),
    collate_fn=dict(type=video_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        frame_size=frame_size,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_videos=evaluation_videos,
        evaluation_inputs_video=evaluation_inputs_video,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        video_frames=video_frames,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
