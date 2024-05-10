# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


# def video_collate_fn(wrapped_instances: Sequence[Dict],
#                     pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
#                     return_hf_format: bool = False,
#                     use_varlen_attn: bool = False):

#     instances = wrapped_instances[0]
#     input_ids, labels = [], []
#     pixel_values = []
#     has_image, has_video = any(inst.get('image_pixel_values') is not None for inst in instances), any(inst.get('video_pixel_values') is not None for inst in instances)
#     seq_parallel_world_size = get_sequence_parallel_world_size()

#     if use_varlen_attn:
#         position_ids, cumulative_len = [], []
#         assert len(instances) == 1, (
#             f'If utilizing varlen attention, the batch size should be'
#             f' set to 1, but got {len(instances)}')
#         assert not (has_image or has_video), 'Currently, it is not configured to '
#         'accommodate the use of varlen Attention in multimodal training'

#     if has_image or has_video:
#         pixel_values = []

#     for example in instances:
#         input_ids.append(torch.tensor(example['input_ids']))
#         labels.append(torch.tensor(example['labels']))

#         if use_varlen_attn:
#             cumulative_len.append(torch.IntTensor(example['cumulative_len']))
#             position_ids.append(torch.LongTensor(example['position_ids']))
#         if has_video and 'video_pixel_values' in example.keys():
#             c,b,h,w = example['video_pixel_values'].shape
#             video_pixel_values = example['video_pixel_values'].reshape(b,c,h,w)
#             pixel_values.extend([video_pixel_values[i] for i in range(video_pixel_values.size(0))])
#             # instances_type.append('video')
#         if has_image and 'image_pixel_values' in example.keys():
#             pixel_values.append(example['image_pixel_values'])
#             # instances_type.append('image')
#             # if torch.any(example['image_pixel_values']):
#             #     instances_type.append('image')
#             # else:
#             #     instances_type.append('text')
        
#     # print(instances_type)    
#     ori_length = [len(ids) for ids in input_ids]
#     if len(instances) > 1:
#         try:
#             input_ids = pad_sequence(
#                 input_ids, batch_first=True, padding_value=pad_index)
#             labels = pad_sequence(
#                 labels, batch_first=True, padding_value=IGNORE_INDEX)
#         except:
#             print(instances)
#             raise
#     else:
#         input_ids = torch.stack(input_ids)
#         labels = torch.stack(labels)

    
#     if use_varlen_attn:
#         assert input_ids.size(1) % seq_parallel_world_size == 0
#         attention_mask = None
#         position_ids = torch.stack(position_ids, dim=0)
#     else:

#         # Some tokenizers have the same eos token and pad token, so input_ids
#         # cannot be masked directly based on the pad token id.
#         attention_mask = torch.zeros_like(input_ids).bool()
#         for i in ori_length:
#             attention_mask[:i] = True
        

#         bs, seq_len = input_ids.shape
#         position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

#     input_ids, labels, position_ids, attention_mask = \
#         pad_for_sequence_parallel(input_ids, labels, position_ids,
#                                   attention_mask)

#     if use_varlen_attn:
#         max_seqlen = (
#             cumulative_len[0][1:] -  # noqa: W504
#             cumulative_len[0][:-1]).max().item()
#         data_dict = {
#             'input_ids': input_ids,
#             'cumulative_len': cumulative_len,
#             'position_ids': position_ids,
#             'labels': labels,
#             'max_seqlen': max_seqlen
#         }
#     else:
#         data_dict = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'position_ids': position_ids,
#             'labels': labels
#         }

#     if has_image or has_video:
#         pixel_values = torch.stack(pixel_values)
#         data_dict['pixel_values'] = pixel_values
#         # data_dict['instance_type'] = instances_type

#     if return_hf_format:
#         return data_dict
#     else:
#         return {'data': data_dict, 'data_samples': None}
            


def video_collate_fn(
    instances: Sequence[Dict],
    pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
    return_hf_format: bool = False,
    extra_collate_keys=None,
):
    input_ids = []
    labels = []
    cumulative_len = []
    position_ids = []

    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_labels = any(inst.get('labels') is not None for inst in instances)
    mode = 'train' if has_labels else 'eval'

    if has_image:
        pixel_values = []
        instance_type = []

    for i, data in enumerate(instances):
        input_ids.append(torch.LongTensor(data['input_ids']))
        if mode == 'train':
            labels.append(torch.LongTensor(data['labels']))

        if 'cumulative_len' in data:
            cumulative_len.append(torch.IntTensor(data['cumulative_len']))

        if has_image:
            if data['pixel_values'].dim() == 3:
                data['pixel_values'] = data['pixel_values'].unsqueeze(0)
                instance_type.append('image')
            elif data['pixel_values'].dim() == 4:
                data['pixel_values'] = data['pixel_values'].permute(1, 0, 2, 3)
                instance_type.append('video')
            pixel_values.append(data['pixel_values'])

    if len(instances) > 1:
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        if mode == 'train':
            labels = torch.stack(labels)

    if mode == 'train':
        attention_mask = input_ids.ne(pad_index)
        position_ids = attention_mask.long().cumsum(-1) - 1

    if len(cumulative_len) == 0:
        cumulative_len = None

    # print(instance_type, input_ids.shape)
    if mode == 'train':
        data_dict = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            # 'cumulative_len': cumulative_len,
        }
    else:
        data_dict = {
            'input_ids': input_ids,
        }

    if has_image:
        pixel_values = torch.cat(pixel_values, 0)
        data_dict['pixel_values'] = pixel_values

    if extra_collate_keys is not None:
        for key in extra_collate_keys:
            data_dict[key] = [inst[key] for inst in instances]

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}



