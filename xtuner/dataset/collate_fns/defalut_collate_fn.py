# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def default_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):
    input_ids = []
    labels = []
    has_image = any(inst.get('image_pixel_values') is not None for inst in instances)
    has_video = any(inst.get('video_pixel_values') is not None for inst in instances)

    if use_varlen_attn:
        cumulative_len, indexes = [], []
        assert len(instances) == 1, (
            f'If utilizing local attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not (has_image or has_video), 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image or has_video:
        pixel_values = []
        instances_type = []
    for example in instances:
        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            indexes.append(torch.LongTensor(example['indexes']))
        if has_image and 'image_pixel_values' in example.keys():
            pixel_values.append(example['image_pixel_values'])
            instances_type.append('image')
        elif has_video and 'video_pixel_values' in example.keys():
            c,b,h,w = example['video_pixel_values'].shape
            video_pixel_values = example['video_pixel_values'].reshape(b,c,h,w)
            pixel_values.extend([video_pixel_values[i] for i in range(video_pixel_values.size(0))])
            instances_type.append('video')
    if has_image or has_video:
        print(instances_type)

    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
    
    if use_varlen_attn:
        indexes = torch.stack(indexes, dim=0)
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'indexes': indexes,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(pad_index),
            'labels': labels
        }

    if has_image or has_video:
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values
        data_dict['instance_type'] = instances_type

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
