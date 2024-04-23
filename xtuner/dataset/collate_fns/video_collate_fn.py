# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def video_collate_fn(wrapped_instances: Sequence[Dict],
                    pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                    return_hf_format: bool = False,
                    use_varlen_attn: bool = False):

    instances = wrapped_instances[0]
    input_ids, labels = [], []
    pixel_values = []
    has_image, has_video = any(inst.get('image_pixel_values') is not None for inst in instances), any(inst.get('video_pixel_values') is not None for inst in instances)
    seq_parallel_world_size = get_sequence_parallel_world_size()

    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not (has_image or has_video), 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image or has_video:
        pixel_values = []

    for example in instances:
        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))

        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))
        if has_video and 'video_pixel_values' in example.keys():
            c,b,h,w = example['video_pixel_values'].shape
            video_pixel_values = example['video_pixel_values'].reshape(b,c,h,w)
            pixel_values.extend([video_pixel_values[i] for i in range(video_pixel_values.size(0))])
            # instances_type.append('video')
        if has_image and 'image_pixel_values' in example.keys():
            pixel_values.append(example['image_pixel_values'])
            # instances_type.append('image')
        
    # print(instances_type)    
    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        try:
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX)
        except:
            print(instances)
            raise
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    
    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:

        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in ori_length:
            attention_mask[:i] = True
        

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    input_ids, labels, position_ids, attention_mask = \
        pad_for_sequence_parallel(input_ids, labels, position_ids,
                                  attention_mask)

    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }

    if has_image or has_video:
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
            
    
    # input_ids = []
    # labels = []
    # image_batch = instances[0][0] #image_batch
    # video_batch = instances[0][1] #video_batch
    # has_image = len(image_batch) > 0
    # has_video = len(video_batch) > 0
    # return_image_batch = False
    # if random.random() > 0.5:
    #     return_image_batch = True
        
    # # has_image = any(inst.get('image_pixel_values') is not None for inst in instances)
    # # has_video = any(inst.get('video_pixel_values') is not None for inst in instances)

    # if use_varlen_attn:
    #     cumulative_len, indexes = [], []
    #     if return_image_batch:
    #         assert len(image_batch) == 1, (
    #             f'If utilizing local attention, the batch size should be'
    #             f' set to 1, but got {len(image_batch)}')
    #         assert not (has_image), 'Currently, it is not configured to '
    #         'accommodate the use of varlen Attention in multimodal training'
    #     else:
    #         assert len(video_batch) == 1, (
    #             f'If utilizing local attention, the batch size should be'
    #             f' set to 1, but got {len(video_batch)}')
    #         assert not (has_video), 'Currently, it is not configured to '
    #         'accommodate the use of varlen Attention in multimodal training'


    # if has_image or has_video:
    #     pixel_values = []
    #     instances_type = []
    # if return_image_batch:
    #     for example in image_batch:
    #         input_ids.append(torch.tensor(example['input_ids']))
    #         labels.append(torch.tensor(example['labels']))
    #         if use_varlen_attn:
    #             cumulative_len.append(torch.IntTensor(example['cumulative_len']))
    #             indexes.append(torch.LongTensor(example['indexes']))
    #         if has_image and 'image_pixel_values' in example.keys():
    #             pixel_values.append(example['image_pixel_values'])
    #             instances_type.append('image')
    # else:
    #     for example in video_batch:
    #         input_ids.append(torch.tensor(example['input_ids']))
    #         labels.append(torch.tensor(example['labels']))
    #         if use_varlen_attn:
    #             cumulative_len.append(torch.IntTensor(example['cumulative_len']))
    #             indexes.append(torch.LongTensor(example['indexes']))
    #         if has_video and 'video_pixel_values' in example.keys():
    #             c,b,h,w = example['video_pixel_values'].shape
    #             video_pixel_values = example['video_pixel_values'].reshape(b,c,h,w)
    #             pixel_values.extend([video_pixel_values[i] for i in range(video_pixel_values.size(0))])
    #             instances_type.append('video')
    
    # print(instances_type)    

    # input_ids = pad_sequence(
    #     input_ids, batch_first=True, padding_value=pad_index)
    # labels = pad_sequence(
    #     labels, batch_first=True, padding_value=IGNORE_INDEX)

    
    # if use_varlen_attn:
    #     indexes = torch.stack(indexes, dim=0)
    #     max_seqlen = (
    #         cumulative_len[0][1:] -  # noqa: W504
    #         cumulative_len[0][:-1]).max().item()
    #     data_dict = {
    #         'input_ids': input_ids,
    #         'cumulative_len': cumulative_len,
    #         'indexes': indexes,
    #         'labels': labels,
    #         'max_seqlen': max_seqlen
    #     }
    # else:
    #     data_dict = {
    #         'input_ids': input_ids,
    #         'attention_mask': input_ids.ne(pad_index),
    #         'labels': labels
    #     }

    # if has_image or has_video:
    #     pixel_values = torch.stack(pixel_values)
    #     data_dict['pixel_values'] = pixel_values
    #     data_dict['instance_type'] = instances_type

    # if return_hf_format:
    #     return data_dict
    # else:
    #     return {'data': data_dict, 'data_samples': None}



