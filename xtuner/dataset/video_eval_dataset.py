# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import warnings
from typing import Iterator, Optional, Sized

import torch
from torch import distributed as dist
from mmengine import print_log
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine.config import Config, ConfigDict
from mmengine.dist import sync_random_seed
from mmengine.utils.misc import get_object_from_string
from PIL import Image
import numpy as np
import cv2
import random
from itertools import zip_longest
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square, load_and_transform_video, get_video_transform
from xtuner.utils import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN,
    VIDEO_TOKEN_INDEX,
)
import copy


class VideoEvalDataset(Dataset):

    def __init__(
        self,
        video_data_path,
        # video_data_path_question,
        # video_data_path_answer,
        video_folder,
        tokenizer,
        image_processor,
        image_folder=None,
        offline_processed_text_folder=None,
        system='',
        prompt_template=None,
        video_frames=8,
        video_batch_size=4,
        image_batch_size=20,
        resolution=224,
        pad_image_to_square=False,
        shuffle_dataset=True,
        seed: Optional[int] = None,
        stop_word=None,
        stop_words=[],
    ):
        super().__init__()

        # json_data = json.load(open(data_path))
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.shuffle_dataset = shuffle_dataset
        self.tokenizer = BUILDER.build(tokenizer)
        self.resolution = resolution
        self.num_frames = video_frames
        self.video_batch_size = video_batch_size
        self.image_batch_size = image_batch_size
        self.remix_batch_size = {'image_num': 2, 'video_num': 2}
        image_count = 0
        video_count = 0
        pure_conv = 0

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            # json_video_data_question = json.load(open(video_data_path_question)) if os.path.splitext(video_data_path_question)[1]=='.json' else [json.loads(line) for line in open(video_data_path_question)]
            # json_video_data_answer = json.load(open(video_data_path_answer)) if os.path.splitext(video_data_path_answer)[1]=='.json' else [json.loads(line) for line in open(video_data_path_answer)]
            if os.path.splitext(video_data_path)[1]=='.json':
                json_video_data = json.load(open(video_data_path)) #if os.path.splitext(video_data_path)[1]=='.json' else [json.loads(line) for line in open(video_data_path)]
            elif os.path.splitext(video_data_path)[1]=='.jsonl':
                json_video_data = [json.loads(line) for line in open(video_data_path)]
            else:
                print("not json/jsonl file!")
                raise

        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
            stop_words += prompt_template.get('STOP_WORDS', [])
        if stop_word is not None:
            # TODO: deprecation, v0.3.0
            warnings.warn(
                ('The `stop_word` argument is deprecated and will be removed '
                 'in v0.3.0, use `stop_words` instead.'), DeprecationWarning)
            stop_words.append(stop_word)

        for j in json_video_data:
            sample_video_input = j['question']
            if 'video_id' in j:
                special_token = DEFAULT_VIDEO_TOKEN
            elif 'image' in j:
                special_token = DEFAULT_IMAGE_TOKEN
            else:
                print("pure conversation")
                special_token=''
            sample_input = special_token + '\n' + sample_video_input
            # inputs = (system + instruction).format(
            #     input=sample_input, round=1, **runner.cfg)
            inputs = (system + instruction).format(input=sample_input, round=1)
            chunk_encode = []
            if special_token:
                for idx, chunk in enumerate(inputs.split(special_token)):
                    if idx == 0:
                        cur_encode = self.tokenizer.encode(chunk)
                    else:
                        cur_encode = self.tokenizer.encode(
                            chunk, add_special_tokens=False)
                    chunk_encode.append(cur_encode)
                assert len(chunk_encode) == 2
            else:
                chunk_encode = [self.tokenizer.encode(inputs)]
            input_ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_ids.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    if special_token == DEFAULT_VIDEO_TOKEN:
                        input_ids.append(VIDEO_TOKEN_INDEX)
                    elif special_token == DEFAULT_IMAGE_TOKEN:
                        input_ids.append(IMAGE_TOKEN_INDEX)
            # input_ids = torch.tensor(input_ids).to(device)
            j['input_ids'] = input_ids
        self.text_data = json_video_data

        self.image_folder = image_folder
        self.video_folder = video_folder
        if (
            isinstance(image_processor, dict)
            or isinstance(image_processor, Config)
            or isinstance(image_processor, ConfigDict)
        ):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        self.pad_image_to_square = pad_image_to_square

        # self.batched_data = self.get_batched_image_OR_video()
        # print('batched_data shape:', len(self.batched_data))

    def get_batched_mix_image_video(self):
        # one batch might be: [image, image, video, video] or [image, ..., image] or [video, video, video, video]
        # mix video and image
        # adopt to self.remix_batch_size
        batched_data = []
        image_data = []
        video_data = []
        random.seed(self.seed)

        for item in self.text_data:
            if 'image' in item.keys():
                image_data.append(item)
            elif 'video' in item.keys():
                video_data.append(item)
            else:
                raise ValueError('No image or video in the data')
        if self.shuffle_dataset:
            random.shuffle(image_data)
            random.shuffle(video_data)

        print(f'image data {len(image_data)}, video data {len(video_data)}\n')
        min_len = (
            int(len(image_data) / self.remix_batch_size['image_num'])
            if int(len(image_data) / self.remix_batch_size['image_num'])
            < int(len(video_data) / self.remix_batch_size['video_num'])
            else int(len(video_data) / self.remix_batch_size['video_num'])
        )
        image_data_more = int(len(image_data) / self.remix_batch_size['image_num']) > int(
            len(video_data) / self.remix_batch_size['video_num']
        )

        for i in range(min_len):
            batched_item = []
            batched_item.extend(image_data[i : i + self.remix_batch_size['image_num']])
            batched_item.extend(video_data[i : i + self.remix_batch_size['video_num']])
            if self.shuffle_dataset:
                random.shuffle(batched_item)
            batched_data.append(batched_item)

        if image_data_more:
            for i in range(
                min_len * self.remix_batch_size['image_num'], len(image_data), self.image_batch_size
            ):
                batched_item.extend(image_data[i : i + self.image_batch_size])
                if self.shuffle_dataset:
                    random.shuffle(batched_item)
                batched_data.append(batched_item)
        else:
            for i in range(
                min_len * self.remix_batch_size['video_num'], len(video_data), self.video_batch_size
            ):
                batched_item.extend(video_data[i : i + self.video_batch_size])
                if self.shuffle_dataset:
                    random.shuffle(batched_item)
                batched_data.append(batched_item)

        if self.shuffle_dataset:
            random.shuffle(batched_data)

        return batched_data

    def get_batched_image_OR_video(self):
        # one batch might be: [image, ..., image] or [video, video, video, video]
        # mix video and image
        # adopt to self.remix_batch_size
        batched_data = []
        image_data = []
        video_data = []
        random.seed(self.seed)

        for item in self.text_data:
            if 'image' in item.keys() and item['image']:
                image_data.append(item)
            elif 'video' in item.keys() and item['video']:
                video_data.append(item)
            else:
                image_data.append(item)
                # raise ValueError('No image or video in the data')
        if self.shuffle_dataset:
            random.shuffle(image_data)
            random.shuffle(video_data)

        print(f'image data {len(image_data)}, video data {len(video_data)}\n')

        for i in range(0, len(image_data), self.image_batch_size):
            batched_item = image_data[i : i + self.image_batch_size]
            batched_data.append(batched_item)
        print(f'finish add image data')

        for i in range(0, len(video_data), self.video_batch_size):
            batched_item = video_data[i : i + self.video_batch_size]
            batched_data.append(batched_item)
        print(f'finish add video data')

        if self.shuffle_dataset:
            random.shuffle(batched_data)

        return batched_data

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):

        index_data = self.text_data[index]
        copyed_index_data = copy.deepcopy(index_data)
        try:
            # if data_dict.get('image', None) is not None and data_dict['image'] != '':
            if 'image' in copyed_index_data and copyed_index_data['image']:
                image_file = copyed_index_data['image']
                # print('image:', image_file)
                image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
                if self.pad_image_to_square:
                    image = expand2square(
                        image, tuple(int(x * 255) for x in self.image_processor.image_mean)
                    )
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                copyed_index_data['image_pixel_values'] = image  # might cause OOM
                # print(f'success get image')
            # elif data_dict.get('video', None) is not None and data_dict['video'] != '':
            elif 'video_id' in copyed_index_data and copyed_index_data['video_id']:
                video_file = copyed_index_data['video_id']
                if len(video_file.split('.')) == 1:
                    if 'format' in copyed_index_data:
                        video_file = video_file+'.'+copyed_index_data['format']
                    else:
                        video_file = video_file+'.mp4'
                # print('video:', video_file)
                video_decode_backend = 'decord'

                video = load_and_transform_video(
                    os.path.join(self.video_folder, video_file),
                    get_video_transform(
                        video_decode_backend=video_decode_backend,
                        num_frames=self.num_frames,
                        frame_size=self.resolution,
                    ),
                    video_decode_backend=video_decode_backend,
                    num_frames=self.num_frames,
                )
                # print(f'success get video')
                copyed_index_data['video_pixel_values'] = video.transpose(0,1)

            else:

                crop_size = self.image_processor.crop_size
                copyed_index_data['image_pixel_values'] = torch.zeros(
                    3, crop_size['height'], crop_size['width']
                )
        except Exception as e:
            print(f'Error with {e}, idx {index_data}')
            pass

        
        return copyed_index_data