# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from typing import Iterator, Optional, Sized

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from PIL import Image
import numpy as np
import cv2
import random
from itertools import zip_longest
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square, load_and_transform_video, get_video_transform
import copy


class LLaVADataset(Dataset):

    def __init__(self,
                 data_path,
                 image_folder,
                 video_data_path,
                 video_folder,
                 tokenizer,
                 image_processor,
                 frame_size = 336,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 video_frames=8,
                 video_batch_size=4,
                 image_batch_size=20,
                 pad_image_to_square=False,
                 shuffle_dataset=True):
        super().__init__()

        # json_data = json.load(open(data_path))
        json_video_data = json.load(open(video_data_path))

        self.shuffle_dataset = shuffle_dataset
        
        self.num_frames = video_frames
        self.video_batch_size = video_batch_size
        self.image_batch_size = image_batch_size
        self.remix_batch_size = {'image_num': 2, 'video_num': 2}
        image_count = 0
        video_count = 0
        for item in json_video_data:
            if 'image' in item.keys() and item['image'] != '':
                image_count+=1
            if 'video' in item.keys() and item['video'] != '':
                video_count+=1
        print(f'initial-image {image_count}, video {video_count}')
        # for idx in range(len(json_data)):
        #     if isinstance(json_data[idx]['id'], int):
        #         json_data[idx]['id'] = str(json_data[idx]['id'])
        for idx in range(len(json_video_data)):
            if isinstance(json_video_data[idx]['id'], int):
                json_video_data[idx]['id'] = str(json_video_data[idx]['id'])

        
        json_video_data = DatasetDict({'train': HFDataset.from_list(json_video_data)})

        self.text_data = process_hf_dataset(
            dataset=json_video_data,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            split='train',
            max_dataset_length=max_dataset_length,
            remove_unused_columns=False,
            pack_to_max_length=False,
            with_image_token=True)
        self.frame_size = frame_size
        self.image_folder = image_folder
        self.video_folder = video_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor


        self.pad_image_to_square = pad_image_to_square

        self.batched_data = self.get_batched_image_OR_video()
        print('batched_data shape:',len(self.batched_data))

    def get_batched_mix_image_video(self):
        #one batch might be: [image, image, video, video] or [image, ..., image] or [video, video, video, video]
        #mix video and image
        #adopt to self.remix_batch_size
        batched_data = []
        image_data = []
        video_data = []
        
        
        for item in self.text_data:
            if item.get('image', None) is not None and item['image'] != '':
                image_data.append(item)
            elif item.get('video', None) is not None and item['video'] != '':
                video_data.append(item)
            else:
                image_data.append(item)
        if self.shuffle_dataset:
            random.shuffle(image_data)
            random.shuffle(video_data)

        print(f'image data {len(image_data)}, video data {len(video_data)}\n')
        min_len = int(len(image_data) / self.remix_batch_size['image_num']) \
                    if int(len(image_data) / self.remix_batch_size['image_num']) < int(len(video_data)/self.remix_batch_size['video_num']) \
                    else int(len(video_data)/self.remix_batch_size['video_num'])
        image_data_more = int(len(image_data) / self.remix_batch_size['image_num']) > int(len(video_data)/self.remix_batch_size['video_num'])

        for i in range(min_len):
            batched_item = []
            batched_item.extend(image_data[i:i+self.remix_batch_size['image_num']])
            batched_item.extend(video_data[i:i+self.remix_batch_size['video_num']])
            if self.shuffle_dataset:    
                random.shuffle(batched_item)
            batched_data.append(batched_item)
        
        if image_data_more:
            for i in range(min_len*self.remix_batch_size['image_num'], len(image_data), self.image_batch_size):
                batched_item.extend(image_data[i:i+self.image_batch_size])
                if self.shuffle_dataset:    
                    random.shuffle(batched_item)
                batched_data.append(batched_item)
        else:
            for i in range(min_len*self.remix_batch_size['video_num'], len(video_data), self.video_batch_size):
                batched_item.extend(video_data[i:i+self.video_batch_size])
                if self.shuffle_dataset:    
                    random.shuffle(batched_item)
                batched_data.append(batched_item)

        if self.shuffle_dataset:    
            random.shuffle(batched_data)

        return batched_data

    def get_batched_image_OR_video(self):
        #one batch might be: [image, ..., image] or [video, video, video, video]
        #mix video and image
        #adopt to self.remix_batch_size
        batched_data = []
        image_data = []
        video_data = []
        
        
        for item in self.text_data:
            if item.get('image', None) is not None and item['image'] != '':
                image_data.append(item)
            elif item.get('video', None) is not None and item['video'] != '':
                video_data.append(item)
            else:
                image_data.append(item)
        if self.shuffle_dataset:
            random.shuffle(image_data)
            random.shuffle(video_data)

        print(f'image data {len(image_data)}, video data {len(video_data)}\n')

        for i in range(0, len(image_data), self.image_batch_size):
            batched_item = image_data[i:i+self.image_batch_size]
            batched_data.append(batched_item)
        print(f'finish add image data')
        
        for i in range(0, len(video_data), self.video_batch_size):
            batched_item = video_data[i:i+self.video_batch_size]
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
        return len(self.batched_data)

    def __getitem__(self, index):

        batched_item = self.batched_data[index]
        copyed_batched_item = copy.deepcopy(batched_item)
        index_list = list(range(len(batched_item)))
        for idx, (data_dict) in enumerate(copyed_batched_item):
            try:
                if data_dict.get('image', None) is not None and data_dict['image'] != '':
                    image_file = data_dict['image']
                    # print('image:', image_file)
                    image = Image.open(os.path.join(self.image_folder,
                                                    image_file)).convert('RGB')
                    if self.pad_image_to_square:
                        image = expand2square(
                            image,
                            tuple(
                                int(x * 255) for x in self.image_processor.image_mean))
                    image = self.image_processor.preprocess(
                        image, return_tensors='pt')['pixel_values'][0]
                    data_dict['image_pixel_values'] = image #might cause OOM 
                    # print(f'success get image')
                elif data_dict.get('video', None) is not None and data_dict['video'] != '':
                    video_file = data_dict['video']
                    # print('video:', video_file)
                    video_decode_backend = 'decord'
                    
                    video = load_and_transform_video(os.path.join(self.video_folder, video_file), 
                                                    get_video_transform(video_decode_backend=video_decode_backend, num_frames=self.num_frames, frame_size=self.frame_size),
                                                    video_decode_backend=video_decode_backend,
                                                    num_frames=self.num_frames)
                    # print(f'success get video')
                    data_dict['video_pixel_values'] = video
                    
                else:
                    crop_size = self.image_processor.crop_size
                    data_dict['image_pixel_values'] = torch.zeros(3, crop_size['height'],
                                                            crop_size['width'])
            except Exception as e:
                print(f'Error with {e}, idx {idx}')
                index_list.remove(idx)
                pass
        
        # print(f'batched_item[index_list] {len([batched_item[i]for i in index_list])}')
        return [copyed_batched_item[i]for i in index_list]

        # try:
        # batch_size = 5
        # num_frames = 10
        # image_batch_size = 20
        # image_batch = []
        # video_batch = []

        # while len(image_batch) < image_batch_size or len(video_batch) < batch_size:
        #     get_index  = random.randint(0, self.__len__()-1)
        #     data_dict = self.text_data[get_index]
        #     try:
        #         if data_dict.get('image', None) is not None and data_dict['image'] != '':
        #             image_file = data_dict['image']
        #             image = Image.open(os.path.join(self.image_folder,
        #                                             image_file)).convert('RGB')
        #             if self.pad_image_to_square:
        #                 image = expand2square(
        #                     image,
        #                     tuple(
        #                         int(x * 255) for x in self.image_processor.image_mean))
        #             image = self.image_processor.preprocess(
        #                 image, return_tensors='pt')['pixel_values'][0]
        #             data_dict['image_pixel_values'] = image
        #             image_batch.append(data_dict)

        #         elif data_dict.get('video', None) is not None and data_dict['video'] != '':
        #             video_file = data_dict['video']
        #             video_decode_backend = 'decord'
                    
        #             video = load_and_transform_video(os.path.join(self.video_folder,video_file), 
        #                                             get_video_transform(video_decode_backend=video_decode_backend,num_frames=num_frames),
        #                                             video_decode_backend=video_decode_backend,
        #                                             num_frames=num_frames)
        #             # print(f'success get video')
        #             data_dict['video_pixel_values'] = video
        #             video_batch.append(data_dict)
        #         else:
        #             crop_size = self.image_processor.crop_size
        #             data_dict['image_pixel_values'] = torch.zeros(3, crop_size['height'],
        #                                                     crop_size['width'])
        #             image_batch.append(data_dict)

        #     except Exception as e:
        #         print(f'Error with {e}')
        #         pass
        # image_batch = random.sample(image_batch, image_batch_size)
        # video_batch = random.sample(video_batch, batch_size)
        
        # return image_batch, video_batch

