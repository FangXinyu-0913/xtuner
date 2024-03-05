# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from PIL import Image
import numpy as np
import cv2
import random
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square, load_and_transform_video, get_video_transform


class LLaVADataset(Dataset):

    def __init__(self,
                 data_path,
                 image_folder,
                 video_data_path,
                 video_folder,
                 tokenizer,
                 image_processor,
                 video_processor,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False):
        super().__init__()

        # json_data = json.load(open(data_path))
        json_video_data = json.load(open(video_data_path))
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

        # json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        json_video_data = DatasetDict({'train': HFDataset.from_list(json_video_data)})
        # self.text_data = process_hf_dataset(
        #     dataset=json_data,
        #     tokenizer=tokenizer,
        #     max_length=max_length,
        #     dataset_map_fn=dataset_map_fn,
        #     template_map_fn=template_map_fn,
        #     split='train',
        #     max_dataset_length=max_dataset_length,
        #     remove_unused_columns=False,
        #     pack_to_max_length=False,
        #     with_image_token=True)
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

        self.image_folder = image_folder
        self.video_folder = video_folder
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        if isinstance(video_processor, dict) or isinstance(
                video_processor, Config) or isinstance(video_processor,
                                                       ConfigDict):
            self.video_processor = BUILDER.build(video_processor)
        else:
            self.video_processor = video_processor
        self.pad_image_to_square = pad_image_to_square

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
        # data_dict = self.text_data[index]
        try:
            data_dict = self.text_data[index]
            # print(data_dict.keys())
            if data_dict.get('image', None) is not None and data_dict['image'] != '':
                image_file = data_dict['image']
                image = Image.open(os.path.join(self.image_folder,
                                                image_file)).convert('RGB')
                if self.pad_image_to_square:
                    image = expand2square(
                        image,
                        tuple(
                            int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                data_dict['image_pixel_values'] = image
            # else:
            #     crop_size = self.image_processor.crop_size
            #     data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
            #                                             crop_size['width'])
            elif data_dict.get('video', None) is not None and data_dict['video'] != '':
                video_file = data_dict['video']
                video_decode_backend = 'decord'
                num_frames = 10
                video = load_and_transform_video(os.path.join(self.video_folder,video_file), 
                                                get_video_transform(video_decode_backend=video_decode_backend,num_frames=num_frames),
                                                video_decode_backend=video_decode_backend,
                                                num_frames=num_frames)
                # print(f'success get video')
                data_dict['video_pixel_values'] = video
            else:
                raise KeyError(f'no video and image for {index} {data_dict}')
            # else:
            #     print(f'{index} no image and video, using all zero image pixel')
            #     crop_size = self.image_processor.crop_size
            #     data_dict['image_pixel_values'] = torch.zeros(3, crop_size['height'],
            #                                             crop_size['width'])
            
            return data_dict
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__()-1))
