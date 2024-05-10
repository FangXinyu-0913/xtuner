# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


def llava_image_only_map_fn(example):
    # input contains the DEFAULT_IMAGE_TOKEN only
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            assert DEFAULT_IMAGE_TOKEN in msg['value']
            input += DEFAULT_IMAGE_TOKEN
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


def llava_map_fn(example):
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            elif DEFAULT_VIDEO_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_VIDEO_TOKEN,
                                                    '').strip()
                msg['value'] = DEFAULT_VIDEO_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


def llava_videochat_sft_map_fn(example):
    messages = example['QA']
    conversation = []
    for msg in messages:
        msg['q'] = msg['q'].replace(DEFAULT_VIDEO_TOKEN,'').strip()
        msg['q'] = DEFAULT_VIDEO_TOKEN + '\n' + msg['q']
        msg['q'] = msg['q'].strip()
        conversation.append({'input': msg['q'], 'output':msg['a']})
    return {'conversation': conversation}
