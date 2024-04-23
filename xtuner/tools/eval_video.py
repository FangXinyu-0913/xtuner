# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp
from mmengine.config import Config, DictAction
from xtuner.registry import BUILDER
import torch
from transformers import GenerationConfig, StoppingCriteriaList
from mmengine.visualization import Visualizer
import numpy as np
import cv2
import json
from xtuner.utils import StopWordStoppingCriteria
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from mmengine.utils.misc import get_object_from_string
from mmengine.dist import collect_results, get_dist_info, get_rank, init_dist, master_only
from mmengine.utils.dl_utils import set_multi_processing
from xtuner.dataset import VideoEvalDataset
from tqdm import tqdm
# os.environ['RANK'] = '0' #'[0,1,2,3,4,5,6,7]'
# os.environ['WORLD_SIZE'] = '8'
# os.environ['LOCAL_RANK'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description='Inference LLM')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('weighs', help='model weights file name or path.')
    parser.add_argument(
        '--output_dir', default='work_dirs/eval', help='the dir to save logs and models'
    )
    parser.add_argument('--launcher', default='pytorch')
    # parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument('--local-rank',type=int, default=0)
    parser.add_argument('--eval_video_data_path', default=None)
    parser.add_argument('--eval_video_folder', default=None)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.launcher != None:
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()

        torch.cuda.set_device(rank)

    else:
        rank = 0
        world_size = 1

    # eval_video_data_path = '/cpfs01/user/fangxinyu/Video-LLaVA/eval/GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/msrvtt_qa/test.jsonl'
    # eval_video_folder = '/cpfs01/user/fangxinyu/Video-LLaVA/eval/GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/videos/all'
        
    eval_video_data_path = args.eval_video_data_path
    eval_video_folder = args.eval_video_folder


    inference_dataset = dict(
        type=VideoEvalDataset,
        video_data_path=eval_video_data_path,
        video_folder=eval_video_folder,
        tokenizer=cfg.tokenizer,
        image_processor=cfg.image_processor,
        offline_processed_text_folder=None,
        system=cfg.SYSTEM,
        prompt_template=cfg.prompt_template,
        video_frames=cfg.video_frames,
        video_batch_size=cfg.video_batch_size,
        image_batch_size=cfg.image_batch_size,
        resolution=cfg.frame_size,
        pad_image_to_square=True,
    )

    cfg.merge_from_dict({'inference_dataset':inference_dataset})

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # vis = Visualizer()
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)


    cfg.model.pretrained_pth = args.weighs
    cfg.model.llm.quantization_config = None
    model = BUILDER.build(cfg.model)
    model.cuda()

    model.llm.config.use_cache = True
    model.eval()

    tokenizer = BUILDER.build(cfg.tokenizer)



    max_new_tokens = 50
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    stop_criteria = StoppingCriteriaList()
    stop_words = cfg.prompt_template.get('STOP_WORDS', [])
    stop_str = cfg.prompt_template.get('SEP', '')
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))



    dataset = BUILDER.build(cfg.inference_dataset)

    mean = dataset.image_processor.image_mean
    std = dataset.image_processor.image_std
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    eval_dataset_type = eval_video_data_path.split('/')[-2]
    answers_file = os.path.join(args.output_dir,eval_dataset_type)+'.json'

    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    results = []
    for i in tqdm(per_rank_ids, desc=f'Rank {rank}'):
        # for data in tqdm(dataset, total = len(dataset)):
        data = dataset[i]
        # print(data['question'], ':', data['answer'])
        if 'video_id' in data.keys():
            image_file = data['video_id']
        if 'image_pixel_values' in data.keys():
            pixel_values = data['image_pixel_values']
            use_image = True
        elif 'video_pixel_values' in data.keys():
            pixel_values = data['video_pixel_values']
            use_image = False
        new_data = {
            'pixel_values': data['video_pixel_values'].to(model.device)[None].half(),
            'input_ids': torch.tensor(data['input_ids']).to(model.device)[None],
        }
        pred_data = {'id': data['video_id'], 'question': data['question'], 'answer': data['answer']}
        input_token_len = len(data['input_ids'])

        with torch.no_grad():
            bs, f, c, res, res = new_data['pixel_values'].shape
            if use_image:
                assert f == 1
            else:
                assert f == model.video_frames
            origin = new_data['pixel_values'].squeeze(0)

            # if use_image:
            #     mamba_input = origin.transpose(1, 2)
            # else:
            #     mamba_input = origin.transpose(1, 2)

            # print(origin.shape)
            # mamba_input = mamba_input.to(torch.bfloat16)
            visual_outputs = model.visual_encoder(
                origin, output_hidden_states=True)

            pixel_values = model.projector(
                visual_outputs.hidden_states[model.visual_select_layer][:, 1:])

            new_data['pixel_values'] = pixel_values

            mm_inputs = prepare_inputs_labels_for_multimodal(llm=model.llm, video_frames=cfg.video_frames, **new_data)
            generation_output = model.generate(
                **mm_inputs,
                max_new_tokens=max_new_tokens,
                generation_config=gen_config,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria
            )
            # output = tokenizer.decode(generation_output[0, input_token_len:])
            output = tokenizer.decode(generation_output[0])
            if '<s>' in output:
                output = output.replace('<s>','')
            if '</s>' in output:
                output = output.replace('</s>','')
            output = output.strip()
            # print(output)
            pred_data['pred']=output
            # ans_file.write(json.dumps(pred_data) + "\n")
            results.append(pred_data)
            # print(output)
    results = collect_results(results, n_samples)

    if get_rank()==0:
        with open(answers_file, 'w') as f:
            json.dump(results, f)
            print(f'{answers_file} saved!')

        



if __name__ == '__main__':
    main()