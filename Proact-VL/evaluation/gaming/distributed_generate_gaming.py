import os
import json
import re
import tqdm
import shutil
import argparse
import multiprocessing
from functools import partial
from datasets import load_dataset


from proactvl.data.custom_commentary_dataset import construct_conversation_prompt
from proactvl.utils.conversations import construct_val_system_prompt
from proactvl.utils.multiprocessor import local_mp
from proactvl.infer.multi_assistant_inference import MultiAssistantStreamInference
from proactvl.infer.video_reader import VideoReader

MAX_VIDEO_PIXELS = 540*36*28*28
MIN_PIXELS = 128*28*28
MAX_PIXELS = 540 * 28 * 28


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed Gaming generation over the Gaming dataset"
    )
    parser.add_argument(
        "--model_id", type=str, default=None,
        help="Model identifier"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default='Qwen/Qwen2.5-Omni-7B',
        help="HuggingFace model path, e.g., chenjoya/Gaming-7B-Instruct"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel processes/gpus to use"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="evaluation/gaming/gaming",
        help="Directory to write generated JSON outputs"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--state_threshold", type=float, default=0.5,
        help="Threshold for state detection"
    )
    parser.add_argument(
        "--max_kv_tokens", type=int, default=16384,
        help="Max key-value tokens for model"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="Dataset name"
    )
    parser.add_argument(
        '--test_name', type=str, default='final_val',
        help='Which validation split to use'
    )
    parser.add_argument(
        '--video_dir', type=str, default=None,
        help='Custom video directory if needed'
    )
    return parser.parse_args() 

def prepare_model_for_inference(args, device_id):
    model_config = None
    ckpt_path = args.ckpt_path

    infer_config = {
        'use_audio_in_video': False,
        'max_kv_tokens': args.max_kv_tokens,
        'assistant_num': 1,
        'enable_tts': False,
    }
    generate_config = {
        'do_sample': True,
        'max_new_tokens': 12,
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.15,
    }

    infer = MultiAssistantStreamInference(model_config, ckpt_path, infer_config, generate_config, None, f'cuda:{device_id}')
    infer.assistants[0].set_threshold(args.state_threshold)
    return infer


def gaming_worker(
    device_id: int,
    save_dir: str,
    num_workers: int,
    args
):
    stream_infer = prepare_model_for_inference(args, device_id)
    ds = load_dataset(args.dataset_name, name=args.test_name, split='test')

    idxs = list(range(len(ds)))
    
    idxs_on_device = idxs[device_id::num_workers]
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm.tqdm(idxs_on_device, desc=f"Device {device_id}", total=len(idxs_on_device)):
        save_path = os.path.join(save_dir, f"{idx}.json")
        if os.path.exists(save_path):
            continue

        record = ds[idx]
        video_path = os.path.join(os.path.expanduser(args.video_dir), record['video_path'])
        record['video_path'] = video_path
        duration = record['video_duration']
        video_begin = record['video_begin']
        video_end = record['video_end']
        anns = record['annotations']
        tag = record['tag']
        dataset_name = record['dataset_name']
        data_uid = record['idx']
        history = record.get('history', '')
        ann_map = {}
        for i in range(duration):
            ann_map[video_begin + i] = None
        for one_ann in anns:
            if one_ann['role'] == 'user':
                begin_time = one_ann['start']
                ann_map[begin_time] = one_ann

        system_prompt = construct_val_system_prompt(dataset_name, tag, record['active_speaker']['persona'])
        # system_prompt = record['system_prompt']
        stream_infer.assistants[0].clear_session()
        stream_infer.assistants[0].prime_system_prompt(system_prompt)
        
        print(f'Duration: {duration}, begin: {video_begin}, end: {video_end}')
        overall_cc = {}
        record['system_prompt'] = system_prompt
        stream_infer.register_video_reader(video_path, video_begin, video_end)
        for i in range(duration):
            
            current_ann = ann_map.get(video_begin + i, None)
            current_second = video_begin + i
            prefix = f"[Sec: {current_second} to {current_second + 1}] "

            # history = ''
            user_query = ''
            context = ''
            if i != 0:
                history = ''
            # print(f'{prefix} Current annotation: {current_ann}')
            if current_ann is not None:
                if current_ann['role'] == 'user':
                    if current_ann['speaker'] == 'user':
                        user_query = current_ann['query']
                    else:
                        context = f'[{current_ann["speaker"]}]: {current_ann["text"]}'

            # print(f'{prefix} history: {history}, user_query: {user_query}, context: {context}')
            assistant_responses, _  = stream_infer.infer_one_chunk(current_second, history=history, user_query=user_query, previous_responses=context)
            if assistant_responses is not None and assistant_responses[0].active:
                overall_cc[current_second] = assistant_responses[0].commentary.strip() if assistant_responses is not None else ''
                print(f'{prefix} Generated commentary: {overall_cc[current_second]}')

        with open(save_path, 'w') as wf:
            json.dump({
                "video_id": video_path.split('/')[-1],
                "begin": video_begin,
                "end": video_end,
                "pred": overall_cc,
                # "label": overall_label,
                "tag": tag,
                'dataset_name': dataset_name,
                'idx': data_uid,
            }, wf)


if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method('spawn', force=True)

    save_dir = os.path.join(os.path.expanduser(args.output_dir), f'{args.model_id}_{int(args.state_threshold*100)}_{args.max_kv_tokens}')
    worker_fn = partial(
        gaming_worker,
        save_dir=save_dir,
        num_workers=args.num_workers,
        args=args
    )
    local_mp(
        list(range(args.num_workers)),
        worker_fn,
        desc="gaming_generation",
        num_workers=args.num_workers
    )
    
    save_path = save_dir + '.jsonl'
    with open(save_path, 'w') as wf:
        for file in os.listdir(save_dir):
            print(file)
            datum = json.load(open(os.path.join(save_dir, file))) 
            wf.write(json.dumps(datum) + '\n')
    # remove save_dir
    shutil.rmtree(save_dir)
