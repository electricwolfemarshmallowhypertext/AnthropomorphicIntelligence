import os, openai, json, functools, argparse
from datasets import load_dataset
from proactvl.utils.multiprocessor import local_mt


# ========================= Initialize GPT Client =========================
try:
    from evaluation.gaming.gpt_client_utils import GPTClientFactory
except ImportError:
    from gpt_client_utils import GPTClientFactory

# deployment_name = "qwen/qwen3.6-plus"
deployment_name = "gpt-5.1_2025-11-13"
try:
    gpt = GPTClientFactory().build()
except Exception as e:
    raise RuntimeError(f"Error: failed to initialize GPT client. {e}") from e

def judge_ab(a_id_with_pred, b_id_with_pred, gt_asr):
    a_id, a_pred = a_id_with_pred
    b_id, b_pred = b_id_with_pred
    ab_prompt = (
        'You are an expert in video commentary. '
        'Your task is to review two commentaries (Commentary A and Commentary B), and select the one that better aligns with the human commentary. '
        'You should consider the criteria:\n'
        '1. Semantic Alignment: The commentary should convey the same meaning, details, and key points as the human commentary.\n'
        'If the above criteria is not enough to judge, then consider:\n'
        '2. Stylistic Consistency: The commentary should maintain a tone, word choice, and structure similar to the human commentary.\n'
        f'\n---Commentary A---\n{a_pred}\n----------\n'
        f'\n---Commentary B---\n{b_pred}\n----------\n'
        f'\n---Human Commentary---\n{gt_asr}\n----------\n'
        '\nYour response should be "Commentary A is better aligned with the human commentary" or "Commentary B is better aligned with the human commentary".\n'
    )
    while True:
        try:
            ab_resp = gpt.chat.completions.create(
                # model='gpt-4o-2024-08-06',
                # model='gpt-4.1',
                model=deployment_name,
                messages=[{"role": "user", "content": [{'type': 'text', 'text': ab_prompt}]}],
                seed=42,
                temperature=0,
            ).choices[0].message.content
            break
        except Exception as e:
            print('Failed to get response...', e)
    if 'Commentary A' in ab_resp:
        ab_winner = a_id
    elif 'Commentary B' in ab_resp:
        ab_winner = b_id
    else:
        ab_winner = 'tie'
    # print(ab_winner)
    return ab_winner


def judge(item, model_id):
    # print('--------call judge once--------')
    video_event_id, model_pred = item
    gt_asr = video_event_id_to_gt_asr[video_event_id]
    baseline_pred = video_event_id_to_baseline_pred[video_event_id]
    dataset_name = video_event_id_to_dataset_name[video_event_id]
    tag = video_event_id_to_tag[video_event_id]
    return {
        'video_event_id': video_event_id, 
        'ab_winner': judge_ab([model_id, model_pred], [baseline_id, baseline_pred], gt_asr), 
        'ba_winner': judge_ab([baseline_id, baseline_pred], [model_id, model_pred], gt_asr),
        'dataset_name': dataset_name,
        'tag': tag
    }

def merge_commentary(commentary_dict):
    sorted_seconds = sorted(commentary_dict.keys(), key=int)
    merged_commentary = ' '.join([commentary_dict[sec].replace(' ...', '') for sec in sorted_seconds])
    return merged_commentary.strip()


def calculate_proactive_metrics(label_dict, pred_dict, video_duration):
    label_events = set(label_dict.keys())
    pred_events = set(pred_dict.keys())

    true_positives = len(label_events.intersection(pred_events))
    false_positives = len(pred_events - label_events)
    false_negatives = len(label_events - pred_events)

    proactive_rate = true_positives / video_duration
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return proactive_rate, precision, recall


def pred2standard(pred):
    if type(pred) == str:
        return pred, pred
    sorted_seconds = sorted(pred.keys(), key=int)
    # print(sorted_seconds)

    asrs = []
    asr_lines = []
    for sec in sorted_seconds:

        line = pred[sec]
        # Trim trailing ' ...'
        if line.endswith(' ...'):
            line = line[:-4]
        if line.strip() == '':
            continue
        else:
            asr_lines.append(f'[t={sec}] {line}')
            asrs.append(line)
    return '\n'.join(asr_lines), ' '.join(asrs)

def gt2standard(annotations):
    active_speaker = annotations['active_speaker']['name']
    asrs = []
    asr_lines = []
    for item in annotations['annotations']:
        # if item['speaker'] != active_speaker:
        #     continue
        sec = item['start']
        line = item['text'] if 'text' in item else item['query']

        if line.strip() == '':
            continue
        else:
            if item['speaker'] == active_speaker:
                asrs.append(line)
                asr_lines.append(f'[t={sec}][ASSISTANT]:{line}')
            elif item['speaker'] == 'user':
                asr_lines.append(f'[t={sec}][USER]:{line}')
            else:
                asr_lines.append(f'[t={sec}][{item["speaker"]}]:{line}')
            
    return '\n'.join(asr_lines), ' '.join(asrs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model name to compare against baseline')
    parser.add_argument('--prediction_jsonl', type=str, required=True, help='Path to model predictions in JSONL format')
    parser.add_argument('--output_dir', type=str, default='evaluation/gaming/judges/', help='Directory to save judgment results')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--baseline_id', type=str, default='gpt-4o-3', help='Baseline model name')
    parser.add_argument('--baseline_jsonl', type=str, default='evaluation/gaming/captions/gpt-4o-3-caption.jsonl', help='Path to baseline predictions in JSONL format')
    parser.add_argument('--asr_jsonl', type=str, default=None, help='Path to ASR transcriptions in JSONL format')
    parser.add_argument('--max_video_length', type=int, default=86400, help='Maximum video length to consider for evaluation (in seconds)')
    # parser.add_argument('--dataset_name_list', type=str, nargs='+', default=['lol'], help='List of dataset names to evaluate on')
    args = parser.parse_args()

    model_id = args.model_id
    prediction_jsonl = args.prediction_jsonl
    output_dir = args.output_dir

    baseline_id = args.baseline_id
    baseline_jsonl = args.baseline_jsonl

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{baseline_id}_{model_id}.jsonl')

    print(f'{model_id} vs. {baseline_id}')

    # prediction_jsonl = 'evaluation/gaming/gaming/Qwen2.5-Omni-7B.jsonl'
    prediction_jsonl = args.prediction_jsonl
    asr_jsonl = args.asr_jsonl
    if args.asr_jsonl is None:
        asr_jsonl = prediction_jsonl
    else:
        asr_jsonl = args.asr_jsonl
    video_event_id_to_baseline_pred, video_event_id_to_gt_asr, video_start_end_to_video_event_id, video_event_id_to_model_pred = {}, {}, {}, {}
    video_event_id_to_dataset_name = {}
    video_event_id_to_tag = {}

    for line in open(baseline_jsonl):
        datum = json.loads(line)
        dataset_name = datum['dataset_name']
        if datum['end'] > args.max_video_length:
            continue
        # if datum['dataset_name'] not in args.dataset_name_list:
        #     continue
        video_event_id = f'{datum["video_id"]}_{datum["begin"]}_{datum["end"]}_{dataset_name}_{datum["idx"]}'
        video_start_end = (datum['video_id'], datum['begin'], datum['end'])
        video_start_end_to_video_event_id[video_start_end] = video_event_id
        video_event_id_to_baseline_pred[video_event_id] = datum['pred']
        video_event_id_to_dataset_name[video_event_id] = datum['dataset_name']
        video_event_id_to_tag[video_event_id] = datum.get('tag', None)

    for line in open(prediction_jsonl):
        datum = json.loads(line)
        dataset_name = datum['dataset_name']
        if datum['end'] > args.max_video_length:
            continue
        # if datum['tag'] not in ['Solo commentators', "Multiple commentators", 'Guidance']:
        #     continue
        # if datum['dataset_name'] not in args.dataset_name_list:
        #     continue
        video_event_id = f'{datum["video_id"]}_{datum["begin"]}_{datum["end"]}_{dataset_name}_{datum["idx"]}'
        if video_event_id in video_event_id_to_model_pred:
            print(f'Warning: duplicate video_event_id {video_event_id} in predictions')
        video_event_id_to_model_pred[video_event_id] = pred2standard(datum['pred'])[1]
        # if type(datum['pred']) == str:
        #     video_event_id_to_model_pred[video_event_id] = datum['pred']
        # else:
        #     video_event_id_to_model_pred[video_event_id] = merge_commentary(datum['pred'])

    for line in open(asr_jsonl):
        datum = json.loads(line)
        if datum['video_end'] > args.max_video_length:
            continue
        video_event_id = f'{os.path.basename(datum["video_path"])}_{datum["video_begin"]}_{datum["video_end"]}_{datum["dataset_name"]}_{datum["idx"]}'
        # video_event_id_to_gt_asr[video_event_id] = merge_commentary(datum['label'])
        video_event_id_to_gt_asr[video_event_id] = gt2standard(datum)[1]

    winner_results = local_mt(
        video_event_id_to_model_pred.items(), 
        functools.partial(judge, model_id=model_id), 
        desc=f'Call gpt-5.1_2025-11-13 for {model_id} vs. {baseline_id}', 
        num_workers=args.num_workers
    )
    
    with open(save_path, 'w') as f:
        for winner_result in winner_results:
            f.write(json.dumps(winner_result) + '\n')
    
    # Report win rates by game, by tag, and overall.
    summary_by_game = {}
    for winner_result in winner_results:
        dataset_name = winner_result['dataset_name']
        if dataset_name not in summary_by_game:
            summary_by_game[dataset_name] = {'win_count': 0, 'count': 0}
        if winner_result['ab_winner'] == model_id:
            summary_by_game[dataset_name]['win_count'] += 1
        if winner_result['ba_winner'] == model_id:
            summary_by_game[dataset_name]['win_count'] += 1
        summary_by_game[dataset_name]['count'] += 2
    for dataset_name, summary in summary_by_game.items():
        win_rate = summary['win_count'] / summary['count'] * 100
        output = f'Winning Rate for {model_id} vs. {baseline_id} on {dataset_name}: {win_rate:.2f}%'
        print(output)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(output + '\n')

    summary_by_tag = {}
    for winner_result in winner_results:
        tag = winner_result['tag']
        if tag not in summary_by_tag:
            summary_by_tag[tag] = {'win_count': 0, 'count': 0}
        if winner_result['ab_winner'] == model_id:
            summary_by_tag[tag]['win_count'] += 1
        if winner_result['ba_winner'] == model_id:
            summary_by_tag[tag]['win_count'] += 1
        summary_by_tag[tag]['count'] += 2
    for tag, summary in summary_by_tag.items():
        win_rate = summary['win_count'] / summary['count'] * 100
        output = f'Winning Rate for {model_id} vs. {baseline_id} on tag {tag}: {win_rate:.2f}%'
        print(output)
        with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
            f.write(output + '\n')

    win_count, count = 0, 0
    for winner_result in winner_results:
        if winner_result['ab_winner'] == model_id:
            win_count += 1
        if winner_result['ba_winner'] == model_id:
            win_count += 1
        count += 2
    
    win_rate = win_count / count * 100
    output = f'Winning Rate for {model_id} vs. {baseline_id}: {win_rate:.2f}%'
    print(output)
    with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
        f.write(output + '\n')

# python -m evaluation.gaming.llm_judge --model_id qwen25omni_7game_s3156_threshold_50.0 --prediction_jsonl evaluation/gaming/gaming/qwen25omni_7game_s3156_threshold_50.0.jsonl --num_workers 4