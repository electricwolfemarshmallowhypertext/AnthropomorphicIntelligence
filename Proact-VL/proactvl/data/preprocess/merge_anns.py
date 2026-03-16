'''
merge anns into one file, final annotation format:
{
    "video_path": str,
    "video_duration": float,
    "speakers": {
        speaker_name: {
            "persona": list[str],
        },
    },
    "metadata": {
        "tag": str,
        "dataset_name": str,
    },
    "annotations": [
        {
            "query": str, (optional)
            "speaker": str,
            "start": float,
            "end": float,
            "text": str,
        },
    ]
}
'''

import json
import os
import ffmpeg
import math
import argparse
from tqdm import tqdm


VIDEO_DIR_MAP = {
    'baldurs_gate_3': 'game_commentary/Baldurs_Gate_3/videos',
    'csgo': 'game_commentary/CSGO/videos',
    'cyberpunk_2077': 'game_commentary/Cyberpunk_2077/videos',
    'elden_ring': 'game_commentary/Elden_Ring/videos',
    'lol': 'game_commentary/LOL/videos',
    'minecraft': 'game_commentary/Minecraft/videos',
    'starcraft2': 'game_commentary/Starcraft2/videos',
    'streetfighter6': 'game_commentary/Streetfighter6/videos',
    'tears_of_the_kingdom': 'game_commentary/Tears_of_the_Kingdom/videos',
    'yu_gi_oh': 'game_commentary/Yu_Gi_Oh/videos',
    'livecc': 'live_sft/videos',
    'ego4d_goal_step': 'ego4d/v2/full_scale',
    'black_myth_wukong': 'game_commentary/Black_Myth_Wukong/videos',
}

PERSONA = [
    'Tone: Neutral and adaptive\nVocabulary: Unrestricted\nRhythm & Pacing: Context-driven',
    'The assistant operates as a default persona without stylistic constraints, allowing responses to emerge naturally from the interaction.',
    'Tone: Unspecified\nVocabulary: Open-ended\nRhythm & Pacing: Variable',
    'The assistant generates responses freely based on input and context, without enforcing any predefined style or expression pattern.',
    'Tone: Neutral\nVocabulary: Free-form\nRhythm & Pacing: Adaptive',
    'The assistant maintains no fixed delivery style, adapting content and structure dynamically as the interaction unfolds.',
    'Tone: Flexible\nVocabulary: General-purpose\nRhythm & Pacing: Naturally varying',
    'The assistant functions as an unconstrained baseline persona, prioritizing relevance while leaving expressive choices open.'
]

def time2seconds(time_str):
    """Convert time string 'HH:MM:SS' to total seconds."""
    m, s = map(int, time_str.split(':'))
    return m * 60 + s

import re
import unicodedata
from typing import Iterable, Tuple, List, Union


_LEET_MAP = str.maketrans({
    "@": "a", "$": "s", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
})


_SINGLE_LETTER_GAP = re.compile(r"(?<=\b[a-z])\s+(?=[a-z]\b)")

def _normalize_en(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"[\u200b-\u200f\uFEFF]", "", s)  # zero-width
    s = s.translate(_LEET_MAP)

    # 常见连接符/点 -> 空格（便于识别 f-u-c-k / f.u.c.k）
    s = re.sub(r"[_\-\u2010-\u2015·•.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # 压缩“单字母拆分”的脏话写法：f u c k -> fuck
    # 多跑几次以覆盖更长序列
    for _ in range(5):
        new_s = _SINGLE_LETTER_GAP.sub("", s)
        if new_s == s:
            break
        s = new_s

    return s

def _loose_seq_regex_no_space(seq: str, repeat_max: int = 3) -> str:
    """
    允许插入符号，但不允许插入空格（防止跨词误判）。
    """
    seq = re.sub(r"[^a-z0-9]", "", seq.lower())
    between = r"[^a-z0-9\s]*"  # ✅ 不包含空格
    parts = [rf"{re.escape(ch)}{{1,{repeat_max}}}" for ch in seq]
    return between.join(parts)

def _make_loose_inflection_pattern(stem: str, suffixes: Iterable[str], repeat_max: int = 3) -> re.Pattern:
    alts = [_loose_seq_regex_no_space(stem + suf, repeat_max=repeat_max) for suf in suffixes]
    return re.compile(rf"(^|[^a-z0-9])(?:{'|'.join(alts)})($|[^a-z0-9])", re.IGNORECASE)

def build_english_profanity_patterns() -> List[re.Pattern]:
    patterns: List[re.Pattern] = []
    patterns.append(re.compile(r"\b(wtf|stfu|omfg|fml)\b", re.IGNORECASE))

    INFLECT = ["", "s", "es", "ed", "ing", "er", "ers", "y", "ier", "iest", "iness"]

    stems = [
        "fuck",
        "shit",
        "shite",
        "bitch",
        "asshole",
        "bastard",
        "dick",
        "pussy",
        "cunt",
        "motherfucker",
        "bullshit",
    ]
    for st in stems:
        patterns.append(_make_loose_inflection_pattern(st, INFLECT, repeat_max=3))

    return patterns

def is_clean_text(
    text: Union[str, None],
    return_match: bool = False,
) -> Union[bool, Tuple[bool, str]]:
    if text is None:
        return (True, "") if return_match else True

    s = _normalize_en(str(text))
    for pat in build_english_profanity_patterns():
        m = pat.search(s)
        if m:
            return (False, m.group(0)) if return_match else False
    return (True, "") if return_match else True

'''
# Traverse all the JSON files, merge the "speaker" fields, and smooth the timestamps.
'''
def merge_into_one_file_for_custom_annotations(dataset_name, ann_dir, video_dir, output_file, tag=''):
    ann_list = []
    for root, dirs, files in os.walk(ann_dir):
        for file in files:
            if (not file.endswith('.json')) or 'role_metadata' in file:
                continue
            print(f"[Info]: Processing annotation file: {file}")
            video_filename = file.replace('.json', '.mp4')
            video_rel_path = os.path.relpath(root, ann_dir)
            print(f"[Info]: Video relative path: {video_rel_path}, video filename: {video_filename}")
            if video_rel_path == '.':
                video_path = os.path.join(video_dir, video_filename)
            else:
                video_path = os.path.join(video_dir, video_rel_path, video_filename)
            try:
                probe = ffmpeg.probe(video_path)
                video_duration = float(probe['format']['duration'])
            except Exception as e:
                print(f"[Warning]: Failed to probe video file: {video_path}, error: {e}")
                continue
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                anns = json.load(f)
            # Initialize the new annotation dictionary
            new_ann = {
                'video_path': os.path.join(video_rel_path, video_filename) if video_rel_path != '.' else video_filename,
                'video_duration': video_duration,
                'speakers':{},
                'metadata': {'tag': tag, 'dataset_name': dataset_name},
                'annotations': []
            }
            new_ann['video_path'] = os.path.join(VIDEO_DIR_MAP[dataset_name], new_ann['video_path'])
            # Add speakers and annotations to the new annotation dictionary
            for speaker_ann in anns:
                speaker_name = speaker_ann['Speaker']
                persona = speaker_ann.get('Persona', PERSONA)
                assert speaker_name not in new_ann['speakers'].keys(), f"Duplicate speaker {speaker_name} in file {file}"
                new_ann['speakers'][speaker_name] = {'persona': persona}
                for conversation in speaker_ann['Conversation']:
                    conversation['speaker'] = speaker_name
                    start = conversation['start']
                    if start < 0 or type(start) not in [int, float]:
                        raise ValueError(f"Invalid start time {start} in file {file}")
                    end = conversation['end']
                    if type(end) not in [int, float] or end < 0:
                        raise ValueError(f"Invalid end time {end} in file {file}")
                    text = conversation['text']
                    if type(text) is not str:
                        # invalid text, skip this conversation
                        print(f"[Warning]: Invalid text {text} in file {file}, convert to string")
                        continue
                    # 如果有脏话，则不添加到最终的annotation中
                    if not is_clean_text(text):
                        print(f"[Warning]: Detected profanity in text: {text} in file {file}, skipping this annotation.")
                        continue
                    new_ann['annotations'].append(conversation)
            new_ann['annotations'].sort(key=lambda x: x['start'])
            # smooth the timestamps
            print(f"[info] Smoothing timestamps for file: {file}")
            for i in range(1, len(new_ann['annotations'])):
                prev_ann = new_ann['annotations'][i-1]
                curr_ann = new_ann['annotations'][i]
                prev_end = prev_ann['end']
                curr_start = curr_ann['start']
                if prev_end > curr_start:
                    avg_time = (prev_end + curr_start) / 2
                    prev_ann['end'] = avg_time
                    curr_ann['start'] = avg_time
            ann_list.append(new_ann)
    # Write the merged annotations to the output file
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[Info]: Writing merged annotations to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for ann in ann_list:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')

def merge_into_one_file_for_guidance_annotations(dataset_name, ann_dir, video_dir, output_file, tag=''):
    ann_list = []
    for root, dirs, files in os.walk(ann_dir):
        for file in files:
            if (not file.endswith('.json')) or 'role_metadata' in file:
                continue
            print(f"[Info]: Processing annotation file: {file}")
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                anns = json.load(f)
            for ann in anns:
                video_filename = ann['video_path']
                if not video_filename.endswith('.mp4'):
                    video_filename = video_filename.split('.')[0] + '.mp4'
                video_rel_path = os.path.relpath(root, ann_dir)

                if video_rel_path == '.':
                    video_path = os.path.join(video_dir, video_filename)
                else:
                    video_path = os.path.join(video_dir, video_rel_path, video_filename)
                try:
                    probe = ffmpeg.probe(video_path)
                    video_duration = float(probe['format']['duration'])
                except Exception as e:
                    print(f"[Warning]: Failed to probe video file: {video_path}, error: {e}")
                    continue
                new_ann = {
                    'video_path': os.path.join(video_rel_path, video_filename) if video_rel_path != '.' else video_filename,
                    'video_duration': video_duration,
                    'speakers':{},
                    'metadata': {'tag': tag, 'dataset_name': dataset_name},
                    'annotations': []
                }
                new_ann['video_path'] = os.path.join(VIDEO_DIR_MAP[dataset_name], new_ann['video_path'])
                # only one speaker for guidance
                speaker_names = ann['speakers']
                for speaker_name in speaker_names:
                    new_ann['speakers'][speaker_name] = {'persona': PERSONA}
                for conversation in ann['annotations']:
                    query = conversation['query']
                    sub_annotations = conversation['sub_annotations']
                    new_sub_annotations = {
                        'query': query,
                        'speaker': speaker_names[0],
                        'start': time2seconds(sub_annotations[0]['begin_time']),
                        'end': time2seconds(sub_annotations[0]['end_time']),
                        'text': sub_annotations[0]['commentary']
                    }
                    check_new_ann(new_sub_annotations)
                    new_ann['annotations'].append(new_sub_annotations)
                    for sub_annotation in sub_annotations[1:]:
                        new_sub_annotations = {
                            'speaker': speaker_names[0],
                            'start': time2seconds(sub_annotation['begin_time']),
                            'end': time2seconds(sub_annotation['end_time']),
                            'text': sub_annotation['commentary']
                        }
                        check_new_ann(new_sub_annotations)
                        new_ann['annotations'].append(new_sub_annotations)
                new_ann['annotations'].sort(key=lambda x: x['start'])
                ann_list.append(new_ann)
    print(f"[Info]: Total merged annotations: {len(ann_list)}")
    # split into train, val and test
    train_split = 0.8
    val_split = 0.1
    train_size = 80
    val_size = 8
    train_anns = ann_list[:train_size]
    val_anns = ann_list[train_size:train_size+val_size]
    test_anns = ann_list[train_size+val_size:]
    print(f"[Info]: Train annotations: {len(train_anns)}, Val annotations: {len(val_anns)}, Test annotations: {len(test_anns)}")
    train_file = output_file.replace('.jsonl', '_train.jsonl')
    val_file = output_file.replace('.jsonl', '_val.jsonl')
    test_file = output_file.replace('.jsonl', '_test.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for ann in train_anns:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    with open(val_file, 'w', encoding='utf-8') as f:
        for ann in val_anns:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    with open(test_file, 'w', encoding='utf-8') as f:
        for ann in test_anns:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    print(f"[Info]: Finished writing merged annotations to {train_file}, {val_file}, {test_file}")

            
def merge_into_one_file_for_ego4d_annotations(dataset_name, ann_dir, video_dir, output_file, tag=''):
    ann_list = []
    for root, dirs, files in os.walk(ann_dir):
        for file in files:
            if (not file.endswith('.json')) or 'role_metadata' in file:
                continue
            print(f"[Info]: Processing annotation file: {file}")
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                anns = json.load(f)
            for ann in tqdm(anns):
                video_filename = ann['video_path']
                video_rel_path = os.path.relpath(root, ann_dir)
                video_begin = ann['video_begin']
                video_end = ann['video_end']
                duration = ann['duration']

                video_path = os.path.join(video_dir, video_rel_path, video_filename)
                # prob = ffmpeg.probe(video_path)
                new_ann = {
                    'video_path': os.path.join(video_rel_path, video_filename) if video_rel_path != '.' else video_filename,
                    'video_duration': video_end,
                    'speakers':{},
                    'metadata': {'tag': tag, 'dataset_name': dataset_name},
                    'annotations': []
                }
                new_ann['video_path'] = os.path.join(VIDEO_DIR_MAP[dataset_name], new_ann['video_path'])
                # only one speaker for ego4d goal step
                speaker_name = ann['speakers']
                persona = ann.get('persona', PERSONA)
                new_ann['speakers'][speaker_name] = {'persona': persona}
                for conversation in ann['annotations']:
                    query = conversation['query']
                    sub_annotations = conversation['sub_annotations']
                    check_ann(sub_annotations[0])
                    new_ann['annotations'].append({
                        'query': query,
                        'speaker': speaker_name,
                        'start': sub_annotations[0]['begin_time'],
                        'end': sub_annotations[0]['end_time'],
                        'text': sub_annotations[0]['commentary']
                    })
                    for sub_annotation in sub_annotations[1:]:
                        check_ann(sub_annotation)
                        new_ann['annotations'].append({
                            'speaker': speaker_name,
                            'start': sub_annotation['begin_time'],
                            'end': sub_annotation['end_time'],
                            'text': sub_annotation['commentary']
                        })
                new_ann['annotations'].sort(key=lambda x: x['start'])
                ann_list.append(new_ann)
    # Write the merged annotations to the output file
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[Info]: Writing merged annotations to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for ann in ann_list:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    print(f"[Info]: Finished writing {len(ann_list)} annotations to {output_file}")

def check_ann(ann):
    if 'begin_time' not in ann or 'end_time' not in ann or 'commentary' not in ann:
        raise ValueError(f"Missing fields in annotation in file {ann}")
    if type(ann['begin_time']) not in [int, float] or type(ann['end_time']) not in [int, float]:
        raise ValueError(f"Invalid begin_time in file {ann}")
    if type(ann['commentary']) is not str or ann['commentary'].strip() == '':
        raise ValueError(f"Invalid commentary in file {ann}")
    if ann['begin_time'] < 0 or ann['end_time'] < 0:
        raise ValueError(f"Invalid timestamps in file {ann}")

def check_new_ann(ann):
    if 'start' not in ann or 'end' not in ann or 'text' not in ann:
        raise ValueError(f"Missing fields in annotation in file {ann}")
    if type(ann['start']) not in [int, float] or type(ann['end']) not in [int, float]:
        raise ValueError(f"Invalid start time in file {ann}")
    if type(ann['text']) is not str or ann['text'].strip() == '':
        raise ValueError(f"Invalid text in file {ann}")
    if ann['start'] < 0 or ann['end'] < 0:
        raise ValueError(f"Invalid timestamps in file {ann}")



def main():
    args = argparse.ArgumentParser()
    args.add_argument("--ann_dir", type=str, required=True, help="path to the annotation directory")
    args.add_argument("--video_dir", type=str, required=True, help="path to the video directory")
    args.add_argument("--dataset_name", type=str, required=True, help="name of the dataset")
    args.add_argument("--output_file", type=str, required=True, help="path to the output merged annotation file")
    args.add_argument("--tag", type=str, default='', help="tag to add to metadata")
    args.add_argument('--dataset_type_list', nargs='+', default=['train', 'val'], help='only useful for commentary game dataset')
    args = args.parse_args()

    print(f"[Merge anns]: Merging {args.dataset_name} annotations from directory: {args.ann_dir}")
    custom_game_list = [
        'cyberpunk_2077', 'starcraft2', 'baldurs_gate_3', 'black_myth_wukong', 'elden_ring',
        'tears_of_the_kingdom', 'yu_gi_oh', 'lol', 'csgo', 'streetfighter6'
    ]

    if args.dataset_name in custom_game_list:

        for dataset_type in args.dataset_type_list:
            if dataset_type == 'train':
                # train
                train_save_path = args.output_file.replace('.jsonl', '_train.jsonl')
                merge_into_one_file_for_custom_annotations(args.dataset_name, os.path.join(args.ann_dir, 'train'), args.video_dir, train_save_path, args.tag)
            elif dataset_type == 'val':
                # val
                val_save_path = args.output_file.replace('.jsonl', '_val.jsonl')
                merge_into_one_file_for_custom_annotations(args.dataset_name, os.path.join(args.ann_dir, 'val'), args.video_dir, val_save_path, args.tag)
            elif dataset_type == 'test':
                # test
                test_save_path = args.output_file.replace('.jsonl', '_test.jsonl')
                merge_into_one_file_for_custom_annotations(args.dataset_name, os.path.join(args.ann_dir, 'test'), args.video_dir, test_save_path, args.tag)
    elif args.dataset_name in ['minecraft', 'genshin_impact']:
        merge_into_one_file_for_guidance_annotations(args.dataset_name, args.ann_dir, args.video_dir, args.output_file, args.tag)
    elif args.dataset_name in ['ego4d_goal_step']:
        # train
        train_save_path = args.output_file.replace('.jsonl', '_train.jsonl')
        merge_into_one_file_for_ego4d_annotations(args.dataset_name, os.path.join(args.ann_dir, 'train'), args.video_dir, train_save_path, args.tag)
        # val
        val_save_path = args.output_file.replace('.jsonl', '_val.jsonl')
        merge_into_one_file_for_ego4d_annotations(args.dataset_name, os.path.join(args.ann_dir, 'val'), args.video_dir, val_save_path, args.tag)
    # elif args.dataset_name in ['livecc']:
    #     merge_into_one_file_for_livecc_annotations(args.dataset_name, args.ann_dir, args.video_dir, args.output_file, args.tag)
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

if __name__ == "__main__":
    if False:
        import debugpy
        try:
            debugpy.listen(('localhost', 9501))
            print(f'debug listen on port 9501')
            debugpy.wait_for_client()
        except Exception as e:
            raise RuntimeError(f"Failed to start debugpy: {e}")
    main()
