'''
Segment the whole dataset into video clips.
split speakers, split text into seconds, split clips.
# Only consider three senarios:
1. Solo commentary by one speaker.
{
    "speaker": str,
    "role": assistant,
    "start" : int,
    "end": int,
    "text": str,
}
2. Q&A between user and one speaker.
{
    "speaker": str,
    "role": assistant,
    "start" : int,
    "end": int,
    "text": str,
}
{
    "query": str, # from user
    "speaker": 'user',
    "role": user,
    "start" : int,
    "end": int,
}
3. Multi-speaker discussion with commentary.
{
    "speaker": str,
    "role": assistant,
    "start" : int,
    "end": int,
    "text": str,
}
commentary from other speakers are considered as user input.
{
    "speaker": str,
    "role": user,
    "start" : int,
    "end": int,
    "text": str,
}
The final annotation format is:
{
    "video_path": str,
    "video_begin": int,
    "video_end": int,
    "video_duration": int,
    "active_speaker": {
        speaker_name: "",
        "persona": list[str],
    },
    "metadata": {
        "tag": str,
        "dataset_name": str,
    },
    "annotations": [
        # solo commentary and commentary from active speaker in multi-speaker discussion
        {
            "role": assistant,
            "speaker": str,
            "start": int,
            "end": int,
            "text": str,
        },
        # commentary from other speakers in multi-speaker discussion
        {
            "role": user,
            "speaker": str,
            "start": int,
            "end": int,
            "text": str,
        },
        # user query in Q&A
        {
            "role": user,
            "speaker": 'user',
            "start": int,
            "end": int,
            "query": str,
        },
    ]
}
'''
import json
import os
import math
import argparse
import random

def split_speakers_user_query(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            speakers = ann['speakers']
            for speaker_name, speaker_info in speakers.items():
                new_ann = {
                    'video_path': ann['video_path'],
                    'video_begin': ann['video_begin'],
                    'video_end': ann['video_end'],
                    'video_duration': ann['video_duration'],
                    'active_speaker': {
                        'name': speaker_name,
                        'persona': speaker_info['persona'],
                    },
                    'metadata': ann['metadata'],
                    'annotations': []
                }
                for segment in ann['annotations']:
                    if 'query' in segment:
                        new_ann['annotations'].append({
                            'query': segment['query'],
                            'speaker': 'user',
                            'role': 'user',
                            'start': segment['start'],
                            'end': segment['start'] + 1,
                        })
                        segment.pop('query')
                    if segment['speaker'] == speaker_name:
                        new_segment = segment.copy()
                        new_segment['role'] = 'assistant'
                        new_ann['annotations'].append(new_segment)
                    else:
                        new_segment = segment.copy()
                        new_segment['role'] = 'user'
                        new_segment['start'] += 1
                        new_segment['end'] += 1
                        new_ann['annotations'].append(new_segment) 
                f_out.write(json.dumps(new_ann, ensure_ascii=False) + '\n')

def split_text_into_segments(text, n_segments):
    # Split text by spaces into n_segments, distributing words as evenly as possible;
    # if there are fewer segments than n_segments, use only the actual number of segments.
    words = text.split()
    # mask = []
    total_words = len(words)
    if total_words == 0:
        return []
    if n_segments <= 0:
        raise ValueError("n_segments must be greater than 0")
    if n_segments > total_words:
        n_segments = total_words
    base_size = total_words // n_segments
    remainder = total_words % n_segments
    segments = []
    start = 0
    for i in range(n_segments):
        end = start + base_size + (1 if i < remainder else 0)

        segments.append(' '.join(words[start:end]))
        start = end
    return segments

def sentence_to_seconds(item):
    new_annotations = []
    if item['role'] == 'user' and item['speaker'] == 'user' and 'query' in item:
        new_annotations.append(item)
        return new_annotations
    text = item['text']
    if not text.strip():
        return new_annotations
    duration = item['end'] - item['start']
    text_chunks = split_text_into_segments(text, duration)
    for i, chunk in enumerate(text_chunks):
        start_time = item['start'] + i
        end_time = start_time + 1
        new_item = {
            'speaker': item['speaker'],
            'role': item['role'],
            'start': start_time,
            'end': end_time,
            'text': chunk,
        }
        new_annotations.append(new_item)
    return new_annotations


def split_text_into_seconds(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            new_annotations = []
            for item in ann['annotations']:
                if item['end'] - item['start'] <= 0:
                    continue
                new_annotations.extend(sentence_to_seconds(item))
            ann['annotations'] = new_annotations
            f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')

# def random_split_clips(input_file, output_file, min_duration, max_duration, history_duration, min_active_rate, max_active_rate):
#     pass

def extract_ann_clip(ann, video_begin, video_end, history_begin, f_out):
    clip_duration = video_end - video_begin
    # randomly select persona
    speaker_name = ann['active_speaker']['name']
    persona_list = ann['active_speaker']['persona']
    chosen_persona = random.choice(persona_list)
    res = {
        'video_path': ann['video_path'],
        'video_begin': video_begin,
        'video_end': video_end,
        'video_duration': clip_duration,
        'active_speaker': {
            'name': speaker_name,
            'persona': chosen_persona,
        },
        'metadata': ann['metadata'],
        'annotations': []
    }
    total_active_duration = 0
    history_texts = []
    for segment in ann['annotations']:
        seg_start = segment['start']
        seg_end = segment['end']
        if seg_start >= history_begin and seg_end <=video_begin:
            # user query
            if segment['role'] == 'user' and segment['speaker'] == 'user' and 'query' in segment:
                history_texts.append({'USER': segment['query']})
            # current assistant previous text
            elif segment['role'] == 'assistant' and segment['speaker'] == speaker_name:
                history_texts.append({'ASSISTANT': segment['text']})
            elif segment['role'] == 'user' and segment['speaker'] != 'user' and segment['speaker'] != speaker_name:
            # other speakers' commentary as user input
                history_texts.append({segment['speaker']: segment['text']})
            else:
                raise ValueError(f'Unexpected segment: {segment}')
        elif seg_start >=video_begin and seg_end <= video_end:
            res['annotations'].append(segment)
            if segment['role'] == 'assistant':
                total_active_duration += 1
        elif seg_start >= video_end:
            break

    # merge history
    if history_texts:
        new_history = ''
        pre_speaker = None
        for ht in history_texts:
            for speaker, text in ht.items():
                if speaker == pre_speaker:
                    new_history += ' ' + text
                else:
                    new_history += '\n' + f'[{speaker}]: ' + text
                    pre_speaker = speaker
        new_history = new_history.strip()
        res['history'] = new_history
    return res, total_active_duration / clip_duration

def split_clips(input_file, output_file, clip_duration, clip_overlap, min_duration, history_duration, min_active_rate, max_active_rate):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            ann = json.loads(line)
            video_begins = list(range(0, ann['video_duration'], clip_duration - clip_overlap))
            for video_begin in video_begins:
                video_end = min(video_begin + clip_duration, ann['video_duration'])
                if video_end - video_begin < min_duration:
                    continue
                history_begin = max(0, video_begin - history_duration)
                res, active_rate = extract_ann_clip(ann, video_begin, video_end, history_begin, f_out)
                if not res:
                    continue
                if active_rate == min_active_rate and active_rate == max_active_rate:
                    # for all silence
                    f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                elif active_rate > min_active_rate and active_rate <= max_active_rate:
                    f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                else:
                    continue
                # if active_rate <= min_active_rate or active_rate > max_active_rate:
                #     continue
                # f_out.write(json.dumps(res, ensure_ascii=False) + '\n')

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--ann_dir', type=str, required=True, help='Path to the directory containing annotation files.')
    # args.add_argument('--random', action='store_true', help='Whether to randomly select clips.')
    args.add_argument('--min_duration', type=int, default=9, help='Minimum duration of clips to keep in seconds.')
    args.add_argument('--max_duration', type=int, default=144, help='Maximum duration of clips to keep in seconds.')
    args.add_argument('--clip_duration', type=int, default=36, help='Duration of each clip in seconds.')
    args.add_argument('--clip_overlap', type=int, default=18, help='Overlap between clips in seconds.')
    args.add_argument('--history_duration', type=int, default=300, help='Maximum duration of clips to keep in seconds.')
    args.add_argument('--min_active_rate', type=float, default=0.3, help='Minimum active speaker rate in each clip.')
    args.add_argument('--max_active_rate', type=float, default=0.7, help='Maximum active speaker rate in each clip.')
    args.add_argument('--dataset_type_list', nargs='+', default=['train', 'val'], help='only useful for commentary game dataset')
    args = args.parse_args()

    for root, dirs, files in os.walk(args.ann_dir):
        for filename in files:
            if filename.endswith('.jsonl') and 'standard_format' in filename and any(dataset_type in filename for dataset_type in args.dataset_type_list):
                file_path = os.path.join(args.ann_dir, filename)
                # split speakers
                file_to_save_split_speakers = file_path.replace('standard_format', 'split_speakers')
                print(f'[Split speakers] Processing {file_path} and saving to {file_to_save_split_speakers}...')
                split_speakers_user_query(file_path, file_to_save_split_speakers)
                # split text into seconds
                file_to_save_split_text = file_to_save_split_speakers.replace('split_speakers', 'split_text')
                print(f'[Split text] Splitting text into seconds and saving to {file_to_save_split_text}...')
                split_text_into_seconds(file_to_save_split_speakers, file_to_save_split_text)
                # split clips
                file_to_save_split_clips = file_to_save_split_text.replace('split_text', f'split_clips_{args.clip_duration}s_overlap{args.clip_overlap}s_{args.min_active_rate}-{args.max_active_rate}')
                print(f'[Split clip] Splitting clips and saving to {file_to_save_split_clips}...')
                if False:
                    random_split_clips(file_to_save_split_text, file_to_save_split_clips, args.min_duration, args.max_duration, args.history_duration, args.min_active_rate, args.max_active_rate)
                else:
                    split_clips(file_to_save_split_text, file_to_save_split_clips, args.clip_duration, args.clip_overlap, args.min_duration, args.history_duration, args.min_active_rate, args.max_active_rate)
                print(f'Finished processing {file_path}.')

if __name__ == '__main__':
    if False:
        import debugpy
        try:
            debugpy.listen(('localhost', 9501))
            print(f'debug listen on port 9501')
            debugpy.wait_for_client()
        except Exception as e:
            raise RuntimeError(f"Failed to start debugpy: {e}")
    main()