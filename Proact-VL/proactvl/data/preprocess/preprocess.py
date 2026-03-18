'''
Round the timestep.
{
    "video_path": str,
    "video_begin": int,
    "video_end": int,
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
            "start": int,
            "end": int,
            "text": str,
        },
    ]
}
'''

import json
import os
import math
import argparse


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--ann_dir', type=str, required=True, help='Path to the directory containing annotation files.')
    args.add_argument('--dataset_type_list', nargs='+', default=['train', 'val'], help='only useful for commentary game dataset')
    args = args.parse_args()

    for root, dirs, files in os.walk(args.ann_dir):
        for filename in files:
            if filename.endswith('.jsonl') and 'merged' in filename and any(dataset_type in filename for dataset_type in args.dataset_type_list):
                file_path = os.path.join(args.ann_dir, filename)
                file_to_save = file_path.replace('merged', 'standard_format')
                print(f'Processing {file_path} and saving to {file_to_save}...')
                with open(file_path, 'r') as f_in, open(file_to_save, 'w') as f_out:
                    for line in f_in:
                        ann = json.loads(line)
                        video_duration = math.floor(ann['video_duration'])
                        ann['video_begin'] = 0
                        ann['video_end'] = video_duration
                        ann['video_duration'] = video_duration
                        for segment in ann['annotations']:
                            segment['start'] = round(segment['start'])
                            segment['end'] = min(round(segment['end']), video_duration)
                        f_out.write(json.dumps(ann, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()