import argparse
import os
import json
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import ffmpeg

LIVE_VIDEO_DIR = "live_sft/videos"
def process_one_file(ann_dir: str, ann_file: str, min_duration: int, max_duration: int):
    """
    子进程函数：处理单个 .json 文件（JSONL 格式：一行一个 JSON）
    返回：List[dict] 该文件产生的 annotations（可能为空）
    """
    out = []
    file_path = os.path.join(ann_dir, ann_file)

    # 只处理 .json
    if not ann_file.endswith(".json"):
        return out

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # 你原逻辑：从 data[0]['content'][0] 取视频信息
            video_start = data[0]['content'][0]['video_start']
            video_end = data[0]['content'][0]['video_end']
            video_duration = math.floor(video_end - video_start)
            video_path = os.path.join(LIVE_VIDEO_DIR, ann_file.replace('.json', '.mp4'))
            try:
                probe = ffmpeg.probe(os.path.join(ann_dir, video_path))
                video_duration = float(probe['format']['duration'])
                video_duration = math.floor(video_duration)
            except ffmpeg.Error as e:
                print(f"ffmpeg probe error for {video_path}: {e}")
                continue
            if video_duration < min_duration:
                continue
            video_duration = min(video_duration, max_duration)
            # query / history / meta
            query = data[0]['content'][1]['text']
            history = data[0]['content'][1]['previous']
            title = data[0]['content'][1]['title']
            category = data[0]['content'][1]['category']
            
            new_ann = {
                'video_path': video_path,   # 你原来就是 ann_file
                'video_begin': 0,
                'video_end': video_duration,
                'video_duration': video_duration,
                'active_speaker': {
                    'name': 'SPEAKER_00',
                    'persona': 'You are a helpful and informative AI assistant that provides detailed and accurate answers based on the video content.'
                },
                'metadata': {
                    'title': title,
                    'category': category,
                    'dataset_name': 'LiveCC',
                    'tag': 'livecc'
                },
                'annotations': []
            }

            if history and history.strip():
                new_ann['history'] = history.strip()

            if query and query.strip():
                new_ann['annotations'].append({
                    'role': 'user',
                    'speaker': 'user',
                    'start': 0,
                    'end': 1,
                    'query': query.strip()
                })
            # assistant stream
            item = None
            for ann in data[1]['content'][0]['text_stream']:
                start_time = ann[0] - video_start
                end_time = ann[1] - video_start

                # 你原逻辑：超出视频时长 or 超出 max_duration 则 break
                if start_time >= video_duration or start_time >= max_duration or end_time > video_duration:
                    break

                text = ann[2].strip()
                if not text:
                    continue

                st = math.floor(start_time)
                if item is None:
                    item = {
                        'role': 'assistant',
                        'speaker': 'SPEAKER_00',
                        'start': st,
                        'end': st + 1,
                        'text': text
                    }
                elif st == item['start']:
                    item['text'] += ' ' + text
                else:
                    new_ann['annotations'].append(item)
                    item = {
                        'role': 'assistant',
                        'speaker': 'SPEAKER_00',
                        'start': st,
                        'end': st + 1,
                        'text': text
                    }

            if item is not None:
                new_ann['annotations'].append(item)

            out.append(new_ann)

    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare LiveCC training data (multiprocessing)")
    parser.add_argument("--ann_dir", required=True, help="Directory containing input JSONL(.json) files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--min_duration", type=int, default=18, help="Minimum duration of clips to be included")
    parser.add_argument("--max_duration", type=int, default=60, help="Maximum duration of clips to be included")
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of worker processes (default: cpu_count-1)")

    # debug：只在主进程开启，避免子进程端口冲突
    parser.add_argument("--debug", action="store_true", help="Enable debugpy in main process only")
    parser.add_argument("--debug_port", type=int, default=9501, help="debugpy port (main process)")

    args = parser.parse_args()

    if args.debug:
        import debugpy
        try:
            debugpy.listen(('0.0.0.0', args.debug_port))
            print(f"debugpy listening on {args.debug_port}")
            debugpy.wait_for_client()
        except Exception as e:
            raise RuntimeError(f"Failed to start debugpy: {e}")

    files = [f for f in os.listdir(args.ann_dir) if f.endswith(".json")]
    files.sort()

    annotations = []
    total_files = len(files)
    if total_files == 0:
        print(f"No .json files found in {args.ann_dir}")
        return

    # 多进程并发：按文件粒度
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [
            ex.submit(process_one_file, args.ann_dir, ann_file, args.min_duration, args.max_duration)
            for ann_file in files
        ]

        for fut in tqdm(as_completed(futures), total=total_files, desc="Processing files"):
            res = fut.result()
            if res:
                annotations.extend(res)

    # 写输出
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as out_f:
        for ann in annotations:
            out_f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    print(f"Processed {len(annotations)} annotations and saved to {args.output_file}")


if __name__ == "__main__":
    main()
