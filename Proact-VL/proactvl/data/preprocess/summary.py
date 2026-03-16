import os
import json
import argparse


def show_distribution(ann_dir):
    for path in os.listdir(ann_dir):
        if 'split_clips' in path:
            with open(os.path.join(ann_dir, path), 'r') as f:
                lines = f.readlines()
            print(f'clip num in {path}: {len(lines)}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann_dir',
        type=str,
        help='Path to annotation directory.'
    )
    args = parser.parse_args()
    show_distribution(args.ann_dir)

if __name__ == "__main__":
    main()