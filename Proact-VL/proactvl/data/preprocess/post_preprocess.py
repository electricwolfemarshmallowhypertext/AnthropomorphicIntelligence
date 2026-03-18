import json
import os
import random
import argparse
from typing import List

def random_select_samples(input_files: List[str], output_file: str, select_nums: List[int], seed: int = 42):
    rng = random.Random(seed)  # ✅ Fixed seed: deterministic results given the same input order/content

    selected_samples = []
    for input_file, select_num in zip(input_files, select_nums):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        if select_num >= len(data):
            selected_samples.extend(data)
        elif select_num > 0:
            selected_samples.extend(rng.sample(data, min(select_num, len(data))))
        elif select_num == 0:
            continue
        else:
            selected_samples.extend(data)

    with open(output_file, 'w', encoding='utf-8') as fo:
        for item in selected_samples:
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f'Finish saving {len(selected_samples)} selected samples to {output_file}')

def main():
    parser = argparse.ArgumentParser(description="Randomly select samples from input files")
    parser.add_argument('--input_files', nargs='+', required=True, help='List of input JSON files')
    parser.add_argument('--select_nums', nargs='+', type=int, required=True, help='Number of samples to select from each file')
    parser.add_argument('--output_file', required=True, help='Output file to save selected samples')
    args = parser.parse_args()
    random_select_samples(args.input_files, args.output_file, args.select_nums)

if __name__ == '__main__':
    main()

