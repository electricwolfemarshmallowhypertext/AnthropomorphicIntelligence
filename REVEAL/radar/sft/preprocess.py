import json
import os
import re
import argparse
from radar.utils.metrics import extract_answer

def main(input_file, output_file):
    data = []
    processed_texts = set()
    
    total_lines = 0
    valid_count = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Processing: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            total_lines += 1
            item = json.loads(line)

            text = item.get("text")
            label = item.get("label")
            response = item.get("response", "")

            if not text or not label or text in processed_texts:
                continue

            pred_answer = extract_answer(response)
            
            if pred_answer and pred_answer == label.lower():
                data.append(item)
                processed_texts.add(text)
                valid_count += 1
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Total lines: {total_lines}, Valid (CoT Correct): {valid_count}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CoT data for SFT.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CoT JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output SFT JSONL file")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file)