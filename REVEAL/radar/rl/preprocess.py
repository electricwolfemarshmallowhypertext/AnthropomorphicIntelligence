import json
import random
import os
import re
import argparse
import glob
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from radar.utils.metrics import extract_answer
from radar.utils.prompts import build_2class_prompt, build_3class_prompt, build_4class_prompt

SEED = 42
random.seed(SEED)

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return data
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} records to {filepath}")

def format_chatml(tokenizer, system_msg, user_msg):
    messages = [{"role": "user", "content": f"{system_msg}\n\nText:\n{user_msg}"}]
    text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return text

def determine_label_from_filename(filename, num_classes):
    fname = filename.lower()
    if "human" in fname:
        return "Human"
    elif "polish" in fname:
        if num_classes == 2: return None
        return "Polish"
    elif "meaningless" in fname:
        if num_classes in [2, 3]: return None
        return "Meaningless"
    else:
        return "AI"

def main(args):
    print("Step 1: Loading existing SFT data to exclude...")
    used_texts = set()
    for fpath in [args.sft_train_file, args.sft_test_file]:
        data = load_jsonl(fpath)
        for item in data:
            text = item.get('text', '').strip()
            if text:
                used_texts.add(text)
    print(f"Loaded {len(used_texts)} used texts to exclude.")

    print(f"Step 2: Loading raw data candidates from {args.input_dir}...")
    candidates_pool = defaultdict(list)    
    files = glob.glob(os.path.join(args.input_dir, "**/*.jsonl"), recursive=True)
    for file_path in files:
        label = determine_label_from_filename(file_path, args.num_classes)
        
        print(f"  - Loading {file_path} -> Classified as [{label}]")
        file_data = load_jsonl(file_path)        
        for item in file_data:
            text = item.get('text', item.get('text_ai', '')).strip()
            
            if not text or text in used_texts:
                continue
                
            entry = {
                'text': text,
                'label': label,
                'source': item.get('model', label)
            }
            candidates_pool[label].append(entry)

    print("\nCandidate Statistics:")
    total_candidates = 0
    for label, items in candidates_pool.items():
        random.shuffle(items)
        print(f"  - {label}: {len(items)}")
        total_candidates += len(items)
    
    if total_candidates == 0:
        print("Error: No candidates loaded. Check input directory and filenames.")
        return

    print(f"\nStep 3: Initializing model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    llm = LLM(
        model=args.model_name, 
        tensor_parallel_size=args.tp_size, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=2048,
        n=8
    )

    print(f"Using prompt for num_classes={args.num_classes}")
    if args.num_classes == 2:
        instruction = build_2class_prompt()
    elif args.num_classes == 3:
        instruction = build_3class_prompt()
    elif args.num_classes == 4:
        instruction = build_4class_prompt()
    else:
        raise ValueError(f"Unsupported num_classes={args.num_classes}")

    def process_class(candidates, target_label):
        hard_list = []
        pointer = 0
        target_label_lower = target_label.lower()
        
        print(f"\nMining Hard Samples for category: {target_label} ...")
        
        while len(hard_list) < args.target_total and pointer < len(candidates):
            batch_data = candidates[pointer : pointer + args.batch_size]
            pointer += args.batch_size
            
            if not batch_data:
                break
                
            prompts = [format_chatml(tokenizer, instruction, item["text"]) for item in batch_data]            
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            
            for j, output_group in enumerate(outputs):
                correct_count = 0
                wrong_count = 0
                
                for generated_seq in output_group.outputs:
                    pred_text = generated_seq.text
                    pred_label = extract_answer(pred_text)
                    
                    if target_label_lower == pred_label:
                        correct_count += 1
                    else:
                        wrong_count += 1
                
                is_hard = (correct_count > 0 and wrong_count > 0)
                
                if is_hard:
                    hard_list.append(batch_data[j])
                    if len(hard_list) >= args.target_total:
                        break
            
            print(f"  -> Processed {pointer} items. Found {len(hard_list)}/{args.target_total} hard samples.")

        return hard_list

    final_dataset = []
    labels_to_process = sorted(candidates_pool.keys())
    
    for label in labels_to_process:
        hard_samples = process_class(candidates_pool[label], label)
        final_dataset.extend(hard_samples)

    print(f"\nStep 5: Finalizing dataset. Total records: {len(final_dataset)}")
    
    final_dist = defaultdict(int)
    for item in final_dataset:
        final_dist[item['label']] += 1
    for k, v in final_dist.items():
        print(f"  - {k}: {v}")

    random.shuffle(final_dataset)
    save_jsonl(final_dataset, args.output_file)
    print(f"Done! RL dataset generated at {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Data Preprocessing with Hard Sample Mining (Multi-class)")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw jsonl files")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output jsonl")
    parser.add_argument("--sft_train_file", type=str, required=True, help="SFT train file to exclude")
    parser.add_argument("--sft_test_file", type=str, required=True, help="SFT test file to exclude")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes (e.g., 4)")

    parser.add_argument("--target_total", type=int, default=3000, help="Target number of hard samples per class")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--tp_size", type=int, default=4, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="GPU memory utilization")

    args = parser.parse_args()
    main(args)