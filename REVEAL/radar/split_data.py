import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def get_config(args):
    config = {
        "Human": {
            "path": Path(args.human_data),
            "is_dir": False,
            "label": "Human",
            "text_keys": ["text"]
        },
        "AI": {
            "path": Path(args.ai_data),
            "is_dir": True,
            "label": "AI",
            "text_keys": ["text_ai"] 
        }
    }
    if args.num_classes == 3:
        if not args.polish_data:
            raise ValueError("[Error] You selected 3-class but did not provide --polish_data path.")
        config["Polish"] = {
            "path": Path(args.polish_data),
            "is_dir": True,
            "label": "Polish",
            "text_keys": ["text_ai"]
        }
        
    return config

def load_jsonl_file(filepath: Path) -> List[Dict]:
    data = []
    if not filepath.exists():
        print(f"[Warning] File {filepath} not found.")
        return data
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def extract_text(item: Dict, keys: List[str]) -> str:
    for key in keys:
        if key in item and item[key]:
            return item[key]
    return ""

def load_category_data(config_name: str, config: Dict) -> List[Dict]:
    print(f"[*] Loading {config_name} data...")
    raw_items = []
    
    path = config["path"]
    files_to_read = []

    if config["is_dir"]:
        if path.exists() and path.is_dir():
            files_to_read = list(path.glob("*.jsonl"))
            print(f"  - Found {len(files_to_read)} files in {path}")
        else:
            print(f"  - [Error] Directory {path} does not exist.")
    else:
        files_to_read = [path]

    for p in files_to_read:
        file_data = load_jsonl_file(p)
        for item in file_data:
            text = extract_text(item, config["text_keys"])
            if text:
                raw_items.append({
                    "text": text,
                    "label": config["label"],
                    "source": item.get("model", "Human")
                })
    
    print(f"  - Loaded {len(raw_items)} raw samples for {config_name}")
    return raw_items

def save_jsonl(data: List[Dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[✔] Saved {len(data)} records to {filepath}")

def balance_subcategories(category_name: str, data: List[Dict], target_total: int) -> List[Dict]:
    if not data:
        return []

    groups = defaultdict(list)
    for item in data:
        groups[item['source']].append(item)
    
    sub_categories = list(groups.keys())
    num_subs = len(sub_categories)
    print(f"\n  > Processing {category_name}: Found {num_subs} sub-categories: {sub_categories}")

    ideal_quota = target_total // num_subs
    
    min_available = min(len(items) for items in groups.values())
    final_quota_per_sub = min(ideal_quota, min_available)
    
    if final_quota_per_sub < ideal_quota:
        print(f"    [WARNING] Data insufficient for ideal balance in {category_name}.")
        print(f"    Target total was {target_total}, ideal per model was {ideal_quota}.")
        print(f"    Limiting factor: min available is {min_available}.")
        print(f"    -> Adjusting to {final_quota_per_sub} per model to ensure strict equality.")
    else:
        print(f"    Each sub-category will contribute {final_quota_per_sub} samples.")

    balanced_data = []
    for sub, items in groups.items():
        sampled = random.sample(items, final_quota_per_sub)
        balanced_data.extend(sampled)
    
    print(f"  > Final count for {category_name}: {len(balanced_data)} (Target was {target_total})")
    return balanced_data

def main():
    parser = argparse.ArgumentParser(description="Balance and split dataset for AI text detection.")

    parser.add_argument("--num_classes", type=int, choices=[2, 3], required=True, help="Number of classes: 2 (Human, AI) or 3 (Human, AI, Polish).")
    parser.add_argument("--human_data", type=str, required=True, help="Path to the Human dataset JSONL file.")
    parser.add_argument("--ai_data", type=str, required=True, help="Path to the directory containing AI native data.")
    parser.add_argument("--polish_data", type=str, default=None, help="Path to the directory containing Polish data (Required if num_classes=3).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split train/test files.")
    
    parser.add_argument("--target_per_class", type=int, default=9000, help="Target number of samples per main class (Human/AI/Polish).")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Ratio of data to be used for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    
    CONFIG = get_config(args)
    
    all_datasets = {}
    for category, cfg in CONFIG.items():
        all_datasets[category] = load_category_data(category, cfg)

    final_dataset = []
    print(f"\n=== Starting Balance Process (Target: {args.target_per_class} per class) ===")
    for category, data in all_datasets.items():
        balanced_cat_data = balance_subcategories(category, data, args.target_per_class)
        final_dataset.extend(balanced_cat_data)

    random.shuffle(final_dataset)
    total_len = len(final_dataset)
    
    if total_len == 0:
        print("\n[!] Error: No data processed. Please check your data paths.")
        return

    test_len = max(1, int(total_len * args.test_ratio))
    test_data = final_dataset[:test_len]
    train_data = final_dataset[test_len:]

    print(f"\n=== Split Results ({args.num_classes}-Class Task) ===")
    print(f"Total Records: {total_len}")
    print(f"Train Size: {len(train_data)}")
    print(f"Test Size:  {len(test_data)}")

    train_file = out_dir / f"train_{args.num_classes}class.jsonl"
    test_file = out_dir / f"test_{args.num_classes}class.jsonl"

    save_jsonl(train_data, train_file)
    save_jsonl(test_data, test_file)
    print("\n🎉 All Done!")

if __name__ == "__main__":
    main()