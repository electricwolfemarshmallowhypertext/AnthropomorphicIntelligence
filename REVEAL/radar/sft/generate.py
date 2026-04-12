import json
import os
import asyncio
import re
import argparse
from radar.utils.prompts import build_2class_cot_prompt, build_3class_cot_prompt, build_4class_cot_prompt
from radar.utils.llm import AsyncLLMEngine 
from radar.utils.metrics import extract_answer

def prompt_builder_2class(item):
    prompt = build_2class_cot_prompt(item['text'], item['label'])
    return prompt

def prompt_builder_3class(item):
    prompt = build_3class_cot_prompt(item['text'], item['label'])
    return prompt

def prompt_builder_4class(item):
    prompt = build_4class_cot_prompt(item['text'], item['label'])
    return prompt

def result_parser(item, response_text):
    return {
        "text": item['text'],
        "label": item['label'],
        "response": response_text
    }

async def main(args):
    system_prompt = "You are a helpful and meticulous AI assistant."
    if args.num_classes == 2:
        user_prompt_builder = prompt_builder_2class
    elif args.num_classes == 3:
        user_prompt_builder = prompt_builder_3class
    elif args.num_classes == 4:
        user_prompt_builder = prompt_builder_4class
    else:
        raise ValueError(
            f"Unsupported num_classes={args.num_classes}. "
            "Only support {2, 3, 4}."
        )

    engine = AsyncLLMEngine(
        model=args.model_name,
        base_url=args.base_url, 
        api_key=args.api_key,
        concurrency=args.concurrency
    )

    processed_set = set()
    if os.path.exists(args.output_file):
        print("Checking existing progress...")
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    response = item.get("response", "")

                    answer = extract_answer(response)
                    if answer and answer == item["label"].lower():
                        processed_set.add(item["text"])
                except Exception:
                    pass
    
    print(f"✅ Found {len(processed_set)} correctly processed items. Resuming...")

    pending_items = []
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    with open(args.input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data["text"] in processed_set:
                continue
            
            pending_items.append({
                "text": data.get('text'),
                "label": data.get('label'),
                "idx": i
            })

    await engine.run_batch(
        items=pending_items,
        prompt_builder=user_prompt_builder,
        system_prompt=system_prompt,       
        output_file=args.output_file,      
        result_parser=result_parser
    )
    print("All Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM generation with CoT.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")

    parser.add_argument("--model_name", type=str, default="o3", help="Model name to use (e.g., gpt-4o, o3)")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for the OpenAI API (or compatible endpoints)")
    parser.add_argument("--api_key", type=str, default=None, help="API Key. Can also be set via OPENAI_API_KEY environment variable.")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests limit")
    
    args = parser.parse_args()
    asyncio.run(main(args))