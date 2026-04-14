import json
import os
import asyncio
import argparse
from openai import AsyncOpenAI

parser = argparse.ArgumentParser(description="Process text and generate summaries using Standard OpenAI API.")

parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to the input JSONL file.")
parser.add_argument("--output_file", "-o", type=str, required=True, help="Path to the output JSONL file.")
parser.add_argument("--model_name", "-m", type=str, default="gpt-4o", help="Model name (e.g., gpt-4o, gpt-3.5-turbo).")
parser.add_argument("--base_url", "-b", type=str, default=None, help="API Base URL (e.g., https://api.openai.com/v1). Leave empty to use OpenAI default.")
parser.add_argument("--api_key", "-k", type=str, default=None, help="API Key. Can also use OPENAI_API_KEY environment variable.")
parser.add_argument("--concurrency", "-c", type=int, default=3, help="Number of concurrent API requests.")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
API_CACHE = {}

api_key = args.api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = "dummy_key" 

client = AsyncOpenAI(
    api_key=api_key,
    base_url=args.base_url,
    timeout=30.0,
)

SYSTEM_PROMPT = """You are an expert linguistic analyst and summarizer.
Your task is to analyze a given text and its source. You will then generate:
1. A concise summary of the content (in one sentence).
2. A brief analysis of the writing style (in one sentence).
Your output must be a single, valid JSON object with the keys "summary" and "writing_style".
"""

async def async_call_gpt(item, semaphore):
    uid = item.get("id")
    text = item.get("text", "")
    text_type = uid[:-8] if uid else "unknown"
    
    if not text.strip():
        return None

    cache_key = uid
    if cache_key in API_CACHE:
        item["summary"] = API_CACHE[cache_key]
        return item

    user_prompt = f"""Analyze the following text, which is from a '{text_type}' post.

Text:
---
{text}
---

Based on the text and its source, provide the following in a single JSON object:
1. "summary": A single-sentence summary of the text's main point.
2. "writing_style": A brief summary of the author's writing style (e.g., formal, informal, humorous, academic, use of slang, sentence structure).

Your response must be only the JSON object, without any additional text or explanations.
"""

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.95,
            )
            summary = response.choices[0].message.content.strip()
            API_CACHE[cache_key] = summary
            item["summary"] = summary
            print(f"[✔] {uid}: {summary}")
            return item
        except Exception as e:
            print(f"[API Error] {uid}: {e}")
            return None

async def main():
    processed_ids = set()
    
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as fout:
            for line in fout:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["id"])
                except:
                    continue

    items = []
    with open(args.input_file, "r") as fin:
        for line in fin:
            try:
                item = json.loads(line)
                if item.get("id") not in processed_ids:
                    items.append(item)
            except json.JSONDecodeError:
                continue

    print(f"🚀 Ready to process {len(items)} items...")
    print(f"[*] Base URL: {args.base_url or 'OpenAI Default'}")
    print(f"[*] Model: {args.model_name}")
    print(f"[*] Concurrency: {args.concurrency}")

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(async_call_gpt(item, semaphore)) for item in items]

    with open(args.output_file, "a") as fout:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"✅ Done! All summaries saved to {args.output_file}.")

if __name__ == "__main__":
    asyncio.run(main())
