import json
import os
import asyncio
import argparse
from openai import AsyncOpenAI

parser = argparse.ArgumentParser(description="Polish text using Standard OpenAI API.")

parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to the input JSONL file (human_data).")
parser.add_argument("--output_file", "-o", type=str, required=True, help="Path to the output JSONL file.")

parser.add_argument("--base_url", "-b", type=str, default=None, help="API Base URL (e.g., https://api.openai.com/v1).")
parser.add_argument("--api_key", "-k", type=str, default=None, help="API Key. Can also use OPENAI_API_KEY environment variable.")
parser.add_argument("--model_name", "-m", type=str, default="gpt-4o", help="Model name (e.g., gpt-4o, gpt-3.5-turbo).")

parser.add_argument("--concurrency", "-c", type=int, default=3, help="Number of concurrent API requests.")
parser.add_argument("--max_retries", "-r", type=int, default=5, help="Maximum number of retries for length mismatch or API errors.")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
API_CACHE = {}

api_key = args.api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = "dummy_key"

client = AsyncOpenAI(
    api_key=api_key,
    base_url=args.base_url,
    timeout=120.0,
)

SYSTEM_PROMPT = """You are a professional editor and writer.
Your task is to polish the provided text to make it clearer, more fluent, and grammatically correct while preserving the original meaning, voice, and style.

Guidelines:
- Correct errors in grammar, spelling, punctuation, and syntax.
- Improve sentence structure and flow; remove awkward phrasing.
- Enhance vocabulary where appropriate, but keep the tone consistent with the original text.
- Do NOT add new information or change the core message.
- Keep special markers (like `(Applause)`, `[Music]`) intact if they exist.
"""

async def async_call_gpt(item, semaphore, max_retries):
    uid = item.get("id")
    original_text = item.get("text", "")

    if not original_text.strip():
        return None

    target_length = len(original_text.split())

    cache_key = uid
    if cache_key in API_CACHE:
        item["text_ai"] = API_CACHE[cache_key]
        item["model"] = args.model_name
        return item

    user_prompt = f"""Please polish and improve the following text.\n
You MUST make your output about about {int(target_length * 0.9)} and {int(target_length * 1.1)} words.
If you exceed or fall short of this range, your output will be rejected.\nDo NOT mention word counts, constraints, or instructions. \n
And output only the polished text. Do not include explanations or preambles.

Original Text:
{original_text}
"""

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=4096*2, 
                )
                
                rewrite = response.choices[0].message.content.strip()
                rewrite_length = len(rewrite.split())
                
                if (target_length * 0.75 < rewrite_length < target_length * 1.25) or (target_length - 5 < rewrite_length < target_length + 5):
                    API_CACHE[cache_key] = rewrite
                    item["text_ai"] = rewrite
                    item["model"] = args.model_name
                    print(f"[✔] {uid}: generated ({target_length} words target).")
                    return item
                else:
                    print(f"[⚠] {uid} (attempt {attempt+1}/{max_retries}): length mismatch ({rewrite_length} vs {target_length}). Retrying...")
                    await asyncio.sleep(2)
                    continue

            except Exception as e:
                print(f"[API Error] {uid} (attempt {attempt+1}/{max_retries}): {e}")
                if "429" in str(e) or "limit" in str(e).lower():
                    await asyncio.sleep(10 + attempt * 5)
                    continue
                else:
                    await asyncio.sleep(2)
                    continue

    print(f"[❌] {uid}: failed after {max_retries} retries.")
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
    tasks = [asyncio.create_task(async_call_gpt(item, semaphore, args.max_retries)) for item in items]

    with open(args.output_file, "a") as fout:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"✅ Done! All polished text saved to {args.output_file}.")

if __name__ == "__main__":
    asyncio.run(main())