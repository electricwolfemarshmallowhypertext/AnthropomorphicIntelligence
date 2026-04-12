import json
import os
import asyncio
import random
import argparse
from openai import AsyncOpenAI

parser = argparse.ArgumentParser(description="Generate AI text based on summaries using Standard OpenAI API.")

parser.add_argument("--summary_file", "-s", type=str, required=True, help="Path to the input summary JSONL file.")
parser.add_argument("--human_file", "-i", type=str, required=True, help="Path to the human reference JSONL file.")
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
    timeout=60.0,
)

SYSTEM_PROMPT = """You are a talented human writer.
You will receive:
1. A short summary describing what the text should convey.
2. A writing style description.
3. An example human-written text from the same category (for stylistic reference).

Your task:
- Write a new text that naturally expresses the ideas in the summary.
- Match the length, tone, and style suggested by the summary and the example.
- Do NOT copy or rewrite the original text; this should be a new composition.
- Keep the tone realistic, human, and coherent.
- The text should feel authentic, as if written by a person in that style.
"""

SYSTEM_PROMPT_HUMAN_LIKE = """
You are a talented human writer.
Your writing must always sound natural, human-like, and free of any AI tone.

You will receive:
1. A short summary describing what the text should convey.
2. A writing style description.
3. An example human-written text from the same category (for stylistic reference).

Your task:
- Write a new text that naturally expresses the ideas in the summary.
- Match the length, tone, and style suggested by the summary and the example.
- Produce a genuinely human-like piece: nuanced wording, slight imperfections, natural transitions, and authentic emotional texture.
- Do NOT copy or rewrite the example; create a fresh original composition.
- Avoid robotic or overly polished AI phrasing.
- Keep the text coherent, realistic, and appropriate for the described style.
- The final text should read as if written by a real person.
"""

def load_human_samples(human_file):
    type_samples = {}
    with open(human_file, "r") as fin:
        for line in fin:
            try:
                item = json.loads(line)
                uid = item.get("id", "")
                maybe_type = uid.rsplit("_", 1)[0] if "_" in uid else "unknown"
                if maybe_type not in type_samples:
                    type_samples[maybe_type] = []
                type_samples[maybe_type].append(item)
            except json.JSONDecodeError:
                continue
    return type_samples

async def async_call_gpt(item, semaphore, type_samples, max_retries):
    uid = item.get("id")
    summary_info = item.get("summary", "")
    original_text = item.get("text", "")
    prefix = uid.rsplit("_", 1)[0] if "_" in uid else "unknown"

    if not summary_info.strip() or not original_text.strip():
        return None

    try:
        summary_json = json.loads(summary_info)
        summary = summary_json.get("summary", "")
        style = summary_json.get("writing_style", "")
    except Exception:
        summary = summary_info
        style = ""

    target_length = len(original_text.split())

    sample_text = ""
    if prefix in type_samples and type_samples[prefix]:
        sample_item = random.choice(type_samples[prefix])
        sample_text = sample_item.get("text", "")

    cache_key = uid
    if cache_key in API_CACHE:
        item["text_ai"] = API_CACHE[cache_key]
        item["model"] = args.model_name
        return item

    user_prompt = f"""You are to write a new piece based on the following information.

Summary:
{summary}

Writing style:
{style}

Human sample (same type for style reference):
{sample_text}

Now, write a new, original text (not a rewrite) that expresses the same ideas as in the summary,
in the same tone and genre as the sample.
"""
    if prefix == "speech":
        user_prompt += "\nFor this text, you should include audience reactions like `(claps)`, `(Music)`, `(Music ends)`, `(Applause)`, `(Applause ends)` and so on."
    elif prefix == "presidential_speech":
        user_prompt += "\nFor this text, you should maintain a polite, formal tone. Show gratitude to potential guests."

    user_prompt += f"""\nYou MUST make your output about about {int(target_length * 0.9)} and {int(target_length * 1.1)} words.
If you exceed or fall short of this range, your output will be rejected.\nDo NOT mention word counts, constraints, or instructions. Simply write the text."""

    for attempt in range(max_retries):
        async with semaphore:
            try:
                system_msg = SYSTEM_PROMPT_HUMAN_LIKE if random.random() < 0.2 else SYSTEM_PROMPT
                
                response = await client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
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
                    item["Human_like"] = False if system_msg == SYSTEM_PROMPT else True
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
    with open(args.summary_file, "r") as fin:
        for line in fin:
            try:
                item = json.loads(line)
                if item.get("id") not in processed_ids:
                    items.append(item)
            except json.JSONDecodeError:
                continue

    type_samples = load_human_samples(args.human_file)
    print(f"🚀 Ready to process {len(items)} items with {len(type_samples)} type pools...")
    print(f"[*] Base URL: {args.base_url or 'OpenAI Default'}")
    print(f"[*] Model: {args.model_name}")
    print(f"[*] Concurrency: {args.concurrency}")

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(async_call_gpt(item, semaphore, type_samples, args.max_retries)) for item in items]

    with open(args.output_file, "a") as fout:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"✅ Done! All generated stories saved to {args.output_file}.")

if __name__ == "__main__":
    asyncio.run(main())