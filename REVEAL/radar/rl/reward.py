import os
import re
import asyncio
from openai import AsyncOpenAI
from radar.utils.prompts import build_2class_eval_prompt, build_3class_eval_prompt, build_4class_eval_prompt

CONCURRENCY_LIMIT = 5 
OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_BASE_URL = "https://api.openai.com/v1"
EVAL_PROMPT = None

def init_eval_prompt(dataset_name: str):
    global EVAL_PROMPT
    dataset_name_lower = dataset_name.lower()
    if "4class" in dataset_name_lower:
        EVAL_PROMPT = build_4class_eval_prompt()
    elif "3class" in dataset_name_lower:
        EVAL_PROMPT = build_3class_eval_prompt()
    else:
        EVAL_PROMPT = build_2class_eval_prompt()

def extract_xml_content(text: str, tag: str):
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None

# 1. format_reward
def format_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    rewards = []
    pattern = re.compile(r"<think>(.+?)</think>\s*<answer>\s*(Human|AI)\s*</answer>", re.DOTALL | re.IGNORECASE)
    
    contents = [comp[0]["content"] for comp in completions]
    
    for generated_content in contents:
        match = pattern.search(generated_content)
        if match:
            rewards.append(0.0)
        else:
            rewards.append(-1.0)
            
    return rewards

# 2. answer_reward
def answer_reward(completions: list[list[dict]], solution: list[str], **kwargs) -> list[float]:
    rewards = []
    pattern = re.compile(r"<answer>\s*(Human|AI)\s*</answer>", re.DOTALL | re.IGNORECASE)
    
    contents = [comp[0]["content"] for comp in completions]
    
    for generated_content, ground_truth in zip(contents, solution):
        match = pattern.search(generated_content)
        if match and match.group(1).lower() == ground_truth.lower():
            rewards.append(5.0)
        else:
            rewards.append(0.0)
            
    return rewards

async def call_gpt_eval_model(client: AsyncOpenAI, formatted_input: str, semaphore: asyncio.Semaphore):
    if EVAL_PROMPT is None:
        raise ValueError("EVAL_PROMPT is not initialized. Please call init_eval_prompt() first.")
    
    full_prompt = f"{EVAL_PROMPT}\n\n{formatted_input}"
    max_retries = 5

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=OPENAI_MODEL_NAME,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=1024
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Error] API call failed: {e}. Retrying (attempt {attempt+1}/{max_retries})...")
                if "429" in str(e) or "limit" in str(e).lower():
                    await asyncio.sleep(10 + attempt * 5)
                else:
                    await asyncio.sleep(5)
                continue
    
    print(f"[Fail] Gave up on prompt after {max_retries} attempts.")
    return None

def extract_scores_from_gpt(response_str):
    if not response_str:
        return None
    match = re.search(r"\[\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*,\s*(\d(?:\.\d+)?)\s*\]", response_str)
    if not match:
        return None
    return [float(match.group(1)), float(match.group(2)), float(match.group(3))]

async def process_single_item(client, solution_str, original_text, semaphore):
    if not original_text:
        return 0.0

    gpt_eval_input = f"Text:\n{original_text}\n\nModel Output:\n{solution_str}"
    
    response_str = await call_gpt_eval_model(client, gpt_eval_input, semaphore)
    scores = extract_scores_from_gpt(response_str)
    
    if scores is None:
        return 0.0

    alignment_score, groundedness_score, genericness_score = scores
    
    alignment_weight = 0.5  
    groundedness_weight = 0.3 
    genericness_weight = 0.2   

    consistency_score = (
        alignment_weight * alignment_score +
        groundedness_weight * groundedness_score +
        genericness_weight * genericness_score
    )
    return consistency_score

def consistency_reward(completions: list[list[dict]], text: list[str], **kwargs) -> list[float]:
    api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
    
    async def run_batch():
        async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENAI_BASE_URL,
            timeout=120.0,
        )
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        
        tasks = []
        contents = [completion[0]["content"] for completion in completions]        
        for content, original_text in zip(contents, text):
            task = process_single_item(async_client, content, original_text, sem)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        await async_client.close()
        return results
    
    return asyncio.run(run_batch())