import asyncio
from openai import AsyncAzureOpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
import re
from radar.utils.prompts import build_2class_eval_prompt, build_3class_eval_prompt, build_4class_eval_prompt

AZURE_SCOPE = "api://trapi/.default"
AZURE_API_VERSION = '2024-10-21'
AZURE_DEPLOYMENT = "gpt-4o_2024-11-20"
AZURE_ENDPOINT = 'https://trapi.research.microsoft.com/gcr/shared'
CONCURRENCY_LIMIT = 5 
EVAL_PROMPT = build_4class_eval_prompt()

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

async def call_gpt_eval_model(client: AsyncAzureOpenAI, formatted_input: str, semaphore: asyncio.Semaphore):
    full_prompt = f"{EVAL_PROMPT}\n\n{formatted_input}"
    max_retries = 5

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=AZURE_DEPLOYMENT,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[Error] API call failed: {e}. Retrying...")
                await asyncio.sleep(5)
    
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
    credential = get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()), AZURE_SCOPE
    )
    
    async def run_batch():
        async_client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            azure_ad_token_provider=credential,
            api_version=AZURE_API_VERSION,
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