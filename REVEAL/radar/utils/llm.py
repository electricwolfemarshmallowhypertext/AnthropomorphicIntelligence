import os
import json
import asyncio
from typing import List, Dict, Callable, Any, Optional
from tqdm.asyncio import tqdm_asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from openai import AsyncOpenAI, RateLimitError, APIError, APITimeoutError

class AsyncLLMEngine:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        concurrency: int = 10,
    ):
        self.model = model
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        
        final_api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy_key")

        self.client = AsyncOpenAI(
            api_key=final_api_key,
            base_url=base_url,
            timeout=120.0,
        )

    @retry(
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(8),
        retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError))
    )
    async def _generate_single(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        max_tokens: int = 1024
    ) -> Optional[str]:
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens 
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"\n[LLM Error] {e}")
                raise e

    async def run_batch(
        self,
        items: List[Dict],
        prompt_builder: Callable[[Dict], str],
        system_prompt: str,
        output_file: str,
        result_parser: Callable[[Dict, str], Dict]
    ):
        if not items:
            return

        print(f"🚀 Engine started. Processing {len(items)} items using '{self.model}' with concurrency {self.concurrency}...")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        async def process_item(item):
            try:
                user_prompt = prompt_builder(item)
                response_text = await self._generate_single(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                if response_text:
                    return result_parser(item, response_text)
                return None
            except Exception:
                return None

        tasks = [process_item(item) for item in items]
        
        with open(output_file, "a", encoding="utf-8") as f:
            for coro in tqdm_asyncio.as_completed(tasks, desc="Generating"):
                result = await coro
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()