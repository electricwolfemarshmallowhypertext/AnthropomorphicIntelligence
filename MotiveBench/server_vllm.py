import argparse
from fastapi import FastAPI, Request
import torch
import uvicorn
import json
import datetime
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

app = FastAPI()
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Automatically select GPUs based on available devices
# gpu_count = torch.cuda.device_count()
# print("GPU_COUNT:", gpu_count)
# if gpu_count > 0:
#     # Set CUDA_VISIBLE_DEVICES to use all available GPUs
#     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
#     device = torch.device("cuda")
# else:
#     print("No GPU found. Defaulting to CPU.")
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.post("/v1/chat/completions")
async def create_item(request: Request):
    global model, tokenizer
    try:
        json_post_raw = await request.json()
        max_length = json_post_raw.get('max_tokens')
        top_p = json_post_raw.get('top_p')
        temperature = json_post_raw.get('temperature')
        messages = json_post_raw.get('messages')
        repetition_penalty = json_post_raw.get('repetition_penalty')

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_length)

        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = model.generate(inputs, sampling_params)
        response = outputs[0].outputs[0].text

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                }
            }],
        }
        log = f"[{time}] prompt: {messages[0]['content']}, response: {repr(response)}"
        print(log)
        return answer["choices"][0]["message"]["content"]

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"error": error_message}

def parse_args():
    parser = argparse.ArgumentParser(description="Run FastAPI server with custom port and model")
    parser.add_argument('--model', type=str, required=True, help="Model to load (e.g., 'microsoft/Phi-3-mini-4k-instruct')")
    parser.add_argument('--port', type=int, default=4000, help="Port to run the server on")
    return parser.parse_args()

if __name__ == '__main__':
    # Command line arguments
    args = parse_args()

    # Model and tokenizer loading based on provided model name
    model_dir = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = LLM(
        model_dir,  
        tensor_parallel_size=torch.cuda.device_count(),
        # tensor_parallel_size=2,
        gpu_memory_utilization=float(os.getenv("TOOL_VLLM_GPU_MEMORY_UTILIZATION", 0.8)),
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=False
    )

    # Running the server with the specified port
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)
