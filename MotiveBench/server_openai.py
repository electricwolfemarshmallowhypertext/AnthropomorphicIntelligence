import os
import json
import argparse
from typing import Literal
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.identity import get_bearer_token_provider, AzureCliCredential
from openai import AzureOpenAI

# Create FastAPI application
app = FastAPI()

# Azure OpenAI client configuration
credential = AzureCliCredential()

token_provider = get_bearer_token_provider(
    credential,
    ""
)

aoiclient = AzureOpenAI(
    azure_endpoint="",  # Set the endpoint for Azure OpenAI service
    azure_ad_token_provider=token_provider,  # Use credentials to obtain a token
    api_version="",  # API version
    max_retries=5,  # Maximum number of retries
)

# Define command-line argument parser
parser = argparse.ArgumentParser(description="FastAPI server with Azure OpenAI support.")
parser.add_argument("--model", required=True, help="Specify the model name to use, e.g., GPT-4o-mini.")
parser.add_argument("--port", type=int, default=3000, help="Specify the port for the FastAPI server (default: 3000).")
args = parser.parse_args()
model_name = args.model  # Retrieve model name from command-line argument
port = args.port  # Retrieve port number from command-line argument

# Request body model
class ChatRequest(BaseModel):
    messages: list  # Chat message history in the format [{"role": "user", "content": "message content"}]
    max_tokens: int  # Default maximum output of 200 tokens
    temperature: float  # Temperature for randomness in the output
    top_p: float  # Top-p sampling parameter


@app.post("/v1/chat/completions")
async def chat_openai(request: ChatRequest):
    try:
        # Call Azure OpenAI API
        response = aoiclient.chat.completions.create(
            model=model_name,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Retrieve the number of input and output tokens
        usage = response.usage
        prompt_tokens = usage.prompt_tokens  # Number of input tokens
        completion_tokens = usage.completion_tokens  # Number of output tokens
        total_tokens = usage.total_tokens  # Total number of tokens

        # Define cost information for each model
        cost_map = {
            "GPT-4-Turbo": (0.01, 0.03),
            "GPT-4o": (0.005, 0.015),
            "GPT-35-Turbo": (0.001, 0.002),
            "GPT-4o-mini": (0.00015, 0.0006),
            "o1-preview": (0.015, 0.06),
        }

        # Check if the model name is valid
        if model_name not in cost_map:
            raise HTTPException(status_code=500, detail="Undefined model name")

        # Calculate costs based on token usage
        input_cost_per_1000_tokens, output_cost_per_1000_tokens = cost_map[model_name]
        total_cost = (prompt_tokens / 1000) * input_cost_per_1000_tokens + (completion_tokens / 1000) * output_cost_per_1000_tokens

        print(f"Total cost: ${total_cost:.6f}")
        print(response)

        # Read and update cost.json file
        if os.path.exists("cost.json"):
            with open("cost.json", "r") as f:
                cost_all = json.load(f)
        else:
            cost_all = {}

        # Update the total cost for the model
        if model_name in cost_all:
            cost_all[model_name] += total_cost
        else:
            cost_all[model_name] = total_cost

        # Write the updated cost data back to the file
        with open("cost.json", "w") as f:
            json.dump(cost_all, f, indent=4)

        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
