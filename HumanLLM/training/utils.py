from datasets import Dataset, DatasetDict, concatenate_datasets
import json
import os
import random
from collections import defaultdict
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential
import tiktoken
from transformers import AutoTokenizer

def load_all_datasets(script_args):
    train_dataset = []
    with open(script_args.train_file, "r") as f:
        for line in f:
            data = json.loads(line)
            train_dataset.append({
                "messages": data["messages"],
            })
    random.seed(42)
    random.shuffle(train_dataset)
    test_dataset_general = []
    test_dataset_indomain = []
    with open(script_args.test_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data['source'].startswith("smoltalk.") or data['source'].startswith("orca."):
                test_dataset_general.append({
                    "messages": data["messages"],
                })
            else:
                test_dataset_indomain.append({
                    "messages": data["messages"],
                })
    
    train_dataset = Dataset.from_list(train_dataset[:script_args.max_samples_per_dataset])
    test_general_dataset = Dataset.from_list(test_dataset_general[:script_args.max_samples_per_dataset])
    test_indomain_dataset = Dataset.from_list(test_dataset_indomain[:script_args.max_samples_per_dataset])
    return DatasetDict({"train": train_dataset, "test_general": test_general_dataset, "test_indomain": test_indomain_dataset})

def load_inference_datasets(script_args):
    test_dataset = defaultdict(list)
    with open(script_args.dataset_file, "r") as f:
        for line in f:
            data = json.loads(line)
            data["uid"] = data.get("uid", "")
            data["source"] = data.get("source", "")
            if script_args.domains and data['source'] not in script_args.domains:
                continue
            test_dataset[data['source']].append({
                "uid": str(data["uid"]),
                "source": data["source"],
                "messages": data["messages"],
            })
    sampled_test_dataset = []
    for key, value in test_dataset.items():
        sampled_test_dataset.extend(value[:script_args.max_samples_per_task])

    test_dataset = {"test": sampled_test_dataset}
    return test_dataset


def analyze_token_counts(dataset, tokenizer, training_args):
    def count_tokens(example):
        total_tokens = sum(len(tokenizer.encode(msg['content'], add_special_tokens=False)) for msg in example['messages'])
        return {"token_count": total_tokens}

    # Add token count to each example
    tokenized_dataset = dataset.map(count_tokens, load_from_cache_file=True)

    filtered_dataset = tokenized_dataset.filter(lambda example: example["token_count"] <= training_args.max_seq_length-40) # 36 is the template length
    # Extract token counts
    token_counts = filtered_dataset["train"]["token_count"]

    # Calculate statistics
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    avg_tokens = sum(token_counts) / len(token_counts)

    filtered_dataset = filtered_dataset.remove_columns("token_count")

    return {
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "avg_tokens": avg_tokens
    }, filtered_dataset

def get_model(args):
    api_key = os.environ.get('OPENAI_API_KEY') if os.environ.get('OPENAI_API_KEY') else None
    api_base =  os.environ.get('OPENAI_API_BASE') if os.environ.get('OPENAI_API_BASE') else None
    api_type = os.environ.get('OPENAI_API_TYPE') if os.environ.get('OPENAI_API_TYPE') else None
    api_version =  os.environ.get('OPENAI_API_VERSION') if os.environ.get('OPENAI_API_VERSION') else None

    if api_key:
        if api_type == "azure":
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base,
            )
        else:
            client = OpenAI(  
                api_key=api_key,
                base_url=api_base,
            )
    else:
        credential = AzureCliCredential()    

        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=api_base,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            max_retries=5,
        )

    if args.model_name_or_path.startswith("gpt-3"):
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif args.model_name_or_path.startswith("gpt-4o"):
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    elif args.model_name_or_path.startswith("gpt-4"):
        tokenizer = tiktoken.encoding_for_model("gpt-4")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    return client, tokenizer