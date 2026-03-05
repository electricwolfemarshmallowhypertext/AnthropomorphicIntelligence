import os
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential
import tiktoken
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from copy import deepcopy
import vllm
import torch
import re
import json
import time
import emoji
import csv
from os.path import expanduser 
from tqdm import tqdm 
import hashlib 
from langdetect import detect
import pandas as pd

def is_null(value):
    if pd.isna(value) or value is None or value == "" or value.lower() == "n/a" or value.lower() == "null" or value.lower() == "none" or value.lower() == "nan":
        return True
    return False

def extract_tag_content(text, tag):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    start_index = text.find(start_tag)
    if start_index == -1:
        return None
    
    start_index += len(start_tag)
    end_index = text.find(end_tag, start_index)
    if end_index == -1:
        return None
    
    return text[start_index:end_index].strip()

def extract_tag_content_with_bounds(text, tag, begin_idx=0):
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"

    start_index = text.find(start_tag, begin_idx)
    if start_index == -1:
        return None, None

    start_index += len(start_tag)
    end_index = text.find(end_tag, start_index)
    if end_index == -1:
        return None, None

    return text[start_index:end_index].strip(), end_index + len(end_tag)

def extract_field_from_json_re(json_string, field):
    # Applicable to extract string type fields (values ​​can contain quotation marks)
    pattern = rf'"{field}"\s*:\s*"([^"]*?)"'
    match = re.search(pattern, json_string, re.DOTALL)

    if match:
        return match.group(1)  # Extract matching field values
    return None  # No matching fields found

def extract_field_from_json_re_2(json_string, field):
    pattern = rf'"{field}"\s*:\s*(?:"([^"]*?)"|([^,}}\s]+))'
    match = re.search(pattern, json_string)
    if match:
        return match.group(1) if match.group(1) is not None else match.group(2)
    return None

def extract_after_keyword(text, keyword):
    index = text.find(keyword)
    if index != -1:
        return text[index + len(keyword):]  # Extract the following text
    return ""

def extract_from_description(text, keyword):
    index = text.find(keyword)
    if index != -1:
        return text[index:]
    return ""

def filter_best_description(descriptions):
    if not descriptions or len(descriptions) == 0:
        return ''    
    # Sort by length, pick the longest description
    best_description = max(descriptions, key=len, default='')
    
    return best_description

def get_offline_model(args):
    model = vllm.LLM(
        args.model_name_or_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=float(os.getenv("TOOL_VLLM_GPU_MEMORY_UTILIZATION", 0.94)),
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=False
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer

def get_online_model(args):
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

def remove_urls(text):
    return re.sub(r'http[s]?://\S+', '', text)

def remove_special_chars(text):
    text = re.sub(r'\\[xu][0-9A-Fa-f]+', '', text)  # Unicode escape characters
    text = re.sub(r'&\w+;', '', text)  # HTML escape characters
    return text.strip()

def remove_emojis(text):
    return emoji.replace_emoji(text, '')

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_html_like(text):
    return re.sub(r'<.*?>', ' ', text)

def is_english_text(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def clean_text(text, remove_mention=False):
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = remove_emojis(text)
    if remove_mention:
        text = remove_mentions(text)
    text = remove_html_like(text)
    return text if is_english_text(text) else ''

def clean_data(samples, remove_mention=False):
    output = []
    for data in samples:
        try:
            persona = clean_text(data['persona'], remove_mention)
            if is_null(persona) or len(persona) < len(data['persona'])//2:
                continue
            data['persona'] = persona
            if 'scenario' in data:
                scenario = clean_text(data['scenario'], remove_mention)
                if is_null(scenario) or len(scenario) < len(data['scenario'])//2:
                    continue
                data['scenario'] = scenario
            if 'behavior' in data:
                behavior = clean_text(data['behavior'], remove_mention)
                if is_null(behavior) or len(behavior) < len(data['behavior'])//2:
                    continue
                data['behavior'] = behavior
            output.append(data)
        except Exception as e:
            continue
    print(f"Processed {len(output)} lines.")
    return output

def safe_save2file(filename, df):
    max_retries = 5
    success = False
    for attempt in range(0, max_retries):
        try:  
            df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)  
            success = True
            break  
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed in saving file {filename}: {e}")
            time.sleep(5)  # Wait before retrying  
            continue  
    return success

def safe_loadcsv(filename):
    max_retries = 5
    success = False
    for attempt in range(0, max_retries):
        try:  
            df = pd.read_csv(filename)  
            success = True
            break  
        except Exception as e:
            time.sleep(1)  # Wait before retrying  
            continue  
    return df if success else None

if __name__ == "__main__":
    pass