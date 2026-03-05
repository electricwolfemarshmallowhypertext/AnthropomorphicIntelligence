import pandas as pd
import os
import torch
from copy import deepcopy
import time
import json
import vllm
from os.path import expanduser  
from tqdm import tqdm
import random
import argparse
from datetime import datetime, timezone

from amazon_prompt import *
from utils import extract_tag_content, filter_best_description, extract_field_from_json_re_2

HOME_DIR = os.path.expanduser("~/HumanLLM_data")
MIN_REVIEW_LENGTH = 150
amazon_categories = [
        'Arts_Crafts_and_Sewing', 'Automotive', 'Baby_Products', 'Beauty_and_Personal_Care', 'Books', 
        'CDs_and_Vinyl', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics', 
        'Grocery_and_Gourmet_Food', 'Health_and_Household', 'Home_and_Kitchen',
        'Industrial_and_Scientific', 'Sports_and_Outdoors', 'Video_Games',
    ]

def load_model_vllm(model_path, MAX_MODEL_LENGTH=131072):
    model = vllm.LLM(
        model_path,
        max_model_len=MAX_MODEL_LENGTH,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=float(os.getenv("TOOL_VLLM_GPU_MEMORY_UTILIZATION", 0.94)),
        trust_remote_code=True,
        dtype="auto",
        enforce_eager=False
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer

def generate(model, tokenizer, batch_prompts, max_out_length=10240):
    responses = model.generate(
        batch_prompts,
        vllm.SamplingParams(
            n=1,
            temperature=0.3,
            skip_special_tokens=True,
            max_tokens=max_out_length,
            stop=[tokenizer.eos_token, "<|end_of_text|>"]
        ),
        use_tqdm=False,
    )
    results = []
    for i, response in enumerate(responses):
        results.append(response.outputs[0].text)
    return results

def truncate_and_generate(model, tokenizer, prompt, max_in_length, max_out_length):
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_in_length)
    truncated_prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    chat = [{"role": "user", "content": truncated_prompt}]
    text = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)]
    response = generate(model, tokenizer, text, max_out_length=max_out_length)
    return response[0]

def truncate_and_generate_batch(model, tokenizer, prompts, max_in_length, max_out_length):
    encoded_batch = tokenizer(prompts, truncation=True, max_length=max_in_length, add_special_tokens=False)
    truncated_prompts = tokenizer.batch_decode(encoded_batch["input_ids"], skip_special_tokens=True)

    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in truncated_prompts
    ]

    responses = generate(model, tokenizer, chat_texts, max_out_length=max_out_length)
    return responses

def run_clean_jobs(model, tokenizer, start_idx, end_idx, batch_size, max_review=100):
    rewrite_prompt_template = Prompt_amazon_rewrite_review
    tag_prompt_template = Prompt_amazon_tag_review

    for category in amazon_categories:
        print(f"Processing category: {category}")
        indir = f"{HOME_DIR}/amazon/raw_data/{category}"
        outdir = f"{HOME_DIR}/amazon/clean_data/{category}"
        os.makedirs(outdir, exist_ok=True)

        all_files = os.listdir(indir)
        for filename in tqdm(all_files, desc="Processing users", total=len(all_files)):
            if not filename.endswith(".json"):
                continue
            userid = filename.split(".")[0]
            infile = os.path.join(indir, filename)
            outfile = os.path.join(outdir, filename)

            if int(userid) < start_idx or int(userid) >= end_idx:
                continue
            if os.path.exists(outfile):
                print(f"File {outfile} already exists, skipping.")
                continue
            in_data = []
            with open(infile, "r", encoding="utf-8") as f:
                for line in f:
                    in_data.append(json.loads(line))

            batch_indices, batch_prompts = [], []
            cnt = 0
            for i in range(len(in_data)):
                cur_tweet = in_data[i]['text']
                if len(cur_tweet) < MIN_REVIEW_LENGTH:
                    continue
                if cnt >= max_review:
                    break

                rewrite_prompt = rewrite_prompt_template.replace("{review}", cur_tweet)
                batch_prompts.append(rewrite_prompt)
                batch_indices.append(i)
                cnt += 1
            
                if len(batch_indices)>=batch_size:
                    batch_tag_prompts = []
                    batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=10000, max_out_length=10000)
                    for idx, response in zip(batch_indices, batch_response):
                        in_data[idx]['rewrite_blog'] = response
                        tag_prompt = tag_prompt_template.replace("{review}", response)
                        batch_tag_prompts.append(tag_prompt)
                    batch_tag_response = truncate_and_generate_batch(model, tokenizer, batch_tag_prompts, max_in_length=10000, max_out_length=100)
                    for idx, response in zip(batch_indices, batch_tag_response):
                        in_data[idx]['quality_tag'] = response
                    batch_indices, batch_prompts = [], []

            if len(batch_indices) > 0:
                batch_tag_prompts = []
                batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=10000, max_out_length=10000)
                for idx, response in zip(batch_indices, batch_response):
                    in_data[idx]['rewrite_blog'] = response
                    tag_prompt = tag_prompt_template.replace("{review}", response)
                    batch_tag_prompts.append(tag_prompt)
                batch_tag_response = truncate_and_generate_batch(model, tokenizer, batch_tag_prompts, max_in_length=10000, max_out_length=100)
                for idx, response in zip(batch_indices, batch_tag_response):
                    in_data[idx]['quality_tag'] = response

            with open(outfile, "w", encoding="utf-8") as f:
                for item in in_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

def run_persona_jobs(model, tokenizer, start_idx, end_idx, batch_size):
    for category in amazon_categories:
        print(f"Processing category: {category}")

        meta_file = f"{HOME_DIR}/amazon/raw_data/amazon_{category}_content.json"
        meta_data = json.load(open(meta_file, 'r'))

        indir = f"{HOME_DIR}/amazon/clean_data/{category}"
        outdir = f"{HOME_DIR}/amazon/users_persona/{category}"
        prompt_template = Prompt_amazon_user_persona
        gen_content_and_output2file_bydir(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, meta_data, max_in_length=80000, max_out_length=2048, batch_size=batch_size, category=category)
        
        indir = f"{HOME_DIR}/amazon/clean_data/{category}"
        outdir = f"{HOME_DIR}/amazon/users_profile/{category}"
        prompt_template = Prompt_amazon_user_profile
        gen_content_and_output2file_bydir(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, meta_data, max_in_length=80000, max_out_length=4096, batch_size=batch_size, category=category)

def run_clean_jobs_v2(model, tokenizer, start_idx, end_idx, batch_size):
    for category in amazon_categories:
        print(f"Processing category: {category}")
        meta_file = f"{HOME_DIR}/amazon/raw_data/amazon_{category}_content.json"
        meta_data = json.load(open(meta_file, 'r'))

        indir = f"{HOME_DIR}/amazon/clean_data/{category}"
        outdir = f"{HOME_DIR}/amazon/by_users_quality_tagging_v2/{category}"
        prompt_template = PROMPT_raw_content_quality_tag_v2
        gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, meta_data,
                    max_in_length=10000, max_out_length=100, batch_size=batch_size, new_column="blog_quality_v2")

        indir = f"{HOME_DIR}/amazon/clean_data/{category}"
        outdir = f"{HOME_DIR}/amazon/by_users_quality_tagging_v2_rewritten/{category}"
        prompt_template = PROMPT_job_rowwise_clean_blog_step2
        gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, meta_data,
                    max_in_length=10000, max_out_length=10000, batch_size=batch_size, new_column="rewritten_text_v2")

def gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, meta_data, max_in_length, max_out_length, batch_size, data_column='rewrite_blog', new_column='summary'):
    os.makedirs(outdir, exist_ok=True)
    
    cnt = 0
    item_cnt = 0
    infer_cnt = 0
    filename_list = os.listdir(indir) 
    filename_list = random.sample(filename_list, len(filename_list))
    for filename in tqdm(filename_list, desc="Processing users", total=len(filename_list)):
        if not filename.endswith(".json"):
            continue
        userid = filename.split(".")[0]
        infile = os.path.join(indir, filename)
        outfile = os.path.join(outdir, filename)
        if int(userid) < start_idx or int(userid) >= end_idx:
            continue
        if os.path.exists(outfile):
            print(f"File {outfile} already exists, skipping.")
            continue
        in_data = []
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))

        batch_indices, batch_prompts = [], []
        for i in range(len(in_data)):
            item_cnt += 1
            if "quality_tag" not in in_data[i]:
                continue
            quality_tag = extract_tag_content(in_data[i]['quality_tag'], "quality")
            harmless_tag = extract_tag_content(in_data[i]['quality_tag'], "harmless")
            if (quality_tag == "high" or quality_tag == "medium") and harmless_tag == "yes":
                cur_review = in_data[i][data_column]
                cur_item_name = meta_data[in_data[i]['item_id']]['title']
                if '{product_name}' in prompt_template:
                    cur_prompt = prompt_template.replace("{product_name}", cur_item_name).replace("{user_review}", cur_review)
                else:
                    cur_prompt = prompt_template.replace("{user_review}", cur_review)
                batch_prompts.append(cur_prompt)
                batch_indices.append(i)
                infer_cnt += 1
            
            if len(batch_indices) >= batch_size:
                batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
                for idx, response in zip(batch_indices, batch_response):
                    in_data[idx][new_column] = response
                batch_indices, batch_prompts = [], []
        
        if len(batch_indices) > 0:
            batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
            for idx, response in zip(batch_indices, batch_response):
                in_data[idx][new_column] = response
        
        with open(outfile, "w", encoding="utf-8") as f:
            for item in in_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        cnt += 1

    print(f"Total users processed: {cnt}, total items: {item_cnt}, total inferences: {infer_cnt}") 

def find_last_qualified_review_idx(in_data):
    all_qualified_ids = []
    idx = -1
    for i in range(len(in_data)):
        if "quality_tag" in in_data[i]:
            quality_tag = extract_tag_content(in_data[i]['quality_tag'], "quality")
            harmless_tag = extract_tag_content(in_data[i]['quality_tag'], "harmless")
            if (quality_tag == "high" or quality_tag == "medium") and harmless_tag == "yes":
                idx = i
                all_qualified_ids.append(in_data[i]['item_id'])
    return idx, all_qualified_ids

def gen_content_and_output2file_bydir(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, meta_data, max_in_length, max_out_length, batch_size, category):
    outdir_prompt = outdir + "_prompt"
    os.makedirs(outdir_prompt, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    cnt = 0
    review_cnt = 0
    non_review_cnt = 0
    filename_list = os.listdir(indir) 
    filename_list = random.sample(filename_list, len(filename_list))
    batch_userids, batch_prompts = [], []
    for filename in tqdm(filename_list, desc="Processing users", total=len(filename_list)):
        if not filename.endswith(".json"):
            continue
        userid = filename.split(".")[0]
        infile = os.path.join(indir, filename)
        if int(userid) < start_idx or int(userid) >= end_idx:
            continue
        if os.path.exists(os.path.join(outdir, filename.replace(".json", ".txt"))) and os.path.exists(os.path.join(outdir_prompt, filename.replace(".json", ".txt"))):
            print(f"File {os.path.join(outdir, filename)} already exists, skipping.")
            continue
        in_data = []
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))
        
        last_qualified_idx, all_qualified_ids = find_last_qualified_review_idx(in_data)
        if last_qualified_idx < 7: # in this case, we use the last item as target, and do not do imitation task
            target_idx = len(in_data) - 1
            non_review_cnt += 1
        else:
            target_idx = last_qualified_idx
            review_cnt += 1

        content = ""
        begin_idx = max(0, target_idx - 30) # We use the last 30 items as context
        for idx, item in enumerate(in_data[begin_idx:target_idx]):
            item_metadata = meta_data[item['item_id']]
            timestamp = datetime.fromtimestamp(item['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S')
            item_description = f"User bought {item_metadata['title']} on {timestamp}. The rating is {item['rating']}.\n"
            if 'rewrite_blog' in item:
                item_description += f"The review is: {item['rewrite_blog']}\n"
            else:
                item_description += f"The review is: {item['text']}\n"
            item_description += f"The item metadata is: price: {item_metadata['price']}, categories: {item_metadata['categories']}, description: {filter_best_description(item_metadata['description'])[:2000]}\n"
            content += f"\nHistory {idx+1}: {item_description}"

        cur_prompt = prompt_template.replace("{amazon}", content).replace("{category}", category)
        batch_userids.append(userid)
        batch_prompts.append(cur_prompt)

        if len(batch_userids) >= batch_size:
            batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
            for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):
                with open(os.path.join(outdir_prompt, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                    f.write(_prompt)
                with open(os.path.join(outdir, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                    f.write(_response)
            batch_userids = []
            batch_prompts = []
        cnt += 1

    if len(batch_userids) > 0:
        batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
        for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):
            with open(os.path.join(outdir_prompt, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                f.write(_prompt)
            with open(os.path.join(outdir, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                f.write(_response)

    print(f"Total users processed: {cnt}, review users: {review_cnt}, non-review users: {non_review_cnt}")
        

def run_writing_style(model, tokenizer, start_idx, end_idx, batch_size):
    for category in amazon_categories:
        print(f"Processing category: {category}")

        meta_file = f"{HOME_DIR}/amazon/raw_data/amazon_{category}_content.json"
        meta_data = json.load(open(meta_file, 'r'))

        indir = f"{HOME_DIR}/amazon/by_users_quality_tagging_v2_rewritten/{category}"
        indir_tag= f"{HOME_DIR}/amazon/by_users_quality_tagging_v2/{category}"
        outdir = f"{HOME_DIR}/amazon/writing_style/{category}"
        prompt_template = Prompt_amazon_writing_style
        gen_content_and_output2file_bytwodir(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, meta_data, max_in_length=80000, max_out_length=10240, min_post=3, batch_size=batch_size, category=category)

def gen_content_and_output2file_bytwodir(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, meta_data, max_in_length, max_out_length, min_post, batch_size, category, tag_field="blog_quality_v2", blog_field="rewritten_text_v2"):
    outdir_prompt = outdir + "_prompt"
    os.makedirs(outdir_prompt, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    cnt = 0
    skip_user_cnt = 0
    filename_list = os.listdir(indir) 
    filename_list = random.sample(filename_list, len(filename_list))
    batch_userids, batch_prompts = [], []
    for filename in tqdm(filename_list, desc="Processing users", total=len(filename_list)):
        if not filename.endswith(".json"):
            continue
        userid = filename.split(".")[0]
        infile = os.path.join(indir, filename)
        infile_tag = os.path.join(indir_tag, filename)
        if not os.path.exists(infile_tag):
            continue
        if int(userid) < start_idx or int(userid) >= end_idx:
            continue
        if os.path.exists(os.path.join(outdir, filename.replace(".json", ".txt"))) and os.path.exists(os.path.join(outdir_prompt, filename.replace(".json", ".txt"))):
            print(f"File {os.path.join(outdir, filename)} already exists, skipping.")
            continue
        in_data = []
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))
        in_data_tag = []
        with open(infile_tag, "r", encoding="utf-8") as f:
            for line in f:
                in_data_tag.append(json.loads(line))

        qualified_data = []
        for i in range(len(in_data)):
            if tag_field in in_data_tag[i] and blog_field in in_data[i]:
                assert in_data[i]['item_id'] == in_data_tag[i]['item_id'], "ID mismatch between data and tag files"
                unsafe = extract_field_from_json_re_2(in_data_tag[i][tag_field], 'unsafe content')
                social = extract_field_from_json_re_2(in_data_tag[i][tag_field], 'social event')
                if unsafe is None or social is None:
                    continue
                if unsafe.lower() == "no" and social.lower() == "yes":
                    qualified_data.append(in_data[i])

        if len(qualified_data) < min_post:
            skip_user_cnt += 1
            print(f"User {userid} has too few qualified posts, skipping.")
            continue

        content = ""
        for idx, item in enumerate(qualified_data):
            item_metadata = meta_data[item['item_id']]
            item_description = f"User bought {item_metadata['title']}.\nThe review is: {item[blog_field]}\n"
            content += f"\nHistory {idx+1}:\n{item_description}"

        cur_prompt = prompt_template.replace("{user_review}", content)
        batch_userids.append(userid)
        batch_prompts.append(cur_prompt)

        if len(batch_userids) >= batch_size:
            batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
            for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):
                with open(os.path.join(outdir_prompt, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                    f.write(_prompt)
                with open(os.path.join(outdir, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                    f.write(_response)
            batch_userids = []
            batch_prompts = []
        cnt += 1

    if len(batch_userids) > 0:
        batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
        for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):
            with open(os.path.join(outdir_prompt, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                f.write(_prompt)
            with open(os.path.join(outdir, f"{_userid}.txt"), "w", encoding="utf-8") as f:
                f.write(_response)

    print(f"Total users processed: {cnt}, skipped users: {skip_user_cnt}")

def run_review_summary(model, tokenizer, start_idx, end_idx, batch_size):
    for category in amazon_categories:
        print(f"Processing category: {category}")
        meta_file = f"{HOME_DIR}/amazon/raw_data/amazon_{category}_content.json"
        meta_data = json.load(open(meta_file, 'r'))

        indir = f"{HOME_DIR}/amazon/by_users_quality_tagging_v2_rewritten/{category}"
        indir_tag= f"{HOME_DIR}/amazon/by_users_quality_tagging_v2/{category}"
        outdir = f"{HOME_DIR}/amazon/review_summary_v2/{category}"
        prompt_template = Prompt_amazon_review_summary_v2
        gen_content_and_output2file_byfile_2(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, meta_data,
                    max_in_length=10000, max_out_length=2048, min_post=3, max_post=10000, batch_size=batch_size, data_column='rewritten_text_v2')


def gen_content_and_output2file_byfile_2(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, meta_data, max_in_length, max_out_length, min_post, max_post, batch_size, tag_field="blog_quality_v2", data_column='rewrite_blog', new_column='summary'):
    os.makedirs(outdir, exist_ok=True)
    
    cnt = 0
    skip_user_cnt = 0
    filename_list = os.listdir(indir) 
    filename_list = random.sample(filename_list, len(filename_list))
    for filename in tqdm(filename_list, desc="Processing users", total=len(filename_list)):
        if not filename.endswith(".json"):
            continue
        userid = filename.split(".")[0]
        infile = os.path.join(indir, filename)
        outfile = os.path.join(outdir, filename)
        infile_tag = os.path.join(indir_tag, filename)
        if not os.path.exists(infile_tag):
            continue
        if int(userid) < start_idx or int(userid) >= end_idx:
            continue
        if os.path.exists(outfile):
            print(f"File {outfile} already exists, skipping.")
            continue
        in_data = []
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))
        in_data_tag = []
        with open(infile_tag, "r", encoding="utf-8") as f:
            for line in f:
                in_data_tag.append(json.loads(line))

        qualified_data = []
        for i in range(len(in_data)):
            if tag_field in in_data_tag[i] and data_column in in_data[i]:
                assert in_data[i]['item_id'] == in_data_tag[i]['item_id'], "ID mismatch between data and tag files"
                unsafe = extract_field_from_json_re_2(in_data_tag[i][tag_field], 'unsafe content')
                social = extract_field_from_json_re_2(in_data_tag[i][tag_field], 'social event')
                if unsafe is None or social is None:
                    continue
                if unsafe.lower() == "no" and social.lower() == "yes":
                    qualified_data.append(in_data[i])
        if len(qualified_data) < min_post:
            skip_user_cnt += 1
            print(f"User {userid} has too few qualified posts, skipping.")
            continue
        qualified_data = qualified_data[:max_post]

        batch_indices, batch_prompts = [], []
        for i in range(len(qualified_data)):
            cur_review = qualified_data[i][data_column]
            cur_item_name = meta_data[qualified_data[i]['item_id']]['title']
            if '{product_name}' in prompt_template:
                cur_prompt = prompt_template.replace("{product_name}", cur_item_name).replace("{user_review}", cur_review)
            else:
                cur_prompt = prompt_template.replace("{user_review}", cur_review)
            batch_prompts.append(cur_prompt)
            batch_indices.append(i)
            
            if len(batch_indices) >= batch_size:
                batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
                for idx, response in zip(batch_indices, batch_response):
                    qualified_data[idx][new_column] = response
                batch_indices, batch_prompts = [], []
        
        if len(batch_indices) > 0:
            batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=max_in_length, max_out_length=max_out_length)
            for idx, response in zip(batch_indices, batch_response):
                qualified_data[idx][new_column] = response
        
        with open(outfile, "w", encoding="utf-8") as f:
            for item in qualified_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        cnt += 1

    print(f"Total users processed: {cnt}, skipped users: {skip_user_cnt}") 

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0, help="start index (Closed)")
    parser.add_argument("--end_idx", type=int, default=2000000000, help="end index (non-Closed)")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    args = parser.parse_args()

    model_path = f"{HOME_DIR}/Llama-3.3-70B-Instruct"
    model, tokenizer = load_model_vllm(model_path)

    run_clean_jobs(model, tokenizer, args.start_idx, args.end_idx, args.batch_size, max_review=100)

    run_persona_jobs(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_clean_jobs_v2(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_writing_style(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_review_summary(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)