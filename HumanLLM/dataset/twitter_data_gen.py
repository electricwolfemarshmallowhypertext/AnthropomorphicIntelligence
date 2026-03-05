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

from twitter_prompt import *
from utils import extract_tag_content, extract_field_from_json_re_2

HOME_DIR = os.path.expanduser("~/HumanLLM_data")

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

def run_clean_jobs(model, tokenizer, start_idx, end_idx, batch_size):
    rewrite_prompt_template = Prompt_twitter_rewrite_blog
    tag_prompt_template = Prompt_twitter_tag_blog
    indir = f"{HOME_DIR}/twitter/raw_data"
    outdir = f"{HOME_DIR}/twitter/clean_data"
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
        for i in range(len(in_data)):
            cur_tweet = in_data[i]['tweet']
            rewrite_prompt = rewrite_prompt_template.replace("{tweet}", cur_tweet)
            batch_prompts.append(rewrite_prompt)
            batch_indices.append(i)
        
            if len(batch_indices)>=batch_size:
                batch_tag_prompts = []
                batch_response = truncate_and_generate_batch(model, tokenizer, batch_prompts, max_in_length=10000, max_out_length=10000)
                for idx, response in zip(batch_indices, batch_response):
                    in_data[idx]['rewrite_blog'] = response
                    tag_prompt = tag_prompt_template.replace("{tweet}", response)
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
                tag_prompt = tag_prompt_template.replace("{tweet}", response)
                batch_tag_prompts.append(tag_prompt)
            batch_tag_response = truncate_and_generate_batch(model, tokenizer, batch_tag_prompts, max_in_length=10000, max_out_length=100)
            for idx, response in zip(batch_indices, batch_tag_response):
                in_data[idx]['quality_tag'] = response

        with open(outfile, "w", encoding="utf-8") as f:
            for item in in_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

def run_persona_jobs(model, tokenizer, start_idx, end_idx, batch_size):
    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_persona_v2(medium_high)"
    prompt_template = Prompt_twitter_user_persona_v2
    gen_content_and_output2file_bydir(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=2048, min_post=5, batch_size=batch_size)
    
    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_profile_v2(medium_high)"
    prompt_template = Prompt_twitter_user_profile_v2
    gen_content_and_output2file_bydir(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=4096, min_post=5, batch_size=batch_size)

    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_stories(medium_high)"
    prompt_template = Prompt_twitter_user_stories
    gen_content_and_output2file_bydir(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, min_post=5, batch_size=batch_size)

def run_scenario_question_answer_from_single_blog_jobs(model, tokenizer, start_idx, end_idx, batch_size):
    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/scenario_question_answer_from_single_blog_emphisize_thoughts(medium_high)"
    prompt_template = Prompt_twitter_scenario_question_answer_from_single_blog_emphisize_thoughts
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=20, batch_size=batch_size)

    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/scenario_question_answer_from_single_blog_emphisize_actions(medium_high)"
    prompt_template = Prompt_twitter_scenario_question_answer_from_single_blog_emphisize_actions
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=20, batch_size=batch_size)

    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/scenario_question_answer_from_single_blog(medium_high)"
    prompt_template = Prompt_twitter_scenario_question_answer_from_single_blog
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=20, batch_size=batch_size)

    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/scenario_question_answer_from_single_blog_emphasizereason_v3(medium_high)"
    prompt_template = PROMPT_twitter_rowwise_scenario_question_answer_from_single_blog_emphasizereason_v3
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=60, batch_size=batch_size)

def run_stories_jobs(model, tokenizer, start_idx, end_idx, batch_size):
    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_long_scenario_from_single_blog(medium_high)"
    prompt_template = Prompt_twitter_long_scenario_from_single_blog
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=40, batch_size=batch_size)

    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_sing_long_story(medium_high)"
    prompt_template = Prompt_twitter_single_long_story
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=40, batch_size=batch_size)

    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_single_long_story_focusonbehavior(medium_high)"
    prompt_template = Prompt_twitter_single_long_story_focusonbehavior
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=40, batch_size=batch_size)

    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_single_long_thought(medium_high)"
    prompt_template = Prompt_twitter_single_long_thought
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=40, batch_size=batch_size)
    
    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/users_thoughts_single_blog(medium_high)"
    prompt_template = Prompt_twitter_thoughts_single_blog
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=8192, 
                min_post=5, max_post=40, batch_size=batch_size)

def gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length, max_out_length, min_post, max_post, batch_size, data_column='rewrite_blog', new_column='sqa_singleblog'):
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
        if int(userid) < start_idx or int(userid) >= end_idx:
            continue
        if os.path.exists(outfile):
            print(f"File {outfile} already exists, skipping.")
            continue
        in_data = []
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))
        
        high_quality_data = []
        medium_quality_data = []
        for i in range(len(in_data)):
            quality_tag = extract_tag_content(in_data[i]['quality_tag'], "quality")
            harmless_tag = extract_tag_content(in_data[i]['quality_tag'], "harmless")
            if quality_tag == "high" and harmless_tag == "yes":
                high_quality_data.append(in_data[i])
            elif quality_tag == "medium" and harmless_tag == "yes":
                medium_quality_data.append(in_data[i])
        qualified_data = high_quality_data + medium_quality_data # prefer high quality data
        if len(qualified_data) < min_post:
            skip_user_cnt += 1
            print(f"User {userid} has too few qualified posts, skipping.")
            continue
        
        batch_indices, batch_prompts = [], []
        for i in range(len(qualified_data)):
            if i >= max_post:
                break
            cur_tweet = qualified_data[i][data_column]
            rewrite_prompt = prompt_template.replace("{tweet}", cur_tweet)
            batch_prompts.append(rewrite_prompt)
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

def gen_content_and_output2file_bydir(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length, max_out_length, min_post, batch_size):
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
        if int(userid) < start_idx or int(userid) >= end_idx:
            continue
        if os.path.exists(os.path.join(outdir, filename.replace(".json", ".txt"))) and os.path.exists(os.path.join(outdir_prompt, filename.replace(".json", ".txt"))):
            print(f"File {os.path.join(outdir, filename)} already exists, skipping.")
            continue
        in_data = []
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))
        
        qualified_data = []
        for i in range(len(in_data)):
            quality_tag = extract_tag_content(in_data[i]['quality_tag'], "quality")
            harmless_tag = extract_tag_content(in_data[i]['quality_tag'], "harmless")
            if (quality_tag == "high" or quality_tag == "medium") and harmless_tag == "yes":
                qualified_data.append(in_data[i])
        if len(qualified_data) < min_post:
            skip_user_cnt += 1
            print(f"User {userid} has too few qualified posts, skipping.")
            continue
        
        qualified_data = [[item['rewrite_blog'], datetime.fromisoformat(item['date']).astimezone(timezone.utc), item['date']] for item in qualified_data]
        qualified_data.sort(key=lambda x: x[1], reverse=False)

        content = ""
        for idx, item in enumerate(qualified_data):
            content += f"\n\nPost {idx+1}:\nDate: {item[2]}\n{item[0]}"
        
        cur_prompt = prompt_template.replace("{tweet}", content)
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

def run_clean_jobs_v2(model, tokenizer, start_idx, end_idx, batch_size):
    prompt_template = PROMPT_raw_content_quality_tag_v2
    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/by_users_quality_tagging_v2"
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=10000, max_out_length=100, 
                min_post=3, max_post=100, batch_size=batch_size, new_column="blog_quality_v2")

    prompt_template = PROMPT_job_rowwise_clean_blog_step2
    indir = f"{HOME_DIR}/twitter/clean_data"
    outdir = f"{HOME_DIR}/twitter/by_users_quality_tagging_v2_rewritten"
    gen_content_and_output2file_byfile(model, tokenizer, indir, outdir, prompt_template, start_idx, end_idx, max_in_length=10000, max_out_length=10000, 
                min_post=3, max_post=100, batch_size=batch_size, new_column="rewritten_text_v2")

def run_writing_style(model, tokenizer, start_idx, end_idx, batch_size):
    indir = f"{HOME_DIR}/twitter/by_users_quality_tagging_v2_rewritten"
    indir_tag = f"{HOME_DIR}/twitter/by_users_quality_tagging_v2"
    outdir = f"{HOME_DIR}/twitter/writing_style"
    prompt_template = PROMPT_twitter_writing_style
    gen_content_and_output2file_bytwodir(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, max_in_length=80000, max_out_length=10240, min_post=3, batch_size=batch_size)

def gen_content_and_output2file_bytwodir(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, max_in_length, max_out_length, min_post, batch_size, tag_field="blog_quality_v2", blog_field="rewritten_text_v2"):
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
                assert in_data[i]['id'] == in_data_tag[i]['id'], "ID mismatch between data and tag files"
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
        
        qualified_data = [[item[blog_field], datetime.fromisoformat(item['date']).astimezone(timezone.utc), item['date']] for item in qualified_data]
        qualified_data.sort(key=lambda x: x[1], reverse=False)

        content = ""
        for idx, item in enumerate(qualified_data):
            content += f"\n\nPost {idx+1}:\n{item[0]}"
        
        cur_prompt = prompt_template.replace("{tweet}", content)
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

def run_post_summary(model, tokenizer, start_idx, end_idx, batch_size):
    indir = f"{HOME_DIR}/twitter/by_users_quality_tagging_v2_rewritten"
    indir_tag = f"{HOME_DIR}/twitter/by_users_quality_tagging_v2"

    outdir = f"{HOME_DIR}/twitter/post_summary_v2"
    prompt_template = Prompt_twitter_post_summary_v2
    gen_content_and_output2file_byfile_2(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, max_in_length=10000, max_out_length=2048, min_post=3, max_post=10000, batch_size=batch_size, data_column='rewritten_text_v2')

def gen_content_and_output2file_byfile_2(model, tokenizer, indir, indir_tag, outdir, prompt_template, start_idx, end_idx, max_in_length, max_out_length, min_post, max_post, batch_size, tag_field="blog_quality_v2", data_column='rewrite_blog', new_column='sqa_singleblog'):
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
                assert in_data[i]['id'] == in_data_tag[i]['id'], "ID mismatch between data and tag files"
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
        
        batch_indices, batch_prompts = [], []
        for i in range(len(qualified_data)):
            if i >= max_post:
                break
            cur_tweet = qualified_data[i][data_column]
            rewrite_prompt = prompt_template.replace("{tweet}", cur_tweet)
            batch_prompts.append(rewrite_prompt)
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
    parser.add_argument("--end_idx", type=int, default=20000000, help="end index (non-Closed)")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    args = parser.parse_args()

    model_path = f"{HOME_DIR}/Llama-3.3-70B-Instruct"
    model, tokenizer = load_model_vllm(model_path)

    run_clean_jobs(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_persona_jobs(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_scenario_question_answer_from_single_blog_jobs(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_stories_jobs(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_clean_jobs_v2(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_writing_style(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)

    run_post_summary(model, tokenizer, args.start_idx, args.end_idx, args.batch_size)