import pandas as pd
import os
import torch
from copy import deepcopy
import time
import json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import vllm
from os.path import expanduser  
from tqdm import tqdm
import hashlib
import random
import re
from datetime import datetime, timezone
# set random seed with time ticks  
random.seed(int(time.time()))

## import constants from prompts.py under the same directory of this file
from judger_prompts import *
from utils import *

HOME_DIR = os.path.expanduser("~/HumanLLM_data")
USER_TEXT_MAX_LEN = 42000 
APPLY_CHAT_TEMPLATE_PARAM = {
    "tokenize": False, 
    "add_generation_prompt": True
}

def load_model_vllm(model_path, MAX_MODEL_LENGTH=40960):
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if total_memory >= 70 * 1024**3: 
        model = vllm.LLM(
            model_path,
            max_model_len=MAX_MODEL_LENGTH,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=float(os.getenv("TOOL_VLLM_GPU_MEMORY_UTILIZATION", 0.9)),
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=False
        )
    else:
        model = vllm.LLM(
            model_path,
            max_model_len=MAX_MODEL_LENGTH,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=float(os.getenv("TOOL_VLLM_GPU_MEMORY_UTILIZATION", 0.93)),
            trust_remote_code=True, 
            dtype="auto",
            max_num_seqs=8,
            max_num_batched_tokens=120000,   
            enforce_eager=False
        )
    tokenizer = model.get_tokenizer()
    return model, tokenizer

def safe_save2file(filename, df):
    max_retries = 5
    success = False
    for attempt in range(0, max_retries):
        try:  
            with open(filename, "w", encoding="utf-8") as f:
                for item in df:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            success = True
            break  
        except Exception as e:
            print(f"[ERROR] Attempt {attempt + 1} failed in saving file {filename}: {e}")
            time.sleep(5)  # Wait before retrying  
            continue  
    return success
            
def generate(model, tokenizer, batch_prompts):
    responses = model.generate(
        batch_prompts,
        vllm.SamplingParams(
            n=1,
            temperature=0.3,
            skip_special_tokens=True,
            max_tokens=10240,
            stop=[tokenizer.eos_token, "<|end_of_text|>"]
        ),
        use_tqdm=False,
    )
    results = []
    for i, response in enumerate(responses):
        results.append(response.outputs[0].text)
    return results

def gen_content_and_output2file_byfile_4judger(
        model, tokenizer, user_prompt, infile, outfile, max_K=-1, 
        new_column="result", data_column_rawblog="cleaned_text", data_column_generated="generated_text",
        override=False, cutoff_string=None):
    ## cutoff_string is used to parse final results from deep thinking mdoels, which is usually </think>\n\n
  
    BATCH_SIZE = 16 
    PROMPT_SEG01 = "\nBelow is the original blog:\n"
    PROMPT_SEG02 = "\n\nBelow is the generated content:\n"
 

    if os.path.exists(outfile) and not override:
        print(f"file {outfile} already exists, skip")
        return
    
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)
    
    try:
        # df = pd.read_csv(infile, header=0)   
        ## read data file as pandas dataframe
        ## data_column_rawblog and data_column_generated columns should in str type 
        df = []
        with open(infile, "r", encoding="utf-8") as f:
            for line in f:
                df.append(json.loads(line))
    except Exception as e:
        print(f"file {infile} is not a valid csv file, skip")
        return
    
    ## check if df is empty 
    if df is None or len(df) == 0:
        print(f"file {infile} is empty, skip")
        return
    
    ## iterate the dataframe  
    batch_indices, batch_prompts = [], []
    ## do not use tqdm here to enumerate rows
    # for i, row in tqdm(df.iterrows(), total=len(df), desc="Blog Cleaning"):
    for i, row in enumerate(df):
        if data_column_generated not in row or not row[data_column_generated] or len(row[data_column_generated]) < 20:
            df[i][new_column] = "NULL"
            continue
        try: 
            content = user_prompt + PROMPT_SEG01 + row[data_column_rawblog] 
        except Exception as e:
            print(f"error: {e}")
            print(f"row data: {row[data_column_rawblog]}")
            raise e
        if len(content) > USER_TEXT_MAX_LEN:
            content = content[:USER_TEXT_MAX_LEN]
        content += PROMPT_SEG02 + row[data_column_generated] + "\n\n"
        chat = [
            # {"role": "system", "content":  "You are a helpful assistant that help people extract high-quality datasets.",}
        ]
        chat.append({"role": "user", "content": content})
        text = tokenizer.apply_chat_template(chat, **APPLY_CHAT_TEMPLATE_PARAM)
        batch_indices.append(i)
        batch_prompts.append(text)          
        
        if len(batch_indices) >= BATCH_SIZE: 
            batch_response = generate(model, tokenizer, batch_prompts) 
            for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
                if cutoff_string is not None and cutoff_string in _response:
                    _response = _response.split(cutoff_string)[1]
                df[_index][new_column] = _response    
                 
            batch_indices, batch_prompts = [], []
        
        if max_K>0 and i > max_K: 
            break 
    if len(batch_indices) > 0:
        batch_response = generate(model, tokenizer, batch_prompts) 
        for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
            if cutoff_string is not None and cutoff_string in _response:
                _response = _response.split(cutoff_string)[1]
            df[_index][new_column] = _response    
    
    ## only keep columns: id,cleaned_text,blog_quality,new_column
    new_df = []
    for row in df:
        # df = [{k: v for k, v in row.items() if k in ["id", new_column, data_column_rawblog, data_column_generated]} for row in df]
        if row[new_column] and len(row[new_column]) > 5:
            new_row = {k: v for k, v in row.items() if k in ["id", "domain", "type", "quality_tag", new_column, data_column_rawblog, data_column_generated]}
            new_df.append(new_row)

    ## if number of rows > 0:
    if len(new_df) > 0:
        safe_save2file(outfile, new_df)


def job_rowwise_judger_socialqa_fulltypes(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, outdir_flag=""):
    user_prompt = PROMPT_judger_socialqa_fulltypes_v2
    dir_prefix = f"{HOME_DIR}"
    
    dir_to_be_processed = [
        f"{dir_prefix}/reddit/scenario_question_answer_from_single_blog_emphisize_actions(medium_high)" ,         
        f"{dir_prefix}/reddit/scenario_question_answer_from_single_blog_emphisize_thoughts(medium_high)" ,
        f"{dir_prefix}/reddit/scenario_question_answer_from_single_blog(medium_high)" ,
        f"{dir_prefix}/reddit/scenario_question_answer_from_single_blog_emphasizereason_v3(medium_high)",
        f"{dir_prefix}/twitter/scenario_question_answer_from_single_blog_emphisize_actions(medium_high)" ,        
        f"{dir_prefix}/twitter/scenario_question_answer_from_single_blog_emphisize_thoughts(medium_high)" ,    
        f"{dir_prefix}/twitter/scenario_question_answer_from_single_blog(medium_high)",    
        f"{dir_prefix}/twitter/scenario_question_answer_from_single_blog_emphasizereason_v3(medium_high)", 
    ]
    
    data_column_to_be_used = [
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
    ]
    data_column_rawblog = "rewrite_blog"
    
    for data_column, indir in zip(data_column_to_be_used, dir_to_be_processed): 
        outdir = os.path.join(indir, f"_judger{outdir_flag}")
 
        os.makedirs(outdir, exist_ok=True)
        ## get all filenames in indir 
        allfiles = os.listdir(indir)  
        ## randomly shuffle the files 
        allfiles = random.sample(allfiles, len(allfiles))
        
        cnt = 0
        for filename in tqdm(allfiles, desc="Processing users", total=len(allfiles)):
            if not filename.endswith(".json"):
                continue
            userid = filename.split(".")[0]
            infile = os.path.join(indir, filename)
            outfile = os.path.join(outdir, filename)
            if base > 0 and split >= 0 and userid.isdigit():
                if int(userid)%base != split:
                    continue 
            gen_content_and_output2file_byfile_4judger(
                model, tokenizer, 
                user_prompt, infile, outfile, 
                max_K=max_K, 
                new_column="quality_result", 
                data_column_rawblog=data_column_rawblog, data_column_generated=data_column,
                override=False, cutoff_string=cutoff_string)
                
            cnt += 1
            if max_K > 0 and cnt > max_K:
                break

def job_rowwise_judger_scenario_fulltypes(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, outdir_flag=""):
    user_prompt = PROMPT_socialscenarios_singleblog_fulltypes
    dir_prefix = f"{HOME_DIR}"
    
    dir_to_be_processed = [
        f"{dir_prefix}/reddit/users_long_scenario_from_single_blog(medium_high)",
        f"{dir_prefix}/reddit/users_sing_long_story(medium_high)",
        f"{dir_prefix}/reddit/users_single_long_story_focusonbehavior(medium_high)",
        f"{dir_prefix}/reddit/users_thoughts_single_blog(medium_high)",
        f"{dir_prefix}/reddit/users_single_long_thought(medium_high)",
        f"{dir_prefix}/twitter/users_long_scenario_from_single_blog(medium_high)",
        f"{dir_prefix}/twitter/users_sing_long_story(medium_high)",
        f"{dir_prefix}/twitter/users_single_long_story_focusonbehavior(medium_high)",
        f"{dir_prefix}/twitter/users_thoughts_single_blog(medium_high)",
        f"{dir_prefix}/twitter/users_single_long_thought(medium_high)",
    ]
    
    data_column_to_be_used = [
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
        "sqa_singleblog",
    ]
    data_column_rawblog = "rewrite_blog"
    
    for data_column, indir in  zip(data_column_to_be_used, dir_to_be_processed): 
        outdir = os.path.join(indir, f"_judger{outdir_flag}")
 
        os.makedirs(outdir, exist_ok=True)
        ## get all filenames in indir 
        allfiles = os.listdir(indir)  
        ## randomly shuffle the files 
        allfiles = random.sample(allfiles, len(allfiles))
        
        cnt = 0
        for filename in tqdm(allfiles, desc="Processing users", total=len(allfiles)):
            if not filename.endswith(".json"):
                continue
            userid = filename.split(".")[0]
            infile = os.path.join(indir, filename)
            outfile = os.path.join(outdir, filename)
            if base > 0 and split >= 0 and userid.isdigit():
                if int(userid)%base != split:
                    continue 
            gen_content_and_output2file_byfile_4judger(
                model, tokenizer, 
                user_prompt, infile, outfile, 
                max_K=max_K, 
                new_column="quality_result", 
                data_column_rawblog=data_column_rawblog, data_column_generated=data_column,
                override=False, cutoff_string=cutoff_string)
                
            cnt += 1
            if max_K > 0 and cnt > max_K:
                break

def job_userwise_judger_user_profiles_fulltypes(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, outdir_flag=""): 
    dir_prefix = f"{HOME_DIR}"
    
    dir_to_raw_blogs = [
        f"{dir_prefix}/reddit/clean_data",
        f"{dir_prefix}/reddit/clean_data",
        f"{dir_prefix}/reddit/clean_data",
        f"{dir_prefix}/twitter/clean_data",
        f"{dir_prefix}/twitter/clean_data",
        f"{dir_prefix}/twitter/clean_data",
    ]

    dir_to_be_processed = [         
        f"{dir_prefix}/reddit/users_persona_v2(medium_high)" , 
        f"{dir_prefix}/reddit/users_profile_v2(medium_high)" ,
        f"{dir_prefix}/reddit/users_stories(medium_high)",
        f"{dir_prefix}/twitter/users_persona_v2(medium_high)" , 
        f"{dir_prefix}/twitter/users_profile_v2(medium_high)" ,
        f"{dir_prefix}/twitter/users_stories(medium_high)"
    ]
    prompts_to_be_used = [
        PROMPT_judger_user_persona,
        PROMPT_judger_user_profile, 
        PROMPT_judger_user_stories,
        PROMPT_judger_user_persona,
        PROMPT_judger_user_profile, 
        PROMPT_judger_user_stories
    ]
    
    for user_prompt, indir, dir_to_raw in zip(prompts_to_be_used, dir_to_be_processed, dir_to_raw_blogs): 
        outdir = os.path.join(indir, f"_judger{outdir_flag}")
                
        gen_content_and_output2file_bydir_profile_judger(
            model, tokenizer, user_prompt, indir, outdir, dir_to_raw, max_K, base, split, 
            check_quality=True, record_threshold=3, cutoff_string=cutoff_string)

def gen_content_and_output2file_bydir_profile_judger(model, tokenizer, user_prompt, indir, outdir, dir_to_raw_blogs,
                                      max_K=-1, base=-1, split=0, check_quality=False, 
                                      record_threshold=5, cutoff_string=None):
        
    BATCH_SIZE = 16
    PROMPT_SEG01 = "\nBelow are the original posts:\n"
    PROMPT_SEG02 = "\n\nBelow are the generated content:\n"

    os.makedirs(outdir, exist_ok=True)
    ## for each file under indir, process data 
    cnt = 0
    skip_user_cnt = 0
    filename_list = os.listdir(indir) 
    filename_list = random.sample(filename_list, len(filename_list))
    batch_userids, batch_prompts = [], []
    for filename in tqdm(filename_list, desc="Processing users", total=len(filename_list)):
        if not filename.endswith(".txt"): ## persona etc files are .txt files
            continue 
        userid = filename.split(".")[0]   
        if base > 0 and split >= 0 and userid.isdigit():
            if int(userid)%base != split:
                continue

        ## judger output file is .txt 
        if os.path.exists(os.path.join(outdir, filename)):
            print(f"file {os.path.join(outdir,filename)} already exists, skip")
            continue
        
        with open(os.path.join(indir, filename), "r") as f:
            user_profile = f.read()
        raw_blog_filename = filename.replace(".txt", ".json")
        in_data = []
        with open(os.path.join(dir_to_raw_blogs, raw_blog_filename), "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))

        qualified_data = []
        for i in range(len(in_data)):
            if 'type' not in in_data[i] or in_data[i]['type'] == "t3": #only use posts, not comments
                quality_tag = extract_tag_content(in_data[i]['quality_tag'], "quality")
                harmless_tag = extract_tag_content(in_data[i]['quality_tag'], "harmless")
                if (quality_tag == "high" or quality_tag == "medium") and harmless_tag == "yes":
                    qualified_data.append(in_data[i])

        if len(qualified_data) < record_threshold:
            skip_user_cnt += 1
            continue

        qualified_data = [[item['rewrite_blog'], item['created_utc'] if 'created_utc' in item else datetime.fromisoformat(item['date']).astimezone(timezone.utc)] for item in qualified_data]
        if 'created_utc' in qualified_data[0]:
            qualified_data.sort(key=lambda x: int(x[1]), reverse=False)
        else:
            qualified_data.sort(key=lambda x: x[1], reverse=False)

        chat = [
            # {"role": "system", "content":  "You are a helpful assistant that help people extract high-quality datasets.",}
        ]   
        content = user_prompt + PROMPT_SEG01
        for idx, item in enumerate(qualified_data):
            content += f"Post {idx + 1}: " + "\n" + item[0] + "\n\n"

        if len(content) > USER_TEXT_MAX_LEN:
            content = content[:USER_TEXT_MAX_LEN]

        content += PROMPT_SEG02 + user_profile + "\n\n"

        chat.append({"role": "user", "content": content})
        text = tokenizer.apply_chat_template(chat, **APPLY_CHAT_TEMPLATE_PARAM)

        batch_userids.append(userid)
        batch_prompts.append(text)

        if len(batch_userids) >= BATCH_SIZE: 
            batch_response = generate(model, tokenizer, batch_prompts) 
            for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):  
                if cutoff_string is not None and cutoff_string in _response:
                    _response = _response.split(cutoff_string)[1]                   
                with open(os.path.join(outdir, str(_userid) + ".txt"), "w") as f:
                    f.write(_response)
            batch_userids, batch_prompts = [], []
        cnt += 1 
        if max_K>0 and cnt > max_K: 
            break 
    
    if len(batch_userids) > 0:
        batch_response = generate(model, tokenizer, batch_prompts) 
        for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):    
            if cutoff_string is not None and cutoff_string in _response:
                _response = _response.split(cutoff_string)[1]                   
            with open(os.path.join(outdir, str(_userid) + ".txt"), "w") as f:
                f.write(_response)
    print(f"processed {cnt} users, skipped {skip_user_cnt} users")

def run_all_jobs(base, split, max_K=-1):
    ### for the judger jobs, use this model as default
    model_path = f"Qwen/Qwen2.5-72B-Instruct"
    outdir_flag = "_Qwen2.5-72B"
    
    input_length = 40960
    if "Qwen2.5" in model_path:
        input_length = 32767
        USER_TEXT_MAX_LEN = int(input_length*0.8)
    model, tokenizer = load_model_vllm(model_path, MAX_MODEL_LENGTH=input_length)
    
    cutoff_string=None
    for _name in ["QwQ-32B", "Qwen3-"]:
        if _name in model_path:
            cutoff_string="</think>"
            break 
    if "Qwen3" in model_path:
        APPLY_CHAT_TEMPLATE_PARAM["enable_thinking"]=False ## disable thinking in qwen3
    
    
    job_rowwise_judger_socialqa_fulltypes(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, outdir_flag=outdir_flag) 
    job_userwise_judger_user_profiles_fulltypes(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, outdir_flag=outdir_flag)
    job_rowwise_judger_scenario_fulltypes(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, outdir_flag=outdir_flag)



if __name__ == "__main__":
    ## parse two arguments: base, split
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=-1)
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()
    base, split = args.base, args.split
 
    run_all_jobs(base, split, max_K=-1)
