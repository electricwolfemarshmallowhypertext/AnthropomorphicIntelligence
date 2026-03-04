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
# set random seed with time ticks  
random.seed(int(time.time()))

## import constants from prompts.py under the same directory of this file
from .blogger_prompt import *
from .utils import *

HOME_DIR = os.path.expanduser("~/HumanLLM_data")
USER_TEXT_MAX_LEN = 32000 
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
    PROMPT_SEG02 = "\nBelow is the generated content:\n"
 

    if os.path.exists(outfile) and not override:
        print(f"file {outfile} already exists, skip")
        return
    
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)
    
    try:
        # df = pd.read_csv(infile, header=0)   
        ## read data file as pandas dataframe
        ## data_column_rawblog and data_column_generated columns should in str type 
        df = pd.read_csv(infile, header=0, dtype={data_column_rawblog: str, data_column_generated: str})
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
    for i, row in df.iterrows():
        if not row[data_column_generated] or len(row[data_column_generated]) < 20:
            df.loc[i, new_column] = "NULL"
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
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        batch_indices.append(i)
        batch_prompts.append(text)          
        
        if len(batch_indices) >= BATCH_SIZE: 
            batch_response = generate(model, tokenizer, batch_prompts) 
            for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
                if cutoff_string is not None and cutoff_string in _response:
                    _response = _response.split(cutoff_string)[1]
                df.loc[_index, new_column] = _response    
                 
            batch_indices, batch_prompts = [], []
        
        if max_K>0 and i > max_K: 
            break 
    if len(batch_indices) > 0:
        batch_response = generate(model, tokenizer, batch_prompts) 
        for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
            if cutoff_string is not None and cutoff_string in _response:
                _response = _response.split(cutoff_string)[1]
            df.loc[_index, new_column] = _response    
    
    ## only keep columns: id,cleaned_text,blog_quality,new_column
    df = df[["id", new_column, data_column_rawblog, data_column_generated]] 
    ## remove rows that the new_column is empty or length is less than 5
    df = df[df[new_column].apply(lambda x: len(str(x)) > 5)]
    ## if number of rows > 0:
    if len(df) > 0:
        safe_save2file(outfile, df)

def job_rowwise_judger_socialqa_fulltypes(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, outdir_flag=""):
    user_prompts = [
        PROMPT_judger_socialqa_fulltypes_v2,
        PROMPT_judger_socialqa_fulltypes_v2,
        PROMPT_judger_socialqa_fulltypes_v2,
        PROMPT_judger_socialqa_fulltypes_v2,
    ]
    
    dir_to_be_processed = [
        f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_emphasizeaction_v2(use_quality_high)" ,         
        f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_emphasizereason_v2(use_quality_high)" ,
        f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_emphasizethoughts_v2(use_quality_high)" ,
        f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_v2(use_quality_high)" ,              
    ]
    
    data_column_to_be_used = [  
        "result",
        "result",
        "result",
        "result",
    ]
    data_column_rawblog = "cleaned_text"
    
    for data_column, indir, user_prompt in  zip(data_column_to_be_used, dir_to_be_processed, user_prompts): 
        outdir = os.path.join(indir, f"_judger{outdir_flag}")
 
        os.makedirs(outdir, exist_ok=True)
        ## get all filenames in indir 
        allfiles = os.listdir(indir)  
        ## randomly shuffle the files 
        allfiles = random.sample(allfiles, len(allfiles))
        
        cnt = 0
        for filename in tqdm(allfiles, desc="Processing users", total=len(allfiles)):
            if not filename.endswith(".csv"):
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

    dir_to_be_processed = [
        f"{HOME_DIR}/blogger/users_long_scenario_from_single_blog(use_quality_high)" ,
        f"{HOME_DIR}/blogger/users_sing_long_story(use_quality_high)" ,
        f"{HOME_DIR}/blogger/users_single_long_story_focusonbehavior(user_quality_high)" ,
        f"{HOME_DIR}/blogger/users_thoughts_single_blog(user_quality_high)" ,
    ]
    
    data_column_to_be_used = [
        "longscenario_singleblog",
        "single_long_story",
        "single_long_story_focusonbebaviors",
        "thoughts_singleblog",
    ]
    data_column_rawblog = "cleaned_text"
    
    for data_column, indir in  zip(data_column_to_be_used, dir_to_be_processed): 
        outdir = os.path.join(indir, f"_judger{outdir_flag}")
 
        os.makedirs(outdir, exist_ok=True)
        ## get all filenames in indir 
        allfiles = os.listdir(indir)  
        ## randomly shuffle the files 
        allfiles = random.sample(allfiles, len(allfiles))
        
        cnt = 0
        for filename in tqdm(allfiles, desc="Processing users", total=len(allfiles)):
            if not filename.endswith(".csv"):
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

def gen_content_and_output2file_bydir_profile_judger(model, tokenizer, user_prompt, indir, outdir, dir_to_raw_blogs,
                                      max_K=-1, base=-1, split=0, check_quality=False, 
                                      record_threshold=5, cutoff_string=None):
        
    BATCH_SIZE = 16
    PROMPT_SEG01 = "\nBelow are the original blogs:\n"
    PROMPT_SEG02 = "\nBelow are the generated content:\n"
 
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
            
        with open(os.path.join(indir, filename), "r") as f:
            user_profile = f.read()
        
        raw_blog_filename = filename.replace(".txt", ".csv")
        try:
            df = pd.read_csv(os.path.join(dir_to_raw_blogs, raw_blog_filename), header=0)  
        except Exception as e:
            print(f"file {os.path.join(dir_to_raw_blogs, raw_blog_filename)} is not a valid csv file, skip")
            continue
        
        ## judger output file is .txt 
        if os.path.exists(os.path.join(outdir, filename.replace(".csv", ".txt"))):
            print(f"file {os.path.join(outdir,filename)} already exists, skip")
            continue
        
        chat = [
            # {"role": "system", "content":  "You are a helpful assistant that help people extract high-quality datasets.",}
        ]   
        content = user_prompt + PROMPT_SEG01
        _good_blog_cnt = 0
        for i, row in df.iterrows():  
            if check_quality and  row["blog_quality"].lower() != "high":
                continue
            _good_blog_cnt += 1
            content += "Post : " + str(_good_blog_cnt) + "\n" + row["cleaned_text"] + "\n\n"
        if _good_blog_cnt < record_threshold:
            skip_user_cnt += 1
            continue
        if len(content) > USER_TEXT_MAX_LEN:
            content = content[:USER_TEXT_MAX_LEN]
            
        content += PROMPT_SEG02 + user_profile + "\n\n"
            
        chat.append({"role": "user", "content": content})
        # text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
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
  

def job_userwise_judger_user_profiles_fulltypes(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, outdir_flag=""):
    
    dir_to_raw_blogs = f"{HOME_DIR}/blogger/by_users_quality_tagging"
    
    dir_to_be_processed = [ 
        f"{HOME_DIR}/blogger/users_persona_v2(use_quality_high)" ,
        f"{HOME_DIR}/blogger/users_profile_v2(use_quality_high)" , 
        f"{HOME_DIR}/blogger/users_stories(use_quality_high)",        
    ]
    prompts_to_be_used = [
        PROMPT_judger_user_persona,
        PROMPT_judger_user_profile,
        PROMPT_judger_user_stories,
    ]
    
    for indir, user_prompt in  zip(dir_to_be_processed, prompts_to_be_used): 
        outdir = os.path.join(indir, f"_judger{outdir_flag}")
                
        gen_content_and_output2file_bydir_profile_judger(
            model, tokenizer, user_prompt, indir, outdir, dir_to_raw_blogs, max_K, base, split, 
            check_quality=True, record_threshold=3, cutoff_string=cutoff_string)
        
def run_all_jobs(base, split, max_K=-1):
    model_path = f"Qwen/Qwen2.5-72B-Instruct"
    outdir_flag = "_Qwen2.5-72B"
    
    input_length = 40960
    if "Qwen2.5" in model_path:
        input_length = 32767
        USER_TEXT_MAX_LEN = int(input_length*0.8)
    model, tokenizer = load_model_vllm(model_path, MAX_MODEL_LENGTH=input_length)

    job_rowwise_judger_socialqa_fulltypes(model, tokenizer, max_K=max_K, base=base, split=split, outdir_flag=outdir_flag) 
    job_rowwise_judger_scenario_fulltypes(model, tokenizer, max_K=max_K, base=base, split=split, outdir_flag=outdir_flag)
    job_userwise_judger_user_profiles_fulltypes(model, tokenizer, max_K=max_K, base=base, split=split, outdir_flag=outdir_flag)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=-1)
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()
    base, split = args.base, args.split
 
    run_all_jobs(base, split, max_K=-1)