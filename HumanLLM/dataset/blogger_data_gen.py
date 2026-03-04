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
    
def gen_content_and_output2file_byfile(
        model, tokenizer, user_prompt, infile, outfile, max_K=-1, 
        new_column="result", check_quality=True, data_column="cleaned_text",
        override=False, cutoff_string=None, blog_length=-1, simplify_old_columns=False, model_name=None, thread_num=-1):
    ## cutoff_string is used to parse final results from deep thinking mdoels, which is usually </think>\n\n
    total_cost = 0
    BATCH_SIZE = 16 
    total_input_tokens, total_output_tokens = 0, 0
        
    if os.path.exists(outfile) and not override:
        print(f"file {outfile} already exists, skip")
        return total_cost
    
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)
    
    try:
        df = pd.read_csv(infile, header=0)   
    except Exception as e:
        print(f"file {infile} is not a valid csv file, skip")
        return total_cost
    
    ## check if df is empty 
    if df is None or len(df) == 0:
        print(f"file {infile} is empty, skip")
        return total_cost
    
    ## iterate the dataframe  
    batch_indices, batch_prompts = [], []
    ## do not use tqdm here to enumerate rows
    # for i, row in tqdm(df.iterrows(), total=len(df), desc="Blog Cleaning"):
    for i, row in df.iterrows():
        if check_quality and  row["blog_quality"].lower() != "high": 
            df.loc[i, new_column] = "SKIP"  
            continue
        if blog_length > 0 and len(row[data_column]) < blog_length:
            df.loc[i, new_column] = "SKIP"  
            continue
        content = user_prompt + row[data_column]
        if len(content) > USER_TEXT_MAX_LEN:
            content = content[:USER_TEXT_MAX_LEN]
        chat = [
            # {"role": "system", "content":  "You are a helpful assistant that help people extract high-quality datasets.",}
        ]
        chat.append({"role": "user", "content": content})
        if tokenizer is None:
            text = chat 
        else:
            text = tokenizer.apply_chat_template(chat, **APPLY_CHAT_TEMPLATE_PARAM)
        batch_indices.append(i)
        batch_prompts.append(text)          
        
        if len(batch_indices) >= BATCH_SIZE: 
            
            batch_response = generate(model, tokenizer, batch_prompts) 
            
            for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
                if cutoff_string is not None and cutoff_string in _response:
                    ## only keep the content after the cutoff_string; if there are multiple cutoff_string, only keep the last one
                    _response = _response.split(cutoff_string)[-1]
                df.loc[_index, new_column] = _response    
                 
            batch_indices, batch_prompts = [], []
        
        if max_K>0 and i > max_K: 
            break 
    if len(batch_indices) > 0:
        batch_response = generate(model, tokenizer, batch_prompts) 
        for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
            if cutoff_string is not None and cutoff_string in _response:
                ## only keep the content after the cutoff_string; if there are multiple cutoff_string, only keep the last one
                _response = _response.split(cutoff_string)[-1]
            df.loc[_index, new_column] = _response    
                
        batch_indices, batch_prompts = [], []    
    
    if simplify_old_columns:
        ## only keep columns: id,cleaned_text,blog_quality,new_column
        df = df[["id", new_column, "cleaned_text", "blog_quality"]] 
        ## remove rows that the new_column is empty or length is less than 5
    df = df[df[new_column].apply(lambda x: len(str(x)) > 0 )]
    ## if number of rows > 0:
    if len(df) > 0:
        safe_save2file(outfile, df)
    
    total_cost = 0
    return total_cost

def gen_content_and_output2file_bydir(model, tokenizer, user_prompt, indir, outdir, 
                                      max_K=-1, base=-1, split=0, check_quality=False, 
                                      record_threshold=5, cutoff_string=None):
        
    BATCH_SIZE = 16
    
    outdir_prompt = outdir + "_prompt"
    os.makedirs(outdir_prompt, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    
    ## for each file under indir, process data 
    cnt = 0
    skip_user_cnt = 0
    filename_list = os.listdir(indir) 
    filename_list = random.sample(filename_list, len(filename_list))
    batch_userids, batch_prompts = [], []
    for filename in tqdm(filename_list, desc="Processing users", total=len(filename_list)):
        if not filename.endswith(".csv"):
            continue        
        
        userid = filename.split(".")[0]   
        if base > 0 and split >= 0 and userid.isdigit():
            if int(userid)%base != split:
                continue
            
        if os.path.exists(os.path.join(outdir, filename.replace(".csv", ".txt"))) and os.path.exists(os.path.join(outdir_prompt, filename.replace(".csv", ".txt"))):
            print(f"file {filename} already exists, skip")
            continue
        
        try:
            df = pd.read_csv(os.path.join(indir, filename), header=0)  
        except Exception as e:
            print(f"file {os.path.join(indir, filename)} is not a valid csv file, skip")
            continue
        
        
        chat = [
            # {"role": "system", "content":  "You are a helpful assistant that help people extract high-quality datasets.",}
        ]   
        if isinstance(user_prompt, str):
            content = user_prompt
        else:
            ## randomly select one prompt
            content = random.choice(user_prompt)
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
            
        chat.append({"role": "user", "content": content})
        text = tokenizer.apply_chat_template(chat, **APPLY_CHAT_TEMPLATE_PARAM)
        
        batch_userids.append(userid)
        batch_prompts.append(text)
        
        if len(batch_userids) >= BATCH_SIZE: 
            batch_response = generate(model, tokenizer, batch_prompts) 
            for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):  
                if cutoff_string is not None and cutoff_string in _response:
                    _response = _response.split(cutoff_string)[1]    
                with open(os.path.join(outdir_prompt, str(_userid) + ".txt"), "w") as f:
                    f.write(_prompt)                  
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
            with open(os.path.join(outdir_prompt, str(_userid) + ".txt"), "w") as f:
                f.write(_prompt)                  
            with open(os.path.join(outdir, str(_userid) + ".txt"), "w") as f:
                f.write(_response)
    print(f"processed {cnt} users, skipped {skip_user_cnt} users")

def job_userwise_blog_quality_tagging(model, tokenizer, base, split, max_K=-1):
    user_prompt = """You are a helpful assistant that help me determine the data quality of a blog. The background is that I want to collect blogs which contains human behaviors or human thoughts, so that I can further study social science based on collected data in next step.    However, as you may know, blogs from the Internet contains various of content, and many of them are irrevelant to my goal so I need to filter them out. \n\n\nTypically, a blog's quality is high if it records detailed events of a human, reflecting human life, or mentions human's social behaviors, or revealing a human's thoughts regarding to something. \n\nA blog's quality is medium if it only very beriefly mentioned some content that are related to human behaviors or thoughts, based on what we cannot infer the whole picture of a story.\n\nA blog's quality is low if it is nothing to do with human behaviors or thoughts, such as ads, job post, company description, plots from novels, random words typed by users, and many more types.\n\nSo your task is to tag the quality a blog. Each time I provide you with a user's blog post, please tag it with "high", "medium", or "low".\nBelow is the user's blog post, please only output one tag, and do not include any other words such as explanations.\n\n"""    
    indir = f"{HOME_DIR}/blogger/by_users"
    outdir = f"{HOME_DIR}/blogger/by_users_quality_tagging"

    os.makedirs(outdir, exist_ok=True)
    ## get all filenames in indir 
    allfiles = os.listdir(indir)
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
        gen_content_and_output2file_byfile(
            model, tokenizer, 
            user_prompt, infile, outfile, 
            max_K=-1, 
            new_column="blog_quality", 
            check_quality=False, data_column="cleaned_text",
            override=False, cutoff_string=None)
        cnt += 1
        if max_K > 0 and cnt > max_K:
            break

def job_rowwise_single_long_story(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, blog_length=-1, outdir_flag=""):
    user_prompt = PROMPT_job_rowwise_single_long_story
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging"
    outdir = f"{HOME_DIR}/blogger/users_sing_long_story(use_quality_high)" 
 
    os.makedirs(outdir, exist_ok=True)
    ## get all filenames in indir 
    allfiles = os.listdir(indir)
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
        gen_content_and_output2file_byfile(
            model, tokenizer, 
            user_prompt, infile, outfile, 
            max_K=-1, 
            new_column="single_long_story", 
            check_quality=True, data_column="cleaned_text",
            override=False, cutoff_string=cutoff_string)
        cnt += 1
        if max_K > 0 and cnt > max_K:
            break

def job_rowwise_single_long_story_focusonbehavior(
    model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None,
    blog_length=-1, outdir_flag=""
    ):
    user_prompt = PROMPT_job_rowwise_single_long_story_focusonbehavior
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging" 
    outdir = f"{HOME_DIR}/blogger/users_single_long_story_focusonbehavior(user_quality_high)" 
 
    os.makedirs(outdir, exist_ok=True)
    ## get all filenames in indir 
    allfiles = os.listdir(indir)
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
        gen_content_and_output2file_byfile(
            model, tokenizer, 
            user_prompt, infile, outfile, 
            max_K=-1, 
            new_column="single_long_story_focusonbebaviors", 
            check_quality=True, data_column="cleaned_text",
            override=False, cutoff_string=cutoff_string,
            blog_length=blog_length)
        cnt += 1
        if max_K > 0 and cnt > max_K:
            break

def job_rowwise_user_thoughs_sing_blog(
    model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None,
    blog_length=-1, outdir_flag=""
    ):
    user_prompt = PROMPT_job_rowwise_user_thoughts_from_single_blog
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging" 
    outdir = f"{HOME_DIR}/blogger/users_thoughts_single_blog(user_quality_high)" 
 
    os.makedirs(outdir, exist_ok=True)
    ## get all filenames in indir 
    allfiles = os.listdir(indir)
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
        gen_content_and_output2file_byfile(
            model, tokenizer, 
            user_prompt, infile, outfile, 
            max_K=-1, 
            new_column="thoughts_singleblog", 
            check_quality=True, data_column="cleaned_text",
            override=False, cutoff_string=cutoff_string,
            blog_length=blog_length)
        cnt += 1
        if max_K > 0 and cnt > max_K:
            break

def job_rowwise_long_scenario_from_single_blog(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, outdir_flag=""):
    user_prompt = PROMPT_job_rowwise_long_scenario_from_single_blog
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging" 
    outdir = f"{HOME_DIR}/blogger/users_long_scenario_from_single_blog(use_quality_high)" 
 
    os.makedirs(outdir, exist_ok=True)
    ## get all filenames in indir 
    allfiles = os.listdir(indir) 
       
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
        gen_content_and_output2file_byfile(
            model, tokenizer, 
            user_prompt, infile, outfile, 
            max_K=-1, 
            new_column="longscenario_singleblog", 
            check_quality=True, data_column="cleaned_text",
            override=False, cutoff_string=cutoff_string)
        cnt += 1
        if max_K > 0 and cnt > max_K:
            break

def job_userwise_vividstories(model, tokenizer, max_K=-1, base=-1, split=-1, cutoff_string=None, outdir_flag=""):
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging" 
    out_dir = f"{HOME_DIR}/blogger/users_stories(use_quality_high)" 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)  
    user_prompt = PROMPT_job_userwise_vividstories
       
    # gen_content_and_output2file_bydir(model, tokenizer, user_prompt, indir, out_dir, max_K, base, split, check_quality=True)
    gen_content_and_output2file_bydir(
        model, tokenizer, user_prompt, indir, out_dir, max_K, base, split, 
        check_quality=True, record_threshold=3, cutoff_string=cutoff_string)
    
def job_userwise_user_persona_v2(model, tokenizer, max_K=-1, base=-1, split=-1, cutoff_string=None, outdir_flag=""):
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging" 
    out_dir = f"{HOME_DIR}/blogger/users_persona_v2(use_quality_high)" 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)  
    user_prompt = PROMPT_user_persona_v2
       
    gen_content_and_output2file_bydir(
        model, tokenizer, user_prompt, indir, out_dir, max_K, base, split, 
        check_quality=True, record_threshold=3, cutoff_string=cutoff_string)
    
def job_userwise_user_profile_v2(model, tokenizer, max_K=-1, base=-1, split=-1, cutoff_string=None, outdir_flag=""):
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging" 
    out_dir = f"{HOME_DIR}/blogger/users_profile_v2(use_quality_high)" 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)  
    user_prompt = PROMPT_user_profile_v2
       
    gen_content_and_output2file_bydir(
        model, tokenizer, user_prompt, indir, out_dir, max_K, base, split, 
        check_quality=True, record_threshold=3, cutoff_string=cutoff_string)
    
def job_rowwise_general_single_prompt(
    model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, outdir_flag="",
    user_prompt="", indir="",  outdir="", new_column="result", data_column="cleaned_text"
    ):  
 
    os.makedirs(outdir, exist_ok=True)
    ## get all filenames in indir 
    allfiles = os.listdir(indir) 
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
        gen_content_and_output2file_byfile(
            model, tokenizer, 
            user_prompt, infile, outfile, 
            max_K=-1, 
            new_column="result", 
            check_quality=True, data_column="cleaned_text",
            override=False, cutoff_string=cutoff_string)
        cnt += 1
        if max_K > 0 and cnt > max_K:
            break

def job_rowwise_clean_blog(model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None):
    user_prompt = PROMPT_job_rowwise_clean_blog
    indir =  f"{HOME_DIR}/blogger/by_users"
    outdir = f"{HOME_DIR}/blogger/by_users"
    os.makedirs(outdir, exist_ok=True)
    allfiles = os.listdir(indir)
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
        gen_content_and_output2file_byfile(
            model, tokenizer, 
            user_prompt, infile, outfile, 
            max_K=-1, 
            new_column="cleaned_text", 
            check_quality=False, data_column="text",
            override=True, cutoff_string=cutoff_string)
        cnt += 1
        if max_K > 0 and cnt > max_K:
            break

def llm_gen_content_simpliest_file2file_csvbase(
        model, tokenizer, user_prompt, infile, outfile, max_K=-1,  
        override=False, cutoff_string=None, model_name="",
        simplify_old_columns=True, new_column="result", data_column="cleaned_text",
        check_quality=True, blog_length=-1, thread_num=-1
    ):
    ## cutoff_string is used to parse final results from deep thinking mdoels, which is usually </think> 
    ## add one column to the csv file, which is named as "judger_result" by default
    ## return a scalar value, which is the cost of api (0 if use local model)
    ## if check_quality is True, there should be a column named "blog_quality" in the csv file 
    
    
    BATCH_SIZE = 16 
    input_token_price, output_token_price = 0, 0 ## local model does not have token price
    total_input_tokens, total_output_tokens = 0, 0
    total_cost = 0

    if os.path.exists(outfile) and not override:
        print(f"file {outfile} already exists, skip")
        return total_cost
    
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)
    
    try: 
        df = pd.read_csv(infile, header=0, dtype={data_column: str})
    except Exception as e:
        print(f"file {infile} is not a valid csv file, skip")
        return total_cost
    
    ## check if df is empty 
    if df is None or len(df) == 0:
        print(f"file {infile} is empty, skip")
        return total_cost
    
    ## iterate the dataframe  
    batch_indices, batch_prompts = [], []
    ## do not use tqdm here to enumerate rows
    # for i, row in tqdm(df.iterrows(), total=len(df), desc="Blog Cleaning"):
    for i, row in df.iterrows():
        try: 
            if check_quality and  row["blog_quality"].lower() != "high": 
                df.loc[i, new_column] = "SKIP"  
                continue
            if blog_length > 0 and len(row[data_column]) < blog_length:
                df.loc[i, new_column] = "SKIP"  
                continue
            content = user_prompt + row[data_column] 
        except Exception as e:
            print(f"error: {e}")
            print(f"row data: {row[data_column]}") 
            raise e
        if len(content) > USER_TEXT_MAX_LEN:
            content = content[:USER_TEXT_MAX_LEN] 
        chat = [
            # {"role": "system", "content":  "You are a helpful assistant that help people extract high-quality datasets.",}
        ]
        chat.append({"role": "user", "content": content})
        
        if tokenizer is None:
            text = chat 
        else:
            text = tokenizer.apply_chat_template(chat, **APPLY_CHAT_TEMPLATE_PARAM)
             
        batch_indices.append(i)
        batch_prompts.append(text)          
        
        if len(batch_indices) >= BATCH_SIZE:  
            batch_response = generate(model, tokenizer, batch_prompts) 
            for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
                if cutoff_string is not None and cutoff_string in _response:
                    _response = _response.split(cutoff_string)[-1]
                df.loc[_index, new_column] = _response    
                 
            batch_indices, batch_prompts = [], []
        
        if max_K>0 and i > max_K: 
            break 
    if len(batch_indices) > 0: 
        batch_response = generate(model, tokenizer, batch_prompts) 
        for _index, _prompt, _response in zip(batch_indices, batch_prompts, batch_response):  
            if cutoff_string is not None and cutoff_string in _response:
                _response = _response.split(cutoff_string)[-1]
            df.loc[_index, new_column] = _response    
                
        batch_indices, batch_prompts = [], []

    if simplify_old_columns:
        ## only keep columns: id,cleaned_text,blog_quality,new_column
        df = df[["id", new_column, "cleaned_text", "blog_quality"]]   
    df = df[df[new_column].apply(lambda x: len(str(x)) > 0 and x != "SKIP" and x != "NULL")]
     
    if len(df) > 0:
        safe_save2file(outfile, df)
    
    if total_input_tokens > 0:
        total_cost = total_input_tokens/1000000 * input_token_price + total_output_tokens/1000000 * output_token_price
    else:
        total_cost = 0
    return total_cost

def job_unified_simplest_llm_generation_csvbase(
        model, tokenizer, max_K=-1, base=-1, split=0, cutoff_string=None, new_column="result", data_column="cleaned_text",
        user_prompt="", indir="",  outdir="", 
        model_name="",  override=False, simplify_old_columns=True, check_quality=True
    ):
    ## the prompt format:  one parmeter in the prompt. Simply append the data_column's content to the end of the user_prompt
   
    start_time = time.time()
    os.makedirs(outdir, exist_ok=True) 
    total_cost = 0
    
    ## get all filenames in indir 
    allfiles = os.listdir(indir)  
    ## randomly shuffle the files 
    original_file_cnt = len(allfiles)
    allfiles = random.sample(allfiles, original_file_cnt)
    if max_K > 0:
        allfiles = allfiles[:max_K] 
        if original_file_cnt > max_K:
            max_K = -1
    cnt = 0
    try:
        for filename in tqdm(allfiles, desc=f"Processing {os.path.basename(indir)}", total=len(allfiles)):
            if not filename.endswith(".csv"):
                continue
            userid = filename.split(".")[0]
            infile = os.path.join(indir, filename)
            outfile = os.path.join(outdir, filename)
            if base > 0 and split >= 0 and userid.isdigit():
                if int(userid)%base != split:
                    continue 
            current_cost = llm_gen_content_simpliest_file2file_csvbase(
                model, tokenizer, 
                user_prompt, infile, outfile, 
                max_K=max_K, check_quality=check_quality, blog_length=50,
                override=override, cutoff_string=cutoff_string, model_name=model_name,
                simplify_old_columns=simplify_old_columns, new_column=new_column, data_column=data_column)
            total_cost += current_cost
            cnt += 1
            tqdm.write(f"Processed {cnt} files, current cost: ${current_cost:.2f}, total cost: ${total_cost:.2f}")
    except Exception as e:
        print(f"Error processing {filename}: \n{e}") 
        
    outfile_log = os.path.join(outdir, "_job_log.txt") 
    end_time = time.time()
    with open(outfile_log, "a") as f:
        f.write("" + "="*20 + "\n")
        f.write(f"Job finished time (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n") 
        f.write(f"Model: {model_name}\n")
        f.write(f"Processed {cnt} files, total cost: ${total_cost:.2f}\n")
        f.write(f"Time taken: {(end_time - start_time)/60/60:.2f} hours\n")
        f.write("" + "="*20 + "\n\n")  

def job_unified_userwise_llm_generation_csv2txt(
    model, tokenizer, max_K=-1, base=-1, split=-1, cutoff_string=None, 
    check_quality=True, record_threshold=3,
    user_prompt="", indir="",  outdir="", model_name="",  override=False, batch_size=16, thread_num=8,
    data_column="cleaned_text"
    ):
    start_time = time.time()
    os.makedirs(outdir, exist_ok=True)  
    total_cost, input_tokens_cnt, output_tokens_cnt = 0, 0, 0
    input_token_price, output_token_price = 0, 0
    
    ## for each file under indir, process data 
    cnt = 0
    skip_user_cnt = 0
    filename_list = os.listdir(indir) 
    filename_list = random.sample(filename_list, len(filename_list))
    if max_K > 0:
        filename_list = filename_list[:max_K] 
        
    batch_userids, batch_prompts = [], []
    for filename in tqdm(filename_list, desc=f"Processing {os.path.basename(indir)}", total=len(filename_list)):
        if not filename.endswith(".csv"):
            continue        
        
        userid = filename.split(".")[0]   
        if base > 0 and split >= 0 and userid.isdigit():
            if int(userid)%base != split:
                continue
            
        if os.path.exists(os.path.join(outdir, filename.replace(".csv", ".txt"))):
            print(f"file {filename} already exists, skip")
            continue       
        
        df = safe_loadcsv(os.path.join(indir, filename))
        if df is None or len(df) == 0:
            print(f"file {os.path.join(indir, filename)} is empty, skip")
            continue
         
        chat = [
            # {"role": "system", "content":  "You are a helpful assistant that help people extract high-quality datasets.",}
        ]   
        if isinstance(user_prompt, str):
            content = user_prompt
        else:
            ## randomly select one prompt
            content = random.choice(user_prompt)
        
        content += "\n"    
        _good_blog_cnt = 0
        for i, row in df.iterrows():  
            if check_quality and  row["blog_quality"].lower() != "high":
                continue
            _good_blog_cnt += 1
            content += "Post : " + str(_good_blog_cnt) + "\n" + row[data_column] + "\n\n"
        if _good_blog_cnt < record_threshold:
            skip_user_cnt += 1
            continue
        if len(content) > USER_TEXT_MAX_LEN:
            content = content[:USER_TEXT_MAX_LEN]
         
        chat.append({"role": "user", "content": content})
        if tokenizer is None:
            text = chat 
        else:
            text = tokenizer.apply_chat_template(chat, **APPLY_CHAT_TEMPLATE_PARAM)
             
        batch_userids.append(userid)
        batch_prompts.append(text)
        
        if len(batch_userids) >= batch_size: 
            batch_response = generate(model, tokenizer, batch_prompts)   
            for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):  
                if cutoff_string is not None and cutoff_string in _response:
                    _response = _response.split(cutoff_string)[1]              
                with open(os.path.join(outdir, str(_userid) + ".txt"), "w") as f:
                    f.write(_response)
            batch_userids, batch_prompts = [], []
            current_cost = input_tokens_cnt/1000000 * input_token_price + output_tokens_cnt/1000000 * output_token_price
            total_cost +=  current_cost
            tqdm.write(f"Processed {cnt} files, current cost: ${current_cost:.2f}, total cost: ${total_cost:.2f}")
        cnt += 1  
        
        
    if len(batch_userids) > 0:
        batch_response = generate(model, tokenizer, batch_prompts)
        for _userid, _prompt, _response in zip(batch_userids, batch_prompts, batch_response):  
            if cutoff_string is not None and cutoff_string in _response:
                _response = _response.split(cutoff_string)[1]              
            with open(os.path.join(outdir, str(_userid) + ".txt"), "w") as f:
                f.write(_response)
        batch_userids, batch_prompts = [], []    
        current_cost = input_tokens_cnt/1000000 * input_token_price + output_tokens_cnt/1000000 * output_token_price
        total_cost +=  current_cost
        tqdm.write(f"Processed {cnt} files, current cost: ${current_cost:.2f}, total cost: ${total_cost:.2f}")
        
    outfile_log = os.path.join(outdir, "_job_log.txt") 
    end_time = time.time()
    with open(outfile_log, "a") as f:
        f.write("" + "="*20 + "\n")
        f.write(f"Job finished time (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n") 
        f.write(f"Model: {model_name}\n")
        f.write(f"processed {cnt} users, skipped {skip_user_cnt} users") 
        f.write(f"Total cost: ${total_cost:.2f}\n") 
        f.write(f"Time taken: {(end_time - start_time)/60/60:.2f} hours\n")
        f.write("" + "="*20 + "\n\n")  

def run_all_jobs(base, split, max_K=-1):
    model_path = "meta-llama/Llama-3.3-70B-Instruct" 
    input_length = 40960
    model, tokenizer = load_model_vllm(model_path, MAX_MODEL_LENGTH=input_length)
    cutoff_string = None

    job_rowwise_clean_blog(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string)
    job_userwise_blog_quality_tagging(model, tokenizer, base, split, max_K=max_K)
    job_rowwise_single_long_story(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, blog_length=1800)
    job_rowwise_single_long_story_focusonbehavior(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, blog_length=1800)
    job_rowwise_user_thoughs_sing_blog(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, blog_length=-1)

    job_rowwise_long_scenario_from_single_blog(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string) 

    job_userwise_vividstories(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string)
    job_userwise_user_persona_v2(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string)
    job_userwise_user_profile_v2(model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string)

    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging"  
    outdir = f"{HOME_DIR}/blogger/by_users_blog_summary(use_quality_high)" 
    job_rowwise_general_single_prompt(
        model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, 
       user_prompt=PROMPT_blog_summary, indir=indir,  outdir=outdir, new_column="result", data_column="cleaned_text")  

    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging"  
    outdir = f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_v2(use_quality_high)" 
    job_rowwise_general_single_prompt(
        model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, 
       user_prompt=PROMPT_job_rowwise_scenario_question_answer_from_single_blog_v2, indir=indir,  outdir=outdir, new_column="result", data_column="cleaned_text")  
    
    outdir = f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_emphasizeaction_v2(use_quality_high)" 
    job_rowwise_general_single_prompt(
        model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, 
       user_prompt=PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizeaction_v2, indir=indir,  outdir=outdir, new_column="result", data_column="cleaned_text")  
    
    outdir = f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_emphasizethoughts_v2(use_quality_high)" 
    job_rowwise_general_single_prompt(
        model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, 
       user_prompt=PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizethoughts_v2, indir=indir,  outdir=outdir, new_column="result", data_column="cleaned_text")  
    
    outdir = f"{HOME_DIR}/blogger/users_scenario_question_answer_from_single_blog_emphasizereason_v2(use_quality_high)" 
    job_rowwise_general_single_prompt(
        model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, 
       user_prompt=PROMPT_job_rowwise_scenario_question_answer_from_single_blog_emphasizereason_v2, indir=indir,  outdir=outdir, new_column="result", data_column="cleaned_text")


    user_prompt = PROMPT_job_rowwise_clean_blog_step2
    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging"  
    outdir = f"{HOME_DIR}/blogger/by_users_quality_tagging_v2_rewritten"
    job_unified_simplest_llm_generation_csvbase(
        model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, 
        new_column="rewritten_text", data_column="cleaned_text", check_quality=False,
        user_prompt=user_prompt, indir=indir,  outdir=outdir, override=False, simplify_old_columns=False
    )

    indir =  f"{HOME_DIR}/blogger/by_users_quality_tagging_v2_rewritten"  
    outdir = f"{HOME_DIR}/blogger/writing_style_rewritten_text(use_quality_high)"  
    user_prompt = PROMPT_writing_style
    job_unified_userwise_llm_generation_csv2txt(
        model, tokenizer, max_K=max_K, base=base, split=split, cutoff_string=cutoff_string, 
        check_quality=True, record_threshold=3,
        user_prompt=user_prompt, indir=indir,  outdir=outdir, override=False, 
        batch_size=16, thread_num=16, data_column="rewritten_text"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=-1)
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()
    base, split = args.base, args.split
 
    run_all_jobs(base, split, max_K=-1)