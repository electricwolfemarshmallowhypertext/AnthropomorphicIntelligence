import json
import pandas as pd
import random
import math
import os
import sys
import re
import csv
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timezone
from datasets import load_dataset
from utils import extract_tag_content, extract_tag_content_with_bounds, extract_field_from_json_re, extract_after_keyword, extract_from_description, filter_best_description, is_null, clean_data, extract_field_from_json_re_2
from generate_sft_data_prompt import *
from amazon_data_gen import find_last_qualified_review_idx
csv.field_size_limit(sys.maxsize)
random.seed(2025)
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-7B-Instruct", use_fast=True)

HOME_DIR = os.path.expanduser("~/HumanLLM_data")

def word_count(text):
    if len(text.split()) <= 5:
        return str(math.ceil(len(text.split()) / 10.0) * 10)
    return str(round(len(text.split()) / 10.0) * 10)   #str(math.ceil(len(text.split()) / 50) * 50)

def filter_too_long_case(examples, max_token_length=8192):
    output = []
    min_length=1000000
    max_length=-1
    avg_length=0
    for prompt in tqdm(examples, desc="Filtering too long cases", total=len(examples)):
        tokens = tokenizer.apply_chat_template(prompt['messages'], tokenize=True, add_generation_prompt=False)
        if len(tokens) > max_token_length:
            continue
        output.append(prompt)
        min_length = min(min_length, len(tokens))
        max_length = max(max_length, len(tokens))
        avg_length += len(tokens)
    return output, min_length, max_length, avg_length/len(output)

def parse_json_string_stories(json_string):
    pattern = r"""
        (?:"summary"|'summary')\s*:\s*    
        (["'])(.*?)\1\s*,\s*              
        (?:"content"|'content')\s*:\s*    
        (["'])(.*?)\3                     
    """
    matches = re.findall(pattern, json_string, re.VERBOSE | re.DOTALL)
    parsed_data = [{"summary": re.sub(r'\\(.)', r'\1', summary).strip(),
                "content": re.sub(r'\\(.)', r'\1', content).strip()}
               for _, summary, _, content in matches]
    return parsed_data

def filter_print(task, cur_data, max_data_num_per_task):
    print(f"{task} data nums: {len(cur_data)}")
    cut_data = random.sample(cur_data, min(len(cur_data), int(1.2*max_data_num_per_task)))
    print(f"{task} data nums after sample: {len(cut_data)}")
    cur_data, min_l, max_l, avg_l = filter_too_long_case(cut_data)
    print(f"{task} data nums after filter too long case: {len(cur_data)}, min_l: {min_l}, max_l: {max_l}, avg_l: {avg_l}")
    # random.shuffle(cur_data)
    cur_data = cur_data[:max_data_num_per_task]
    return cur_data[:int(len(cur_data)*0.9)], cur_data[int(len(cur_data)*0.9):]

def parse_xml_string_as_kv_dict(lines):  
    lines = re.sub(r"\n", "", lines) 
    
    ## only use the content wrapped within <data> and </data>
    lines = re.sub(r"<data>(.*)</data>", r"\1", lines)    
    
    ## parse the xml data
    score_dict = {}
    errors = False
    try:
        score_obj = re.findall(r"<(\w+)>\s*(\d+)\s*</\1>", lines)
        for key, value in score_obj:
            score_dict[key] = int(value)
    except Exception as e:
        errors = True
        print(f"Error in parsing xml data: {e}")
    
    return score_dict, errors

def check_quality(judger_data, overall_threshold=9, other_threshold=8, metrics=None):
    if judger_data.endswith(".txt"):
        if os.path.exists(judger_data):
            with open(judger_data, 'r', encoding='utf-8') as f:
                data = f.read()
        else:
            return False
    else:
        data = judger_data
    score_dict, errors = parse_xml_string_as_kv_dict(data)
    if errors:
        return False
    if metrics is not None:
        for metric, thred in metrics.items():
            if metric not in score_dict:
                return False
            if score_dict[metric] < thred:
                return False
    else:
        if 'overall' not in score_dict or score_dict['overall'] < overall_threshold:
            return False
        for key in score_dict:
            if key != 'overall' and score_dict[key] < other_threshold:
                return False
    
    return True

def truncate_text_by_words(text, max_words=300, marker='[truncated]'):
    words = text.split(" ")
    if len(words) > max_words:
        truncated = ' '.join(words[:max_words])
        return truncated + ' ' + marker
    else:
        return text

def count_tokens(text):
    return len(tokenizer.encode(text))

def split_text_randomly(text, max_total_tokens=8192, min_front_ratio=0.3, max_front_ratio=0.7):
    tokens = tokenizer.encode(text)
    if len(tokens) < 16:  
        return None, None

    tokens = tokens[:max_total_tokens]
    truncated_text = tokenizer.decode(tokens)
    split_text = truncated_text.split(" ")

    min_idx = int(len(split_text) * min_front_ratio)
    max_idx = int(len(split_text) * max_front_ratio)
    if max_idx <= min_idx or min_idx < 1 or max_idx >= len(split_text):
        return None, None  
    split_idx = random.randint(min_idx, max_idx)
    front_text = ' '.join(split_text[:split_idx])
    back_text = ' '.join(split_text[split_idx:])

    return front_text.strip(), back_text.strip()

def generate_for_reddit(dataset_name="reddit"):
    in_dir = f'{HOME_DIR}/reddit'
    judger_flag = "_judger_Qwen2.5-72B"
    output_train_file = f"{HOME_DIR}/sft_dataset/train_reddit.json"
    output_test_file = f"{HOME_DIR}/sft_dataset/test_reddit.json"
    all_train_data = []
    all_test_data = []

    ############# persona -> profile
    max_data_num_per_task = 30000
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'

    persona_files = sorted(os.listdir(in_dir_persona))

    cur_data = []
    skip_cnt = 0
    for filename in tqdm(persona_files, desc="Processing persona -> profile", total=len(persona_files)):
        if not filename.endswith(".txt") or not os.path.exists(os.path.join(in_dir_profile, filename)):
            continue
        
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if not check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}) or not check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            skip_cnt += 1
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
            persona_data = f.read()
        with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
            profile_data = f.read()

        if is_null(persona_data) or is_null(profile_data):
            continue

        messages = [
            {"role": "user", "content": persona2profile_prompt.replace("{persona}", persona_data).replace("{num}", word_count(profile_data))},
            {"role": "assistant", "content": profile_data},
        ]

        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.persona2profile", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.persona2profile", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(persona_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# half2half_stories
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    skip_cnt = 0
    for filename in tqdm(stories_files, desc="Processing half2half_stories", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            skip_cnt += 1
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        if is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index:]

        messages = [
            {"role": "user", "content": half2half_stories_prompt.replace("{past_life_stories}", json.dumps(front_part)).replace("{num}", str(len(back_part)))},
            {"role": "assistant", "content": json.dumps(back_part)},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half2half_stories", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half2half_stories", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(stories_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# half_persona2half_stories
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    skip_cnt = 0
    for filename in tqdm(stories_files, desc="Processing half_persona2half_stories", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            skip_cnt += 1
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        profiles = []
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if os.path.exists(os.path.join(in_dir_persona, filename)) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
            with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            if not is_null(persona_data):
                profiles.append(persona_data)
        if os.path.exists(os.path.join(in_dir_profile, filename)) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
                profile_data = f.read()
            if not is_null(profile_data):
                profiles.append(profile_data)
        if len(profiles) == 0 or is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index:]
        cur_persona = random.choice(profiles)

        messages = [
            {"role": "user", "content": half_persona2half_stories_prompt.replace("{persona}", cur_persona).replace("{past_life_stories}", json.dumps(front_part)).replace("{num}", str(len(back_part)))},
            {"role": "assistant", "content": json.dumps(back_part)},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_persona2half_stories", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_persona2half_stories", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(stories_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# half_theme2target_story
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    for filename in tqdm(stories_files, desc="Processing half_theme2target_story", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        if is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index]

        messages = [
            {"role": "user", "content": half_theme2target_story_prompt.replace("{past_life_stories}", json.dumps(front_part)).replace("{target_summary}", back_part['summary']).replace("{num}", word_count(back_part['content']))},
            {"role": "assistant", "content": back_part['content']},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_theme2target_story", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_theme2target_story", cur_data, max_data_num_per_task)
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# half_theme_persona2target_story
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    for filename in tqdm(stories_files, desc="Processing half_theme_persona2target_story", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue

        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        profiles = []
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if os.path.exists(os.path.join(in_dir_persona, filename)) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
            with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            if not is_null(persona_data):
                profiles.append(persona_data)
        if os.path.exists(os.path.join(in_dir_profile, filename)) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
                profile_data = f.read()
            if not is_null(profile_data):
                profiles.append(profile_data)
        
        if len(profiles) == 0 or is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index]
        cur_persona = random.choice(profiles)

        messages = [
            {"role": "user", "content": half_theme_persona2target_story_prompt.replace("{persona}", cur_persona).replace("{past_life_stories}", json.dumps(front_part)).replace("{target_summary}", back_part['summary']).replace("{num}", word_count(back_part['content']))},
            {"role": "assistant", "content": back_part['content']},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_theme_persona2target_story", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_theme_persona2target_story", cur_data, max_data_num_per_task)
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)



    ############# social QA 
    max_data_num_per_task = 30000
    in_dir_social_qas = [
        f'{in_dir}/scenario_question_answer_from_single_blog_emphasizereason_v3(medium_high)/{judger_flag}',
        f'{in_dir}/scenario_question_answer_from_single_blog_emphisize_actions(medium_high)/{judger_flag}',
        f'{in_dir}/scenario_question_answer_from_single_blog_emphisize_thoughts(medium_high)/{judger_flag}',
        f'{in_dir}/scenario_question_answer_from_single_blog(medium_high)/{judger_flag}',
    ]
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'
    
    
    for in_dir_social_qa in in_dir_social_qas:
        task_name = in_dir_social_qa.split("/")[-2]
        social_qa_files = sorted(os.listdir(in_dir_social_qa))
        cur_data = []
        skip_cnt = 0
        all_cnt = 0
        for filename in tqdm(social_qa_files, desc="Processing social_qa", total=len(social_qa_files)):
            if not filename.endswith(".json"):
                continue
            userid = filename.split(".")[0]

            social_qa_data = []
            with open(os.path.join(in_dir_social_qa, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    all_cnt += 1
                    line_data = json.loads(line)
                    if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                        continue
                    if task_name == 'scenario_question_answer_from_single_blog_emphasizereason_v3(medium_high)':
                        metrics = {"hallucination":8, "coverage":8, "fidelity":8, "novelty":7, "leakage": 7, "overall":8}
                    else:
                        metrics = None
                    if 'quality_result' not in line_data or not check_quality(line_data['quality_result'], metrics=metrics):
                        skip_cnt += 1
                        continue
                    social_qa_data.append(line_data)

            profiles = []
            judger_persona = os.path.join(in_dir_persona, judger_flag, filename.replace(".json", ".txt"))
            judger_profile = os.path.join(in_dir_profile, judger_flag, filename.replace(".json", ".txt"))
            if os.path.exists(os.path.join(in_dir_persona, filename.replace(".json", ".txt"))) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
                with open(os.path.join(in_dir_persona, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                    persona_data = f.read()
                if not is_null(persona_data):
                    profiles.append(persona_data)
            if os.path.exists(os.path.join(in_dir_profile, filename.replace(".json", ".txt"))) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
                with open(os.path.join(in_dir_profile, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                    profile_data = f.read()
                if not is_null(profile_data):
                    profiles.append(profile_data)
            
            if len(profiles) == 0 or len(social_qa_data) == 0:
                continue

            for social_qa in social_qa_data:
                scenario = extract_tag_content(social_qa['sqa_singleblog'], 'scenario')
                question = extract_tag_content(social_qa['sqa_singleblog'], 'question')
                answer = extract_tag_content(social_qa['sqa_singleblog'], 'answer')
                if is_null(scenario) or is_null(question) or is_null(answer):
                    continue
                cur_persona = random.choice(profiles)

                messages = [
                    {"role": "user", "content": socialQA_prompt.replace("{persona}", cur_persona).replace("{scenario}", scenario).replace("{question}", question).replace("{num}", word_count(answer))},
                    {"role": "assistant", "content": answer},
                ]
                cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.{task_name}", 
                        "messages": messages,
                    })
        train_data, test_data = filter_print(f"{dataset_name}.{task_name}", cur_data, max_data_num_per_task)
        print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

    ############# social scenario long_scenario_from_single_blog
    max_data_num_per_task = 13000
    in_dir_social_scenario = f'{in_dir}/users_long_scenario_from_single_blog(medium_high)/{judger_flag}'
    social_scenario_files = sorted(os.listdir(in_dir_social_scenario))
    cur_data = []
    skip_cnt = 0
    all_cnt = 0
    for filename in tqdm(social_scenario_files, desc="Processing long_scenario_from_single_blog", total=len(social_scenario_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        social_scena_data = []
        with open(os.path.join(in_dir_social_scenario, filename), 'r', encoding='utf-8') as f:
            for line in f:
                all_cnt += 1
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                if 'quality_result' not in line_data or not check_quality(line_data['quality_result']):
                    skip_cnt += 1
                    continue
                social_scena_data.append(line_data)

        if len(social_scena_data) == 0:
            continue

        for social_scena in social_scena_data:
            background = extract_tag_content(social_scena['sqa_singleblog'], 'summary')
            story = extract_tag_content(social_scena['sqa_singleblog'], 'scenario')
            all_characters = []
            charac_begin_idx = 0
            while True:
                character, charac_end_idx = extract_tag_content_with_bounds(social_scena['sqa_singleblog'], 'character', charac_begin_idx)
                if is_null(character) or character in all_characters:
                    break
                all_characters.append(character)
                charac_begin_idx = charac_end_idx
            characters = "; ".join(all_characters)
            if is_null(background) or is_null(characters) or is_null(story):
                continue

            messages = [
                {"role": "user", "content": socialScenario_long_scenario_from_single_blog_prompt.replace("{summary}", background).replace("{character}", characters).replace("{num}", word_count(story))},
                {"role": "assistant", "content": story},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.long_scenario_from_single_blog", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.long_scenario_from_single_blog", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# social scenario single_long_story
    max_data_num_per_task = 13000
    in_dir_social_scenarios = [
        f'{in_dir}/users_sing_long_story(medium_high)/{judger_flag}',
        f'{in_dir}/users_single_long_story_focusonbehavior(medium_high)/{judger_flag}',
        f'{in_dir}/users_thoughts_single_blog(medium_high)/{judger_flag}'
    ]
    for in_dir_social_scenario in in_dir_social_scenarios:
        task_name = in_dir_social_scenario.split("/")[-2]
        
        social_scenario_files = sorted(os.listdir(in_dir_social_scenario))
        cur_data = []
        skip_cnt = 0
        all_cnt = 0
        for filename in tqdm(social_scenario_files, desc="Processing social_scenario", total=len(social_scenario_files)):
            if not filename.endswith(".json"):
                continue

            userid = filename.split(".")[0]

            social_scena_data = []
            with open(os.path.join(in_dir_social_scenario, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    all_cnt += 1
                    line_data = json.loads(line)
                    if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                        continue
                    if 'quality_result' not in line_data or not check_quality(line_data['quality_result']):
                        skip_cnt += 1
                        continue
                    social_scena_data.append(line_data)

            if len(social_scena_data) == 0:
                continue

            for social_scena in social_scena_data:
                background = extract_tag_content(social_scena['sqa_singleblog'], 'background')
                character = extract_tag_content(social_scena['sqa_singleblog'], 'characters')
                if task_name == 'users_sing_long_story(medium_high)':
                    story = extract_tag_content(social_scena['sqa_singleblog'], 'plots')
                elif task_name == 'users_single_long_story_focusonbehavior(medium_high)':
                    story = extract_tag_content(social_scena['sqa_singleblog'], 'story')
                elif task_name == 'users_thoughts_single_blog(medium_high)':
                    story = extract_tag_content(social_scena['sqa_singleblog'], 'thoughts')
                else:
                    story = None
                if is_null(background) or is_null(character) or is_null(story):
                    continue

                messages = [
                    {"role": "user", "content": socialScenario_prompt.replace("{background}", background).replace("{character}", character).replace("{num}", word_count(story))},
                    {"role": "assistant", "content": story},
                ]
                cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.{task_name}", 
                        "messages": messages,
                    })
        train_data, test_data = filter_print(f"{dataset_name}.{task_name}", cur_data, max_data_num_per_task)
        print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

    ############# writing imitation type1: history posts + topic
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        if len(writing_imitation_data) < 3:
            continue

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue
            history_data = writing_imitation_data[:i] + writing_imitation_data[i+1:]
            sampled_num = random.randint(1, min(5, len(history_data)))
            sampled_history = random.sample(history_data, sampled_num)
            history_description = ""
            for idx_his, history in enumerate(sampled_history):
                history_description += f"\nPost {idx_his+1}:\n{truncate_text_by_words(history['rewritten_text_v2'])}"
            
            try:
                user_prompt = writing_imitation_prompt_type1.replace("{past_posts}", history_description).replace("{scenario}", writing_imitation['sqa_singleblog'].strip())
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text_v2']))},
                {"role": "assistant", "content": writing_imitation['rewritten_text_v2']},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type1", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type1", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# writing imitation type2: history posts and completion
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        if len(writing_imitation_data) < 3:
            continue

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue
            history_data = writing_imitation_data[:i] + writing_imitation_data[i+1:]
            sampled_num = random.randint(1, min(5, len(history_data)))
            sampled_history = random.sample(history_data, sampled_num)
            history_description = ""
            for idx_his, history in enumerate(sampled_history):
                history_description += f"\nPost {idx_his+1}:\n{truncate_text_by_words(history['rewritten_text_v2'])}"
            
            front, back = split_text_randomly(writing_imitation['rewritten_text_v2'])
            if not front or not back or len(back) < 5 or len(front) < 5:
                continue 

            try:
                user_prompt = writing_imitation_prompt_type2.replace("{past_posts}", history_description).replace("{front}", front)
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                {"role": "assistant", "content": back},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type2", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type2", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# writing imitation type3: writing style + topic
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    in_dir_writing_style = f'{in_dir}/writing_style'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_style_file = os.path.join(in_dir_writing_style, f"{userid}.txt")
        if not os.path.exists(writing_style_file):
            continue
        with open(writing_style_file, 'r', encoding='utf-8') as f:
            writing_style = f.read().strip()
        if is_null(writing_style):
            continue

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue
            try:
                user_prompt = writing_imitation_prompt_type3.replace("{style}", writing_style).replace("{scenario}", writing_imitation['sqa_singleblog'].strip())
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text_v2']))},
                {"role": "assistant", "content": writing_imitation['rewritten_text_v2']},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type3", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type3", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# writing imitation type4: writing style + completion
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    in_dir_writing_style = f'{in_dir}/writing_style'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_style_file = os.path.join(in_dir_writing_style, f"{userid}.txt")
        if not os.path.exists(writing_style_file):
            continue
        with open(writing_style_file, 'r', encoding='utf-8') as f:
            writing_style = f.read().strip()
        if is_null(writing_style):
            continue

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue
            front, back = split_text_randomly(writing_imitation['rewritten_text_v2'])
            if not front or not back or len(back) < 5 or len(front) < 5:
                continue 
            
            try:
                user_prompt = writing_imitation_prompt_type4.replace("{style}", writing_style).replace("{front}", front)
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                {"role": "assistant", "content": back},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type4", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type4", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# personalized comment
    max_data_num_per_task = 100000
    in_dir_comments = f'{in_dir}/by_users_quality_tagging_v2_rewritten'
    in_dir_tagging = f'{in_dir}/by_users_quality_tagging_v2'
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'

    comments_files = sorted(os.listdir(in_dir_comments))
    cur_data = []
    all_cnt = 0
    for filename in tqdm(comments_files, desc="Processing comments", total=len(comments_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        in_data = []
        with open(os.path.join(in_dir_comments, filename), "r", encoding="utf-8") as f:
            for line in f:
                in_data.append(json.loads(line))
        in_data_tag = []
        with open(os.path.join(in_dir_tagging, filename), "r", encoding="utf-8") as f:
            for line in f:
                in_data_tag.append(json.loads(line))

        comments_data = []
        for i in range(len(in_data)):
            if in_data[i]['type'] == 't1':
                all_cnt += 1
                if is_null(in_data[i]['rewritten_text_v2']) or is_null(in_data[i]['rewrite_blog']):
                    continue
                assert in_data[i]['id'] == in_data_tag[i]['id'], "ID mismatch between data and tag files"
                unsafe = extract_field_from_json_re_2(in_data_tag[i]['blog_quality_v2'], 'unsafe content')
                social = extract_field_from_json_re_2(in_data_tag[i]['blog_quality_v2'], 'social event')
                if unsafe is None or social is None:
                    continue
                if unsafe.lower() == "no" and social.lower() == "yes":
                    comments_data.append(in_data[i])

        profiles = []
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename.replace(".json", ".txt"))
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename.replace(".json", ".txt"))
        if os.path.exists(os.path.join(in_dir_persona, filename.replace(".json", ".txt"))) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
            with open(os.path.join(in_dir_persona, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            if not is_null(persona_data):
                profiles.append(persona_data)
        if os.path.exists(os.path.join(in_dir_profile, filename.replace(".json", ".txt"))) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            with open(os.path.join(in_dir_profile, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                profile_data = f.read()
            if not is_null(profile_data):
                profiles.append(profile_data)

        if len(profiles) == 0 or len(comments_data) < 3:
            continue

        for i, comments in enumerate(comments_data):
            target_count = count_tokens(comments['rewritten_text_v2'])
            if target_count > 1000 or target_count < 20:
                continue
            cur_persona = random.choice(profiles)
            other_comments = comments_data[:i] + comments_data[i+1:]
            sampled_num = random.randint(1, min(5, len(other_comments)))
            sampled_comments = random.sample(other_comments, sampled_num)
            comments_description = ""
            for idx_his, history in enumerate(sampled_comments):
                comments_description += f"\nComment {idx_his+1}:\n{truncate_text_by_words(history['rewritten_text_v2'])}"

            messages = [
                {"role": "user", "content": personalized_comment_prompt.replace("{persona}", cur_persona).replace("{post}", comments['rewrite_blog']).replace("{past_comments}", comments_description).replace("{num}", word_count(comments['rewritten_text_v2']))},
                {"role": "assistant", "content": comments['rewritten_text_v2']},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.personalized_comment", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.personalized_comment", cur_data, max_data_num_per_task)
    print(f"all raw data: {all_cnt}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    print(f"reddit total train data nums: {len(all_train_data)}, total test data nums: {len(all_test_data)}")
    with open(output_train_file, 'a') as f:
        for r in all_train_data:
            f.write(json.dumps(r)+'\n')

    with open(output_test_file, 'a') as f:
        for r in all_test_data:
            f.write(json.dumps(r)+'\n')

def generate_for_twitter(dataset_name="twitter"):
    in_dir = f'{HOME_DIR}/twitter'
    judger_flag = "_judger_Qwen2.5-72B"
    output_train_file = f"{HOME_DIR}/sft_dataset/train_twitter.json"
    output_test_file = f"{HOME_DIR}/sft_dataset/test_twitter.json"
    all_train_data = []
    all_test_data = []

    ############# persona -> profile
    max_data_num_per_task = 30000
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'
    
    persona_files = sorted(os.listdir(in_dir_persona))

    cur_data = []
    skip_cnt = 0
    for filename in tqdm(persona_files, desc="Processing persona -> profile", total=len(persona_files)):
        if not filename.endswith(".txt") or not os.path.exists(os.path.join(in_dir_profile, filename)):
            continue

        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if not check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}) or not check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            skip_cnt += 1
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
            persona_data = f.read()
        with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
            profile_data = f.read()

        if is_null(persona_data) or is_null(profile_data):
            continue

        messages = [
            {"role": "user", "content": persona2profile_prompt.replace("{persona}", persona_data).replace("{num}", word_count(profile_data))},
            {"role": "assistant", "content": profile_data},
        ]

        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.persona2profile", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.persona2profile", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(persona_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# half2half_stories
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    skip_cnt = 0
    for filename in tqdm(stories_files, desc="Processing half2half_stories", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            skip_cnt += 1
            continue
        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        if is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index:]

        messages = [
            {"role": "user", "content": half2half_stories_prompt.replace("{past_life_stories}", json.dumps(front_part)).replace("{num}", str(len(back_part)))},
            {"role": "assistant", "content": json.dumps(back_part)},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half2half_stories", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half2half_stories", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(stories_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# half_persona2half_stories
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    skip_cnt = 0
    for filename in tqdm(stories_files, desc="Processing half_persona2half_stories", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue

        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            skip_cnt += 1
            continue
        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        profiles = []
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if os.path.exists(os.path.join(in_dir_persona, filename)) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
            with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            if not is_null(persona_data):
                profiles.append(persona_data)
        if os.path.exists(os.path.join(in_dir_profile, filename)) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
                profile_data = f.read()
            if not is_null(profile_data):
                profiles.append(profile_data)
        if len(profiles) == 0 or is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index:]
        cur_persona = random.choice(profiles)

        messages = [
            {"role": "user", "content": half_persona2half_stories_prompt.replace("{persona}", cur_persona).replace("{past_life_stories}", json.dumps(front_part)).replace("{num}", str(len(back_part)))},
            {"role": "assistant", "content": json.dumps(back_part)},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_persona2half_stories", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_persona2half_stories", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(stories_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# half_theme2target_story
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    for filename in tqdm(stories_files, desc="Processing half_theme2target_story", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        if is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index]

        messages = [
            {"role": "user", "content": half_theme2target_story_prompt.replace("{past_life_stories}", json.dumps(front_part)).replace("{target_summary}", back_part['summary']).replace("{num}", word_count(back_part['content']))},
            {"role": "assistant", "content": back_part['content']},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_theme2target_story", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_theme2target_story", cur_data, max_data_num_per_task)
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# half_theme_persona2target_story
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(medium_high)'
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    for filename in tqdm(stories_files, desc="Processing half_theme_persona2target_story", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        profiles = []
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if os.path.exists(os.path.join(in_dir_persona, filename)) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
            with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            if not is_null(persona_data):
                profiles.append(persona_data)
        if os.path.exists(os.path.join(in_dir_profile, filename)) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
                profile_data = f.read()
            if not is_null(profile_data):
                profiles.append(profile_data)
        
        if len(profiles) == 0 or is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index]
        cur_persona = random.choice(profiles)

        messages = [
            {"role": "user", "content": half_theme_persona2target_story_prompt.replace("{persona}", cur_persona).replace("{past_life_stories}", json.dumps(front_part)).replace("{target_summary}", back_part['summary']).replace("{num}", word_count(back_part['content']))},
            {"role": "assistant", "content": back_part['content']},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_theme_persona2target_story", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_theme_persona2target_story", cur_data, max_data_num_per_task)
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# social QA 
    max_data_num_per_task = 30000
    in_dir_social_qas = [
        f'{in_dir}/scenario_question_answer_from_single_blog_emphasizereason_v3(medium_high)/{judger_flag}',
        f'{in_dir}/scenario_question_answer_from_single_blog_emphisize_actions(medium_high)/{judger_flag}',
        f'{in_dir}/scenario_question_answer_from_single_blog_emphisize_thoughts(medium_high)/{judger_flag}',
        f'{in_dir}/scenario_question_answer_from_single_blog(medium_high)/{judger_flag}'
    ]
    in_dir_persona = f'{in_dir}/users_persona_v2(medium_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(medium_high)'

    for in_dir_social_qa in in_dir_social_qas:
        task_name = in_dir_social_qa.split("/")[-2]
        social_qa_files = sorted(os.listdir(in_dir_social_qa))
        cur_data = []
        skip_cnt = 0
        all_cnt = 0
        for filename in tqdm(social_qa_files, desc="Processing social_qa", total=len(social_qa_files)):
            if not filename.endswith(".json"):
                continue
            userid = filename.split(".")[0]

            social_qa_data = []
            with open(os.path.join(in_dir_social_qa, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    all_cnt += 1
                    line_data = json.loads(line)
                    if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                        continue
                    if task_name == "scenario_question_answer_from_single_blog_emphasizereason_v3(medium_high)":
                        metrics = {"hallucination":9, "coverage":8, "fidelity":8, "novelty":7, "leakage": 7, "overall":7}
                    else:
                        metrics = None
                    if 'quality_result' not in line_data or not check_quality(line_data['quality_result'], metrics=metrics):
                        skip_cnt += 1
                        continue
                    social_qa_data.append(line_data)

            profiles = []
            judger_persona = os.path.join(in_dir_persona, judger_flag, filename.replace(".json", ".txt"))
            judger_profile = os.path.join(in_dir_profile, judger_flag, filename.replace(".json", ".txt"))
            if os.path.exists(os.path.join(in_dir_persona, filename.replace(".json", ".txt"))) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
                with open(os.path.join(in_dir_persona, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                    persona_data = f.read()
                if not is_null(persona_data):
                    profiles.append(persona_data)
            if os.path.exists(os.path.join(in_dir_profile, filename.replace(".json", ".txt"))) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
                with open(os.path.join(in_dir_profile, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                    profile_data = f.read()
                if not is_null(profile_data):
                    profiles.append(profile_data)
            
            if len(profiles) == 0 or len(social_qa_data) == 0:
                continue

            for social_qa in social_qa_data:
                scenario = extract_tag_content(social_qa['sqa_singleblog'], 'scenario')
                question = extract_tag_content(social_qa['sqa_singleblog'], 'question')
                answer = extract_tag_content(social_qa['sqa_singleblog'], 'answer')
                if is_null(scenario) or is_null(question) or is_null(answer):
                    continue
                cur_persona = random.choice(profiles)

                messages = [
                    {"role": "user", "content": socialQA_prompt.replace("{persona}", cur_persona).replace("{scenario}", scenario).replace("{question}", question).replace("{num}", word_count(answer))},
                    {"role": "assistant", "content": answer},
                ]
                cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.{task_name}", 
                        "messages": messages,
                    })
        train_data, test_data = filter_print(f"{dataset_name}.{task_name}", cur_data, max_data_num_per_task)
        print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)


    ############# social scenario long_scenario_from_single_blog
    max_data_num_per_task = 13000
    in_dir_social_scenario = f'{in_dir}/users_long_scenario_from_single_blog(medium_high)/{judger_flag}'
    social_scenario_files = sorted(os.listdir(in_dir_social_scenario))
    cur_data = []
    skip_cnt = 0
    all_cnt = 0
    for filename in tqdm(social_scenario_files, desc="Processing long_scenario_from_single_blog", total=len(social_scenario_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        social_scena_data = []
        with open(os.path.join(in_dir_social_scenario, filename), 'r', encoding='utf-8') as f:
            for line in f:
                all_cnt += 1
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                if 'quality_result' not in line_data or not check_quality(line_data['quality_result']):
                    skip_cnt += 1
                    continue
                social_scena_data.append(line_data)

        if len(social_scena_data) == 0:
            continue

        for social_scena in social_scena_data:
            background = extract_tag_content(social_scena['sqa_singleblog'], 'summary')
            story = extract_tag_content(social_scena['sqa_singleblog'], 'scenario')
            all_characters = []
            charac_begin_idx = 0
            while True:
                character, charac_end_idx = extract_tag_content_with_bounds(social_scena['sqa_singleblog'], 'character', charac_begin_idx)
                if is_null(character) or character in all_characters:
                    break
                all_characters.append(character)
                charac_begin_idx = charac_end_idx
            characters = "; ".join(all_characters)
            if is_null(background) or is_null(characters) or is_null(story):
                continue

            messages = [
                {"role": "user", "content": socialScenario_long_scenario_from_single_blog_prompt.replace("{summary}", background).replace("{character}", characters).replace("{num}", word_count(story))},
                {"role": "assistant", "content": story},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.long_scenario_from_single_blog", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.long_scenario_from_single_blog", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# social scenario single_long_story
    max_data_num_per_task = 13000
    in_dir_social_scenarios = [
        f'{in_dir}/users_sing_long_story(medium_high)/{judger_flag}',
        f'{in_dir}/users_single_long_story_focusonbehavior(medium_high)/{judger_flag}',
        f'{in_dir}/users_thoughts_single_blog(medium_high)/{judger_flag}'
    ]
    for in_dir_social_scenario in in_dir_social_scenarios:
        task_name = in_dir_social_scenario.split("/")[-2]
        
        social_scenario_files = sorted(os.listdir(in_dir_social_scenario))
        cur_data = []
        skip_cnt = 0
        all_cnt = 0
        for filename in tqdm(social_scenario_files, desc="Processing social_scenario", total=len(social_scenario_files)):
            if not filename.endswith(".json"):
                continue

            userid = filename.split(".")[0]

            social_scena_data = []
            with open(os.path.join(in_dir_social_scenario, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    all_cnt += 1
                    line_data = json.loads(line)
                    if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                        continue
                    if 'quality_result' not in line_data or not check_quality(line_data['quality_result']):
                        skip_cnt += 1
                        continue
                    social_scena_data.append(line_data)

            if len(social_scena_data) == 0:
                continue

            for social_scena in social_scena_data:
                background = extract_tag_content(social_scena['sqa_singleblog'], 'background')
                character = extract_tag_content(social_scena['sqa_singleblog'], 'characters')
                if task_name == 'users_sing_long_story(medium_high)':
                    story = extract_tag_content(social_scena['sqa_singleblog'], 'plots')
                elif task_name == 'users_single_long_story_focusonbehavior(medium_high)':
                    story = extract_tag_content(social_scena['sqa_singleblog'], 'story')
                elif task_name == 'users_thoughts_single_blog(medium_high)':
                    story = extract_tag_content(social_scena['sqa_singleblog'], 'thoughts')
                else:
                    story = None
                if is_null(background) or is_null(character) or is_null(story):
                    continue

                messages = [
                    {"role": "user", "content": socialScenario_prompt.replace("{background}", background).replace("{character}", character).replace("{num}", word_count(story))},
                    {"role": "assistant", "content": story},
                ]
                cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.{task_name}", 
                        "messages": messages,
                    })
        train_data, test_data = filter_print(f"{dataset_name}.{task_name}", cur_data, max_data_num_per_task)
        print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)


    ############# writing imitation type1: history posts + topic
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        if len(writing_imitation_data) < 3:
            continue

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue

            history_data = writing_imitation_data[:i] + writing_imitation_data[i+1:]
            sampled_num = random.randint(1, min(5, len(history_data)))
            sampled_history = random.sample(history_data, sampled_num)
            history_description = ""
            for idx_his, history in enumerate(sampled_history):
                history_description += f"\nPost {idx_his+1}:\n{truncate_text_by_words(history['rewritten_text_v2'])}"
            
            try:
                user_prompt = writing_imitation_prompt_type1.replace("{past_posts}", history_description).replace("{scenario}", writing_imitation['sqa_singleblog'].strip())
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text_v2']))},
                {"role": "assistant", "content": writing_imitation['rewritten_text_v2']},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type1", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type1", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# writing imitation type2: history posts and completion
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        if len(writing_imitation_data) < 3:
            continue

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue
            history_data = writing_imitation_data[:i] + writing_imitation_data[i+1:]
            sampled_num = random.randint(1, min(5, len(history_data)))
            sampled_history = random.sample(history_data, sampled_num)
            history_description = ""
            for idx_his, history in enumerate(sampled_history):
                history_description += f"\nPost {idx_his+1}:\n{truncate_text_by_words(history['rewritten_text_v2'])}"
            
            front, back = split_text_randomly(writing_imitation['rewritten_text_v2'])
            if not front or not back or len(back) < 5 or len(front) < 5:
                continue 

            try:
                user_prompt = writing_imitation_prompt_type2.replace("{past_posts}", history_description).replace("{front}", front)
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                {"role": "assistant", "content": back},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type2", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type2", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# writing imitation type3: writing style + topic
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    in_dir_writing_style = f'{in_dir}/writing_style'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_style_file = os.path.join(in_dir_writing_style, f"{userid}.txt")
        if not os.path.exists(writing_style_file):
            continue
        with open(writing_style_file, 'r', encoding='utf-8') as f:
            writing_style = f.read().strip()
        if is_null(writing_style):
            continue

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue
            try:
                user_prompt = writing_imitation_prompt_type3.replace("{style}", writing_style).replace("{scenario}", writing_imitation['sqa_singleblog'].strip())
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text_v2']))},
                {"role": "assistant", "content": writing_imitation['rewritten_text_v2']},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type3", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type3", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# writing imitation type4: writing style + completion
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/post_summary_v2'
    in_dir_writing_style = f'{in_dir}/writing_style'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".json"):
            continue

        userid = filename.split(".")[0]

        writing_style_file = os.path.join(in_dir_writing_style, f"{userid}.txt")
        if not os.path.exists(writing_style_file):
            continue
        with open(writing_style_file, 'r', encoding='utf-8') as f:
            writing_style = f.read().strip()
        if is_null(writing_style):
            continue

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line)
                if 'sqa_singleblog' not in line_data or is_null(line_data['sqa_singleblog']):
                    continue
                writing_imitation_data.append(line_data)

        for i, writing_imitation in enumerate(writing_imitation_data):
            target_count = count_tokens(writing_imitation['rewritten_text_v2'])
            if target_count > 1000 or target_count < 30:
                continue
            front, back = split_text_randomly(writing_imitation['rewritten_text_v2'])
            if not front or not back or len(back) < 5 or len(front) < 5:
                continue 
            
            try:
                user_prompt = writing_imitation_prompt_type4.replace("{style}", writing_style).replace("{front}", front)
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                {"role": "assistant", "content": back},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type4", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type4", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    print(f"twitter total train data nums: {len(all_train_data)}, total test data nums: {len(all_test_data)}")
    with open(output_train_file, 'a') as f:
        for r in all_train_data:
            f.write(json.dumps(r)+'\n')

    with open(output_test_file, 'a') as f:
        for r in all_test_data:
            f.write(json.dumps(r)+'\n')

def generate_for_amazon(dataset_name="amazon"):
    amazon_categories = [
        'Arts_Crafts_and_Sewing', 'Automotive', 'Baby_Products', 'Beauty_and_Personal_Care', 'Books', 
        'CDs_and_Vinyl', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics', 
        'Grocery_and_Gourmet_Food', 'Health_and_Household', 'Home_and_Kitchen',
        'Industrial_and_Scientific', 'Sports_and_Outdoors', 'Video_Games',
    ]
    
    in_dir = f'{HOME_DIR}/amazon'
    output_train_file = f"{HOME_DIR}/sft_dataset/train_amazon.json"
    output_test_file = f"{HOME_DIR}/sft_dataset/test_amazon.json"
    all_train_data = []
    all_test_data = []

    for category in amazon_categories:
        in_dir_persona = f'{in_dir}/users_persona/{category}'
        in_dir_profile = f'{in_dir}/users_profile/{category}'
        in_file_meta = f'{in_dir}/raw_data/amazon_{category}_content.json'
        in_dir_clean = f'{in_dir}/clean_data/{category}'
        in_dir_summary = f'{in_dir}/review_summary_v2/{category}'
        in_dir_style = f'{in_dir}/writing_style/{category}'
        meta_data = json.load(open(in_file_meta, 'r'))
        all_item_set = set(item for item in meta_data.keys())

        ############# persona -> profile
        max_data_num_per_task = 2000
        persona_files = sorted(os.listdir(in_dir_persona))
        cur_data = []
        for filename in tqdm(persona_files, desc="Processing persona -> profile", total=len(persona_files)):
            if not filename.endswith(".txt") or not os.path.exists(os.path.join(in_dir_profile, filename)):
                continue

            userid = filename.split(".")[0]

            with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
                profile_data = f.read()

            if is_null(persona_data) or is_null(profile_data):
                continue

            messages = [
                {"role": "user", "content": persona2profile_prompt.replace("{persona}", persona_data).replace("{num}", word_count(profile_data))},
                {"role": "assistant", "content": profile_data},
            ]

            cur_data.append({
                            "uid": userid,
                            "source": f"{dataset_name}.{category}.persona2profile", 
                            "messages": messages,
                        })

        train_data, test_data = filter_print(f"{dataset_name}.{category}.persona2profile", cur_data, max_data_num_per_task)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

    
        ############# item selection
        max_data_num_per_task = 15000
        user_files = sorted(os.listdir(in_dir_clean))
        cur_data = []
        for filename in tqdm(user_files, desc="Processing item selection", total=len(user_files)):
            if not filename.endswith(".json"):
                continue

            userid = filename.split(".")[0]

            user_data = []
            with open(os.path.join(in_dir_clean, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    line_data = json.loads(line)
                    user_data.append(line_data)

            profiles = []
            if os.path.exists(os.path.join(in_dir_persona, filename.replace(".json", ".txt"))):
                with open(os.path.join(in_dir_persona, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                    persona_data = f.read()
                if not is_null(persona_data):
                    profiles.append(persona_data)
            if os.path.exists(os.path.join(in_dir_profile, filename.replace(".json", ".txt"))):
                with open(os.path.join(in_dir_profile, filename.replace(".json", ".txt")), 'r', encoding='utf-8') as f:
                    profile_data = f.read()
                if not is_null(profile_data):
                    profiles.append(profile_data)

            if len(profiles) == 0 or len(user_data) < 8:
                continue
            
            last_qualified_idx, all_qualified_ids = find_last_qualified_review_idx(user_data)
            if last_qualified_idx < 7: # in this case, we use the last item as target, and do not do imitation task
                target_idx = len(user_data) - 1
            else:
                target_idx = last_qualified_idx

            begin_idx = max(0, target_idx - 30) # We use the last 30 items as context

            history_description = ""
            for i_idx, item in enumerate(user_data[begin_idx:target_idx]):
                item_metadata = meta_data[item['item_id']]
                history_description += f"History {i_idx+1}: {item_metadata['title']}\n"

            neg_candidates = all_item_set - set([item['item_id'] for item in user_data])
            neg_candidates = random.sample(list(neg_candidates), min(len(neg_candidates), 19))
            sample_insert_idx = random.randint(0, len(neg_candidates))
            neg_candidates.insert(sample_insert_idx, user_data[target_idx]['item_id'])

            candidate_description = ""
            for i_idx, item_id in enumerate(neg_candidates):
                item_metadata = meta_data[item_id]
                candidate_description += f"Candidate {i_idx+1}: {item_metadata['title']}\n"

            template = random.choice(item_selection_scenario_template)
            final_scenario = template.format(domain=category, candidate_items=candidate_description)

            behavior = meta_data[user_data[target_idx]['item_id']]['title']

            user_prompt = item_selection_prompt

            rand_num = random.random()
            if rand_num < 0.5:
                user_prompt = user_prompt.replace("{persona}", "").replace("{past_items}", f"\nHere are the user's history items:\n{history_description}")
            elif rand_num < 0.65:
                user_prompt = user_prompt.replace("{persona}", f"\nHere's the user description:\n{random.choice(profiles)}\n").replace("{past_items}", "")
            else:
                user_prompt = user_prompt.replace("{persona}", f"\nHere's the user description:\n{random.choice(profiles)}\n").replace("{past_items}", f"\nHere are the user's history items:\n{history_description}")

            messages = [
                {"role": "user", "content": user_prompt.replace("{scenario}", final_scenario)},
                {"role": "assistant", "content": behavior},
            ]

            cur_data.append({
                "uid": userid,
                "source": f"{dataset_name}.{category}.item_selection", 
                "messages": messages,
            })
        train_data, test_data = filter_print(f"{dataset_name}.{category}.item_selection", cur_data, max_data_num_per_task)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

        ############# writing imitation type1: history posts + topic
        max_data_num_per_task = 1700
        user_files = sorted(os.listdir(in_dir_summary))
        cur_data = []
        for filename in tqdm(user_files, desc="Processing writing imitation", total=len(user_files)):
            if not filename.endswith(".json"):
                continue

            userid = filename.split(".")[0]

            writing_imitation_data = []
            with open(os.path.join(in_dir_summary, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    line_data = json.loads(line)
                    if 'summary' not in line_data or is_null(line_data['summary']):
                        continue
                    writing_imitation_data.append(line_data)

            if len(writing_imitation_data) < 3:
                continue

            for i, writing_imitation in enumerate(writing_imitation_data):
                target_count = count_tokens(writing_imitation['rewritten_text_v2'])
                if target_count > 1000 or target_count < 30:
                    continue
                target_item = writing_imitation['item_id']
                target_meta = meta_data[target_item]
                target_desc = f"{target_meta['title']}"

                items = writing_imitation_data[:i] + writing_imitation_data[i+1:]
                sampled_num = random.randint(1, min(len(items), 5))
                items = random.sample(items, sampled_num)

                history_description = ""
                for i_idx, item in enumerate(items):
                    item_metadata = meta_data[item['item_id']]
                    item_description = f"User bought {item_metadata['title']}.\nThe review is: {truncate_text_by_words(item['rewritten_text_v2'])}\n"
                    history_description += f"\nHistory {i_idx+1}: {item_description}"
                
                try:
                    user_prompt = review_imitation_prompt_type1.replace("{past_reviews}", history_description).replace("{scenario}", writing_imitation['summary'].strip()).replace("{item_name}", target_desc)
                except:
                    continue

                messages = [
                    {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text_v2']))},
                    {"role": "assistant", "content": writing_imitation['rewritten_text_v2']},
                ]

                cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.{category}.writing_imitation_type1", 
                    "messages": messages,
                })
        train_data, test_data = filter_print(f"{dataset_name}.{category}.writing_imitation_type1", cur_data, max_data_num_per_task)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

        ############# writing imitation type2: history posts and completion
        max_data_num_per_task = 1700
        user_files = sorted(os.listdir(in_dir_summary))
        cur_data = []
        for filename in tqdm(user_files, desc="Processing writing imitation", total=len(user_files)):
            if not filename.endswith(".json"):
                continue

            userid = filename.split(".")[0]

            writing_imitation_data = []
            with open(os.path.join(in_dir_summary, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    line_data = json.loads(line)
                    if 'summary' not in line_data or is_null(line_data['summary']):
                        continue
                    writing_imitation_data.append(line_data)

            if len(writing_imitation_data) < 3:
                continue

            for i, writing_imitation in enumerate(writing_imitation_data):
                target_count = count_tokens(writing_imitation['rewritten_text_v2'])
                if target_count > 1000 or target_count < 30:
                    continue
                target_item = writing_imitation['item_id']
                target_meta = meta_data[target_item]
                target_desc = f"{target_meta['title']}"

                items = writing_imitation_data[:i] + writing_imitation_data[i+1:]
                sampled_num = random.randint(1, min(len(items), 5))
                items = random.sample(items, sampled_num)

                history_description = ""
                for i_idx, item in enumerate(items):
                    item_metadata = meta_data[item['item_id']]
                    item_description = f"User bought {item_metadata['title']}.\nThe review is: {truncate_text_by_words(item['rewritten_text_v2'])}\n"
                    history_description += f"\nHistory {i_idx+1}: {item_description}"

                front, back = split_text_randomly(writing_imitation['rewritten_text_v2'])
                if not front or not back or len(back) < 5 or len(front) < 5:
                    continue 
                
                user_prompt = review_imitation_prompt_type2.replace("{past_reviews}", history_description).replace("{front}", front).replace("{item_name}", target_desc)

                messages = [
                    {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                    {"role": "assistant", "content": back},
                ]

                cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.{category}.writing_imitation_type2", 
                    "messages": messages,
                })
        train_data, test_data = filter_print(f"{dataset_name}.{category}.writing_imitation_type2", cur_data, max_data_num_per_task)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

        ############# writing imitation type3: writing style + topic
        max_data_num_per_task = 1700
        user_files = sorted(os.listdir(in_dir_summary))
        cur_data = []
        for filename in tqdm(user_files, desc="Processing writing imitation", total=len(user_files)):
            if not filename.endswith(".json"):
                continue

            userid = filename.split(".")[0]

            writing_style_file = os.path.join(in_dir_style, f"{userid}.txt")
            if not os.path.exists(writing_style_file):
                continue
            with open(writing_style_file, 'r', encoding='utf-8') as f:
                writing_style = f.read().strip()
            if is_null(writing_style):
                continue

            writing_imitation_data = []
            with open(os.path.join(in_dir_summary, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    line_data = json.loads(line)
                    if 'summary' not in line_data or is_null(line_data['summary']):
                        continue
                    writing_imitation_data.append(line_data)

            for i, writing_imitation in enumerate(writing_imitation_data):
                target_count = count_tokens(writing_imitation['rewritten_text_v2'])
                if target_count > 1000 or target_count < 30:
                    continue
                target_item = writing_imitation['item_id']
                target_meta = meta_data[target_item]
                target_desc = f"{target_meta['title']}"
                
                try:
                    user_prompt = review_imitation_prompt_type3.replace("{style}", writing_style).replace("{scenario}", writing_imitation['summary'].strip()).replace("{item_name}", target_desc)
                except:
                    continue

                messages = [
                    {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text_v2']))},
                    {"role": "assistant", "content": writing_imitation['rewritten_text_v2']},
                ]

                cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.{category}.writing_imitation_type3", 
                    "messages": messages,
                })
        train_data, test_data = filter_print(f"{dataset_name}.{category}.writing_imitation_type3", cur_data, max_data_num_per_task)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

        ############# writing imitation type4: writing style + completion
        max_data_num_per_task = 1700
        user_files = sorted(os.listdir(in_dir_summary))
        cur_data = []
        for filename in tqdm(user_files, desc="Processing writing imitation", total=len(user_files)):
            if not filename.endswith(".json"):
                continue

            userid = filename.split(".")[0]

            writing_style_file = os.path.join(in_dir_style, f"{userid}.txt")
            if not os.path.exists(writing_style_file):
                continue
            with open(writing_style_file, 'r', encoding='utf-8') as f:
                writing_style = f.read().strip()
            if is_null(writing_style):
                continue

            writing_imitation_data = []
            with open(os.path.join(in_dir_summary, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    line_data = json.loads(line)
                    if 'summary' not in line_data or is_null(line_data['summary']):
                        continue
                    writing_imitation_data.append(line_data)

            for i, writing_imitation in enumerate(writing_imitation_data):
                target_count = count_tokens(writing_imitation['rewritten_text_v2'])
                if target_count > 1000 or target_count < 30:
                    continue
                target_item = writing_imitation['item_id']
                target_meta = meta_data[target_item]
                target_desc = f"{target_meta['title']}"

                front, back = split_text_randomly(writing_imitation['rewritten_text_v2'])
                if not front or not back or len(back) < 5 or len(front) < 5:
                    continue

                user_prompt = review_imitation_prompt_type4.replace("{style}", writing_style).replace("{front}", front).replace("{item_name}", target_desc)

                messages = [
                    {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                    {"role": "assistant", "content": back},
                ]

                cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.{category}.writing_imitation_type4", 
                    "messages": messages,
                })
        train_data, test_data = filter_print(f"{dataset_name}.{category}.writing_imitation_type4", cur_data, max_data_num_per_task)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)
    
    print(f"amazon total train data nums: {len(all_train_data)}, total test data nums: {len(all_test_data)}")
    with open(output_train_file, 'a') as f:
        for r in all_train_data:
            f.write(json.dumps(r)+'\n')

    with open(output_test_file, 'a') as f:
        for r in all_test_data:
            f.write(json.dumps(r)+'\n')

def generate_for_blogauthorship(dataset_name="blogauthorship"):
    in_dir = f'{HOME_DIR}/blogger'
    judger_flag = "_judger_Qwen2.5-72B"
    output_train_file = f"{HOME_DIR}/sft_dataset/train_blogger.json"
    output_test_file = f"{HOME_DIR}/sft_dataset/test_blogger.json"
    all_train_data = []
    all_test_data = []

    ############# persona -> profile
    max_data_num_per_task = 30000
    in_dir_persona = f'{in_dir}/users_persona_v2(use_quality_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(use_quality_high)'
    
    persona_files = sorted(os.listdir(in_dir_persona))

    cur_data = []
    skip_cnt = 0
    for filename in tqdm(persona_files, desc="Processing persona -> profile", total=len(persona_files)):
        if not filename.endswith(".txt") or not os.path.exists(os.path.join(in_dir_profile, filename)):
            continue
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if not check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}) or not check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            skip_cnt += 1
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
            persona_data = f.read()
        with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
            profile_data = f.read()

        if is_null(persona_data) or is_null(profile_data):
            continue

        messages = [
            {"role": "user", "content": persona2profile_prompt.replace("{persona}", persona_data).replace("{num}", word_count(profile_data))},
            {"role": "assistant", "content": profile_data},
        ]

        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.persona2profile", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.persona2profile", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(persona_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# half2half_stories
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(use_quality_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    skip_cnt = 0
    for filename in tqdm(stories_files, desc="Processing half2half_stories", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            skip_cnt += 1
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        if is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index:]

        messages = [
            {"role": "user", "content": half2half_stories_prompt.replace("{past_life_stories}", json.dumps(front_part)).replace("{num}", str(len(back_part)))},
            {"role": "assistant", "content": json.dumps(back_part)},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half2half_stories", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half2half_stories", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(stories_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# half_persona2half_stories
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(use_quality_high)'
    in_dir_persona = f'{in_dir}/users_persona_v2(use_quality_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(use_quality_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    skip_cnt = 0
    for filename in tqdm(stories_files, desc="Processing half_persona2half_stories", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            skip_cnt += 1
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        profiles = []
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if os.path.exists(os.path.join(in_dir_persona, filename)) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
            with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            if not is_null(persona_data):
                profiles.append(persona_data)
        if os.path.exists(os.path.join(in_dir_profile, filename)) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
                profile_data = f.read()
            if not is_null(profile_data):
                profiles.append(profile_data)
        if len(profiles) == 0 or is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index:]
        cur_persona = random.choice(profiles)

        messages = [
            {"role": "user", "content": half_persona2half_stories_prompt.replace("{persona}", cur_persona).replace("{past_life_stories}", json.dumps(front_part)).replace("{num}", str(len(back_part)))},
            {"role": "assistant", "content": json.dumps(back_part)},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_persona2half_stories", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_persona2half_stories", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {len(stories_files)}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# half_theme2target_story
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(use_quality_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    for filename in tqdm(stories_files, desc="Processing half_theme2target_story", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        if is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index]

        messages = [
            {"role": "user", "content": half_theme2target_story_prompt.replace("{past_life_stories}", json.dumps(front_part)).replace("{target_summary}", back_part['summary']).replace("{num}", word_count(back_part['content']))},
            {"role": "assistant", "content": back_part['content']},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_theme2target_story", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_theme2target_story", cur_data, max_data_num_per_task)
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# half_theme_persona2target_story
    max_data_num_per_task = 10000
    in_dir_stories = f'{in_dir}/users_stories(use_quality_high)'
    in_dir_persona = f'{in_dir}/users_persona_v2(use_quality_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(use_quality_high)'
    stories_files = sorted(os.listdir(in_dir_stories))
    cur_data = []
    for filename in tqdm(stories_files, desc="Processing half_theme_persona2target_story", total=len(stories_files)):
        if not filename.endswith(".txt"):
            continue
        judger_stories = os.path.join(in_dir_stories, judger_flag, filename)
        if not check_quality(judger_stories, metrics={"hallucination":8, "coverage":8, "informativeness":8,"novelty":7, "overall":8}):
            continue

        userid = filename.split(".")[0]

        with open(os.path.join(in_dir_stories, filename), 'r', encoding='utf-8') as f:
            stories_data = f.read()

        profiles = []
        judger_persona = os.path.join(in_dir_persona, judger_flag, filename)
        judger_profile = os.path.join(in_dir_profile, judger_flag, filename)
        if os.path.exists(os.path.join(in_dir_persona, filename)) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
            with open(os.path.join(in_dir_persona, filename), 'r', encoding='utf-8') as f:
                persona_data = f.read()
            if not is_null(persona_data):
                profiles.append(persona_data)
        if os.path.exists(os.path.join(in_dir_profile, filename)) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
            with open(os.path.join(in_dir_profile, filename), 'r', encoding='utf-8') as f:
                profile_data = f.read()
            if not is_null(profile_data):
                profiles.append(profile_data)
        
        if len(profiles) == 0 or is_null(stories_data):
            continue

        parsed_stories = parse_json_string_stories(stories_data)
        if len(parsed_stories) < 3:
            continue
        split_index = random.randint(2, len(parsed_stories) - 1)
        front_part = parsed_stories[:split_index]
        back_part = parsed_stories[split_index]
        cur_persona = random.choice(profiles)

        messages = [
            {"role": "user", "content": half_theme_persona2target_story_prompt.replace("{persona}", cur_persona).replace("{past_life_stories}", json.dumps(front_part)).replace("{target_summary}", back_part['summary']).replace("{num}", word_count(back_part['content']))},
            {"role": "assistant", "content": back_part['content']},
        ]
        cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.half_theme_persona2target_story", 
                        "messages": messages,
                    })

    train_data, test_data = filter_print(f"{dataset_name}.half_theme_persona2target_story", cur_data, max_data_num_per_task)
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# social QA 
    max_data_num_per_task = 30000
    in_dir_social_qas = [
        f'{in_dir}/users_scenario_question_answer_from_single_blog_emphasizeaction_v2(use_quality_high)/{judger_flag}',
        f'{in_dir}/users_scenario_question_answer_from_single_blog_emphasizethoughts_v2(use_quality_high)/{judger_flag}',
        f'{in_dir}/users_scenario_question_answer_from_single_blog_v2(use_quality_high)/{judger_flag}',
        f'{in_dir}/users_scenario_question_answer_from_single_blog_emphasizereason_v2(use_quality_high)/{judger_flag}'
    ]
    in_dir_persona = f'{in_dir}/users_persona_v2(use_quality_high)'
    in_dir_profile = f'{in_dir}/users_profile_v2(use_quality_high)'
    
    for in_dir_social_qa in in_dir_social_qas:
        task_name = in_dir_social_qa.split("/")[-2]
        social_qa_files = sorted(os.listdir(in_dir_social_qa))
        cur_data = []
        skip_cnt = 0
        all_cnt = 0
        for filename in tqdm(social_qa_files, desc="Processing social_qa", total=len(social_qa_files)):
            if not filename.endswith(".csv"):
                continue
            userid = filename.split(".")[0]

            social_qa_data = []
            with open(os.path.join(in_dir_social_qa, filename), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_cnt += 1
                    if 'result' not in row or is_null(row['result']):
                        continue
                    if 'quality_result' not in row or not check_quality(row['quality_result']):
                        skip_cnt += 1
                        continue
                    social_qa_data.append(row)

            profiles = []
            judger_persona = os.path.join(in_dir_persona, judger_flag, filename.replace(".csv", ".txt"))
            judger_profile = os.path.join(in_dir_profile, judger_flag, filename.replace(".csv", ".txt"))
            if os.path.exists(os.path.join(in_dir_persona, filename.replace(".csv", ".txt"))) and check_quality(judger_persona, metrics={"hallucination":8, "coverage":7, "conciseness":8,"relevance":8, "overall":8}):
                with open(os.path.join(in_dir_persona, filename.replace(".csv", ".txt")), 'r', encoding='utf-8') as f:
                    persona_data = f.read()
                if not is_null(persona_data):
                    profiles.append(persona_data)
            if os.path.exists(os.path.join(in_dir_profile, filename.replace(".csv", ".txt"))) and check_quality(judger_profile, metrics={"hallucination":8, "coverage":8, "relevance":9, "fluency": 9, "conciseness":8, "informativeness":8, "novelty":8, "overall":8}):
                with open(os.path.join(in_dir_profile, filename.replace(".csv", ".txt")), 'r', encoding='utf-8') as f:
                    profile_data = f.read()
                if not is_null(profile_data):
                    profiles.append(profile_data)
            
            if len(profiles) == 0 or len(social_qa_data) == 0:
                continue

            for social_qa in social_qa_data:
                scenario = extract_tag_content(social_qa['result'], 'scenario')
                question = extract_tag_content(social_qa['result'], 'question')
                answer = extract_tag_content(social_qa['result'], 'answer')
                if is_null(scenario) or is_null(question) or is_null(answer):
                    continue
                cur_persona = random.choice(profiles)

                messages = [
                    {"role": "user", "content": socialQA_prompt.replace("{persona}", cur_persona).replace("{scenario}", scenario).replace("{question}", question).replace("{num}", word_count(answer))},
                    {"role": "assistant", "content": answer},
                ]
                cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.{task_name}", 
                        "messages": messages,
                    })
        train_data, test_data = filter_print(f"{dataset_name}.{task_name}", cur_data, max_data_num_per_task)
        print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)


    ############# social scenario long_scenario_from_single_blog
    max_data_num_per_task = 13000
    in_dir_social_scenario = f'{in_dir}/users_long_scenario_from_single_blog(use_quality_high)/{judger_flag}'
    social_scenario_files = sorted(os.listdir(in_dir_social_scenario))
    cur_data = []
    skip_cnt = 0
    all_cnt = 0
    for filename in tqdm(social_scenario_files, desc="Processing long_scenario_from_single_blog", total=len(social_scenario_files)):
        if not filename.endswith(".csv"):
            continue

        userid = filename.split(".")[0]

        social_scena_data = []
        with open(os.path.join(in_dir_social_scenario, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_cnt += 1
                if 'longscenario_singleblog' not in row or is_null(row['longscenario_singleblog']):
                    continue
                if 'quality_result' not in row or not check_quality(row['quality_result']):
                    skip_cnt += 1
                    continue
                social_scena_data.append(row)

        if len(social_scena_data) == 0:
            continue

        for social_scena in social_scena_data:
            background = extract_tag_content(social_scena['longscenario_singleblog'], 'summary')
            story = extract_tag_content(social_scena['longscenario_singleblog'], 'scenario')
            all_characters = []
            charac_begin_idx = 0
            while True:
                character, charac_end_idx = extract_tag_content_with_bounds(social_scena['longscenario_singleblog'], 'character', charac_begin_idx)
                if is_null(character) or character in all_characters:
                    break
                all_characters.append(character)
                charac_begin_idx = charac_end_idx
            characters = "; ".join(all_characters)
            if is_null(background) or is_null(characters) or is_null(story):
                continue

            messages = [
                {"role": "user", "content": socialScenario_long_scenario_from_single_blog_prompt.replace("{summary}", background).replace("{character}", characters).replace("{num}", word_count(story))},
                {"role": "assistant", "content": story},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.long_scenario_from_single_blog", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.long_scenario_from_single_blog", cur_data, max_data_num_per_task)
    print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)


    ############# social scenario single_long_story
    max_data_num_per_task = 13000
    in_dir_social_scenarios = [
        f'{in_dir}/users_sing_long_story(use_quality_high)/{judger_flag}',
        f'{in_dir}/users_single_long_story_focusonbehavior(user_quality_high)/{judger_flag}',
        f'{in_dir}/users_thoughts_single_blog(user_quality_high)/{judger_flag}'
    ]
    key_map = {
        'users_sing_long_story(use_quality_high)': 'single_long_story',
        'users_single_long_story_focusonbehavior(user_quality_high)': 'single_long_story_focusonbebaviors',
        'users_thoughts_single_blog(user_quality_high)': 'thoughts_singleblog'
    }
    for in_dir_social_scenario in in_dir_social_scenarios:
        task_name = in_dir_social_scenario.split("/")[-2]
        
        social_scenario_files = sorted(os.listdir(in_dir_social_scenario))
        cur_data = []
        skip_cnt = 0
        all_cnt = 0
        for filename in tqdm(social_scenario_files, desc="Processing social_scenario", total=len(social_scenario_files)):
            if not filename.endswith(".csv"):
                continue

            userid = filename.split(".")[0]

            social_scena_data = []
            with open(os.path.join(in_dir_social_scenario, filename), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_cnt += 1
                    if key_map[task_name] not in row or is_null(row[key_map[task_name]]):
                        continue
                    if 'quality_result' not in row or not check_quality(row['quality_result']):
                        skip_cnt += 1
                        continue
                    social_scena_data.append(row)

            if len(social_scena_data) == 0:
                continue

            for social_scena in social_scena_data:
                scena = social_scena[key_map[task_name]].split("</think>")[-1].strip()
                background = extract_tag_content(scena, 'background')
                if task_name == 'users_sing_long_story(use_quality_high)':
                    story = extract_tag_content(scena, 'plots')
                    character = extract_tag_content(scena, 'characters')
                elif task_name == 'users_single_long_story_focusonbehavior(user_quality_high)':
                    story = extract_tag_content(scena, 'story')
                    all_characters = []
                    charac_begin_idx = 0
                    while True:
                        character, charac_end_idx = extract_tag_content_with_bounds(scena, 'character', charac_begin_idx)
                        if is_null(character) or character in all_characters:
                            break
                        all_characters.append(character)
                        charac_begin_idx = charac_end_idx
                    character = "; ".join(all_characters)
                elif task_name == 'users_thoughts_single_blog(user_quality_high)':
                    story = extract_tag_content(scena, 'thoughts')
                    character = extract_tag_content(scena, 'characters')
                else:
                    story = None
                    character = None
                if is_null(background) or is_null(character) or is_null(story):
                    continue

                messages = [
                    {"role": "user", "content": socialScenario_prompt.replace("{background}", background).replace("{character}", character).replace("{num}", word_count(story))},
                    {"role": "assistant", "content": story},
                ]
                cur_data.append({
                        "uid": userid,
                        "source": f"{dataset_name}.{task_name}", 
                        "messages": messages,
                    })
        train_data, test_data = filter_print(f"{dataset_name}.{task_name}", cur_data, max_data_num_per_task)
        print(f"skip_cnt: {skip_cnt}, all raw data: {all_cnt}")
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

    ############# writing imitation type1: history posts + topic
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/by_users_blog_summary(use_quality_high)'
    in_dir_rewritten = f'{in_dir}/by_users_quality_tagging_v2_rewritten'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".csv"):
            continue

        userid = filename.split(".")[0]

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'result' not in row or is_null(row['result']):
                    continue
                writing_imitation_data.append(row)
        
        rerewritten_data = []
        with open(os.path.join(in_dir_rewritten, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'rewritten_text' not in row or is_null(row['rewritten_text']):
                    continue
                rerewritten_data.append(row)

        final_data = []
        for i in range(len(writing_imitation_data)):
            clean_text = writing_imitation_data[i]['cleaned_text']
            for j, item in enumerate(rerewritten_data):
                if clean_text == item['cleaned_text']:
                    writing_imitation_data[i]['rewritten_text'] = item['rewritten_text']
                    final_data.append(writing_imitation_data[i])
                    rerewritten_data.pop(j)
                    break

        if len(final_data) < 3:
            continue

        for i, writing_imitation in enumerate(final_data):
            target_count = count_tokens(writing_imitation['rewritten_text'])
            if target_count > 1000 or target_count < 30:
                continue
            history_data = final_data[:i] + final_data[i+1:]
            sampled_num = random.randint(1, min(5, len(history_data)))
            sampled_history = random.sample(history_data, sampled_num)
            history_description = ""
            for idx_his, history in enumerate(sampled_history):
                history_description += f"\nPost {idx_his+1}:\n{truncate_text_by_words(history['rewritten_text'])}"

            try:
                user_prompt = writing_imitation_prompt_type1.replace("{past_posts}", history_description).replace("{scenario}", writing_imitation['result'])
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text']))},
                {"role": "assistant", "content": writing_imitation['rewritten_text']},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type1", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type1", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# writing imitation type2: history posts and completion
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/by_users_blog_summary(use_quality_high)'
    in_dir_rewritten = f'{in_dir}/by_users_quality_tagging_v2_rewritten'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".csv"):
            continue

        userid = filename.split(".")[0]

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'result' not in row or is_null(row['result']):
                    continue
                writing_imitation_data.append(row)

        rerewritten_data = []
        with open(os.path.join(in_dir_rewritten, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'rewritten_text' not in row or is_null(row['rewritten_text']):
                    continue
                rerewritten_data.append(row)
        
        final_data = []
        for i in range(len(writing_imitation_data)):
            clean_text = writing_imitation_data[i]['cleaned_text']
            for j, item in enumerate(rerewritten_data):
                if clean_text == item['cleaned_text']:
                    writing_imitation_data[i]['rewritten_text'] = item['rewritten_text']
                    final_data.append(writing_imitation_data[i])
                    rerewritten_data.pop(j)
                    break

        if len(final_data) < 3:
            continue

        for i, writing_imitation in enumerate(final_data):
            target_count = count_tokens(writing_imitation['rewritten_text'])
            if target_count > 1000 or target_count < 30:
                continue
            history_data = final_data[:i] + final_data[i+1:]
            sampled_num = random.randint(1, min(5, len(history_data)))
            sampled_history = random.sample(history_data, sampled_num)
            history_description = ""
            for idx_his, history in enumerate(sampled_history):
                history_description += f"\nPost {idx_his+1}:\n{truncate_text_by_words(history['rewritten_text'])}"

            front, back = split_text_randomly(writing_imitation['rewritten_text'])
            if not front or not back or len(back) < 5 or len(front) < 5:
                continue 

            try:
                user_prompt = writing_imitation_prompt_type2.replace("{past_posts}", history_description).replace("{front}", front)
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                {"role": "assistant", "content": back},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type2", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type2", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# writing imitation type3: writing style + topic
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/by_users_blog_summary(use_quality_high)'
    in_dir_rewritten = f'{in_dir}/by_users_quality_tagging_v2_rewritten'
    in_dir_style = f'{in_dir}/writing_style_rewritten_text(use_quality_high)'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".csv"):
            continue

        userid = filename.split(".")[0]

        writing_style_file = os.path.join(in_dir_style, f"{userid}.txt")
        if not os.path.exists(writing_style_file):
            continue
        with open(writing_style_file, 'r', encoding='utf-8') as f:
            writing_style = f.read().strip()
        if is_null(writing_style):
            continue

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'result' not in row or is_null(row['result']):
                    continue
                writing_imitation_data.append(row)
        
        rerewritten_data = []
        with open(os.path.join(in_dir_rewritten, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'rewritten_text' not in row or is_null(row['rewritten_text']):
                    continue
                rerewritten_data.append(row)

        final_data = []
        for i in range(len(writing_imitation_data)):
            clean_text = writing_imitation_data[i]['cleaned_text']
            for j, item in enumerate(rerewritten_data):
                if clean_text == item['cleaned_text']:
                    writing_imitation_data[i]['rewritten_text'] = item['rewritten_text']
                    final_data.append(writing_imitation_data[i])
                    rerewritten_data.pop(j)
                    break

        for i, writing_imitation in enumerate(final_data):
            target_count = count_tokens(writing_imitation['rewritten_text'])
            if target_count > 1000 or target_count < 30:
                continue

            try:
                user_prompt = writing_imitation_prompt_type3.replace("{style}", writing_style).replace("{scenario}", writing_imitation['result'])
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(writing_imitation['rewritten_text']))},
                {"role": "assistant", "content": writing_imitation['rewritten_text']},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type3", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type3", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    ############# writing imitation type4: writing style + completion
    max_data_num_per_task = 20000
    in_dir_writing_imitation = f'{in_dir}/by_users_blog_summary(use_quality_high)'
    in_dir_rewritten = f'{in_dir}/by_users_quality_tagging_v2_rewritten'
    in_dir_style = f'{in_dir}/writing_style_rewritten_text(use_quality_high)'
    writing_imitation_files = sorted(os.listdir(in_dir_writing_imitation))
    cur_data = []
    for filename in tqdm(writing_imitation_files, desc="Processing writing_imitation", total=len(writing_imitation_files)):
        if not filename.endswith(".csv"):
            continue

        userid = filename.split(".")[0]

        writing_style_file = os.path.join(in_dir_style, f"{userid}.txt")
        if not os.path.exists(writing_style_file):
            continue
        with open(writing_style_file, 'r', encoding='utf-8') as f:
            writing_style = f.read().strip()
        if is_null(writing_style):
            continue

        writing_imitation_data = []
        with open(os.path.join(in_dir_writing_imitation, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'result' not in row or is_null(row['result']):
                    continue
                writing_imitation_data.append(row)
        
        rerewritten_data = []
        with open(os.path.join(in_dir_rewritten, filename), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'rewritten_text' not in row or is_null(row['rewritten_text']):
                    continue
                rerewritten_data.append(row)

        final_data = []
        for i in range(len(writing_imitation_data)):
            clean_text = writing_imitation_data[i]['cleaned_text']
            for j, item in enumerate(rerewritten_data):
                if clean_text == item['cleaned_text']:
                    writing_imitation_data[i]['rewritten_text'] = item['rewritten_text']
                    final_data.append(writing_imitation_data[i])
                    rerewritten_data.pop(j)
                    break

        for i, writing_imitation in enumerate(final_data):
            target_count = count_tokens(writing_imitation['rewritten_text'])
            if target_count > 1000 or target_count < 30:
                continue

            front, back = split_text_randomly(writing_imitation['rewritten_text'])
            if not front or not back or len(back) < 5 or len(front) < 5:
                continue 

            try:
                user_prompt = writing_imitation_prompt_type4.replace("{style}", writing_style).replace("{front}", front)
            except:
                continue

            messages = [
                {"role": "user", "content": user_prompt.replace("{num}", word_count(back))},
                {"role": "assistant", "content": back},
            ]
            cur_data.append({
                    "uid": userid,
                    "source": f"{dataset_name}.writing_imitation_type4", 
                    "messages": messages,
                })
    train_data, test_data = filter_print(f"{dataset_name}.writing_imitation_type4", cur_data, max_data_num_per_task) 
    all_train_data.extend(train_data)
    all_test_data.extend(test_data)

    print(f"blogauthorship total train data nums: {len(all_train_data)}, total test data nums: {len(all_test_data)}")
    with open(output_train_file, 'a') as f:
        for r in all_train_data:
            f.write(json.dumps(r)+'\n')

    with open(output_test_file, 'a') as f:
        for r in all_test_data:
            f.write(json.dumps(r)+'\n')

def merge_data():
    in_train_files = [
        f'{HOME_DIR}/sft_dataset/train_reddit.json',
        f'{HOME_DIR}/sft_dataset/train_twitter.json',
        f'{HOME_DIR}/sft_dataset/train_amazon.json',
        f'{HOME_DIR}/sft_dataset/train_blogger.json',
    ]
    in_test_files = [
        f'{HOME_DIR}/sft_dataset/test_reddit.json',
        f'{HOME_DIR}/sft_dataset/test_twitter.json',
        f'{HOME_DIR}/sft_dataset/test_amazon.json',
        f'{HOME_DIR}/sft_dataset/test_blogger.json',
    ]

    out_train_file = f'{HOME_DIR}/sft_dataset/train.json'
    out_test_file = f'{HOME_DIR}/sft_dataset/test.json'

    out_dir = f'{HOME_DIR}/sft_dataset/train_splits'


    train_data = []
    for in_file in in_train_files:
        with open(in_file, "r") as f:
            for line in f:
                data = json.loads(line)
                del data['uid']  # Remove the 'uid' field
                del data['source']  # Remove the 'source' field
                train_data.append(data)
        print(f"Loaded {len(train_data)} training samples from {in_file}")

    test_data = []
    for in_file in in_test_files:
        with open(in_file, "r") as f:
            for line in f:
                data = json.loads(line)
                del data['uid']
                del data['source']
                test_data.append(data)
        print(f"Loaded {len(test_data)} test samples from {in_file}")

    random.shuffle(train_data)
    with open(out_train_file, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(out_test_file, "w") as f:
        json.dump(test_data, f, indent=4)



    # Load the dataset
    with open(out_train_file, "r") as f:
        dataset = json.load(f)

    # split the dataset into 10 equal parts
    num_splits = 10
    split_size = len(dataset) // num_splits
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    # Save each split to a separate file
    for i in range(num_splits):
        start_index = i * split_size
        end_index = (i + 1) * split_size if i < num_splits - 1 else len(dataset)
        split_data = dataset[start_index:end_index]
        
        out_file = os.path.join(out_dir, f"train_split_{i+1}.json")
        with open(out_file, "w") as f:
            json.dump(split_data, f, indent=4)
        
        print(f"Saved split {i+1} with {len(split_data)} samples to {out_file}")

if __name__ == "__main__":
    generate_for_reddit()
    generate_for_twitter()
    generate_for_amazon()
    generate_for_blogauthorship()
    merge_data()