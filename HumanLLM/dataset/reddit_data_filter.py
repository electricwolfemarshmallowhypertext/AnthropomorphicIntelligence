from datasets import load_dataset
import json
import statistics
from tqdm import tqdm
from collections import defaultdict
import re
import os
import pandas as pd

HOME_DIR = os.path.expanduser("~/HumanLLM_data")

def is_bot_or_spam(user):
    bot_keywords = ["[deleted]", "autotldr", "automoderator"]
    return any(keyword in user.lower() for keyword in bot_keywords)

def process_submissions():
    submissions_dataset = load_dataset("HuggingFaceGECLM/REDDIT_submissions")
    print(len(submissions_dataset))

    fields_to_keep_submissions = ['author', 'created_utc', 'domain', 'id', 'is_self', 'name', 'num_comments', 'score', 'selftext', 'subreddit_id', 'title']

    MIN_SUBMISSION_LENGTH = 300
    MIN_SUBMISSION_COUNT = 5

    output_dir = f"{HOME_DIR}/reddit/before_raw_data"
    uid_map_file = f"{HOME_DIR}/reddit/reddit_uid_map.json"
    sub2uid_map_file = f"{HOME_DIR}/reddit/reddit_sub2uid_map.json"
    os.makedirs(output_dir, exist_ok=True)

    def is_low_quality_post(post):
        # if contains_too_many_urls(post['selftext']) or contains_too_many_urls(post['title']):
        #     return True

        if len(post['title']) + len(post['selftext']) < MIN_SUBMISSION_LENGTH:
            return True
        
        if len(post['selftext'].split()) < 10 and post['selftext'].strip().endswith("?"):
            return True
        
        return False

    uid_map = {}
    sub2uid_map = {}
    for split_name, split_dataset in submissions_dataset.items():
        for idx, record in tqdm(enumerate(split_dataset), desc=f"Processing submissions_dataset {split_name}", total=len(split_dataset)):
            if not record['author'] or not record['created_utc'] or not record['id']:
                continue
            if is_bot_or_spam(record['author']):
                continue
            if record['is_self'] == "False" or record['distinguished'] == "moderator" or record['distinguished'] == "admin":
                continue
            if int(record['score']) < 2 and int(record['num_comments']) < 2:
                continue
            if not record['title'] or "this comment has been removed" in record['title'].strip().lower() or "this submission has been removed" in record['title'].strip().lower() or record['title'].strip().lower() in ["[removed]", "[deleted]"]:
                record['title'] = ""
            if not record['selftext'] or "this comment has been removed" in record['selftext'].strip().lower() or "this submission has been removed" in record['selftext'].strip().lower() or record['selftext'].strip().lower() in ["[removed]", "[deleted]"]:
                record['selftext'] = ""
            
            if is_low_quality_post(record):
                continue
            
            author = record['author']
            uid_map[author] = uid_map.get(author, len(uid_map))
            sub2uid_map[record['id']] = uid_map[author]
            ouptut_file = os.path.join(output_dir, f"{uid_map[author]}.json")
            filtered_record = {field: record[field] for field in fields_to_keep_submissions}
            filtered_record['split'] = split_name
            filtered_record['type'] = "t3"
            
            with open(ouptut_file, 'a') as f:
                f.write(json.dumps(filtered_record) + "\n")

            # row_df = pd.DataFrame([filtered_record])
            # if os.path.exists(ouptut_file):
            #     row_df.to_csv(ouptut_file, mode='a', header=False, index=False)
            # else:
            #     row_df.to_csv(ouptut_file, mode='w', header=True, index=False)

    print(f"Total unique authors: {len(uid_map)}")
    print(f"Total unique submissions: {len(sub2uid_map)}")

    with open(uid_map_file, 'w') as f:
        json.dump(uid_map, f)

    with open(sub2uid_map_file, 'w') as f:
        json.dump(sub2uid_map, f)


def filter_low_submissions():
    in_dir = "{HOME_DIR}/reddit/before_raw_data"
    out_dir = "{HOME_DIR}/reddit/raw_data"

    files = os.listdir(in_dir)
    for file in tqdm(files):
        file_path = os.path.join(in_dir, file)
        lines = []
        with open(file_path, 'r') as f:
            for line in f:
                lines.append(line)
        
        if len(lines) < 6:
            continue
        
        out_file_path = os.path.join(out_dir, file.replace(".csv", ".json"))
        with open(out_file_path, 'w') as f:
            for line in lines:
                f.write(line)

def process_comments():
    in_dir = "{HOME_DIR}/reddit/before_raw_data"
    out_dir = "{HOME_DIR}/reddit/raw_data"
    sub2uid_map_file = "{HOME_DIR}/reddit/reddit_sub2uid_map.json"
    uid_map_file = "{HOME_DIR}/reddit/reddit_uid_map.json"

    with open(sub2uid_map_file, 'r') as f:
        sub2uid_map = json.load(f)
    with open(uid_map_file, 'r') as f:
        uid_map = json.load(f)

    comments_dataset = load_dataset("HuggingFaceGECLM/REDDIT_comments")
    fields_to_keep_comments = ['author', 'body', 'created_utc', 'id', 'name', 'link_id', 'parent_id', 'score', 'subreddit_id']
    MIN_COMMENT_LENGTH = 200
    for split_name, split_dataset in comments_dataset.items():
        for idx, record in tqdm(enumerate(split_dataset), desc=f"Processing comments_dataset {split_name}", total=len(split_dataset)):
            if not record['author'] or not record['created_utc'] or not record['id'] or record['author'] not in uid_map:
                continue
            if is_bot_or_spam(record['author']):
                continue
            if not record['body'] or record['body'].strip().lower() in ["[removed]", "[deleted]"]:
                continue
            if len(record['body']) < MIN_COMMENT_LENGTH:
                continue
            if "this comment has been removed" in record['body'].strip().lower() or "this submission has been removed" in record['body'].strip().lower():
                continue
            if not record['link_id'] or record['link_id'] == "[deleted]" or record['link_id'] != record['parent_id']:
                continue
            if record['link_id'][3:] not in sub2uid_map:
                continue
            uid = sub2uid_map[record['link_id'][3:]]
            ouptut_file = os.path.join(out_dir, f"{uid_map[record['author']]}.json")
            if not os.path.exists(os.path.join(in_dir, f"{uid}.csv")) or not os.path.exists(ouptut_file):
                continue
            submission_data = None
            with open(os.path.join(in_dir, f"{uid}.csv"), 'r') as f:
                for line in f:
                    submission_record = json.loads(line)
                    if submission_record['id'] == record['link_id'][3:]:
                        submission_data = submission_record
                        break
            if not submission_data:
                continue
            filtered_record = submission_data.copy()
            filtered_record['type'] = "t1"
            filtered_record['split'] = split_name
            for field in fields_to_keep_comments:
                filtered_record["com_"+field] = record[field]

            with open(ouptut_file, 'a') as f:
                f.write(json.dumps(filtered_record) + "\n")

if __name__ == "__main__":
    process_submissions()
    filter_low_submissions()
    process_comments()