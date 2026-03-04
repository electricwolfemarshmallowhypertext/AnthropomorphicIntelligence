from datasets import load_dataset
import json
import statistics
from tqdm import tqdm
from collections import defaultdict
import os
import re

HOME_DIR = os.path.expanduser("~/HumanLLM_data")

user_dataset = load_dataset("enryu43/twitter100m_users", split="train")
posts_dataset = load_dataset("enryu43/twitter100m_tweets", split="train")

print(user_dataset)
print(posts_dataset)

output_dir = f"{HOME_DIR}/twitter/raw_data"
uid_map_file = f"{HOME_DIR}/twitter/twitter_uid_map.json"
data_file = f"{HOME_DIR}/twitter/twitter_temp_data.json"
os.makedirs(output_dir, exist_ok=True)


MIN_POST_LENGTH = 300
MIN_POST_COUNT = 5
fields_to_keep_user = ['id', 'followers', 'description', 'location']
fields_to_keep_tweets = ['id', 'tweet', 'date']

user_data = {}
for user in tqdm(user_dataset, desc=f"Processing user_dataset"):
    assert user['user'] not in user_data
    filtered_record = {field: user[field] for field in fields_to_keep_user}
    user_data[user['user']] = filtered_record
    user_data[user['user']]['map_id'] = len(user_data) #start from 1
    user_data[user['user']]['history_count'] = 0
    user_data[user['user']]['behaviors'] = []

for record in tqdm(posts_dataset, desc=f"Processing posts_dataset"):
    if not record['tweet'] or record['tweet']=="[removed]" or record['tweet']=="[deleted]":
        continue
    if len(record['tweet'].strip()) < MIN_POST_LENGTH:
        continue
    filtered_record = {field: record[field] for field in fields_to_keep_tweets}
    assert record['user'] in user_data
    user_data[record['user']]['behaviors'].append(filtered_record)
    user_data[record['user']]['history_count'] += 1

user_data = {k: v for k, v in user_data.items() if v['history_count'] > MIN_POST_COUNT}

post_counts = [v['history_count'] for v in user_data.values()]
print("Average post count:", statistics.mean(post_counts))
print("Max post count:", max(post_counts))
print("Min post count:", min(post_counts))
print("User count:", len(user_data))

with open(data_file, 'w', encoding="utf-8") as f:
    json.dump(user_data, f, ensure_ascii=False, indent=4)


def is_corporate_account(user):
    corporate_keywords = ["company", "official", "network", "store", "brand", "business", "shop", "solutions", "digital", "services", "marketing"]
    others = ["Inc.", "Ltd.", "LLC"]
    # 1. check description
    if user['description']:
        if any(keyword in user['description'].lower().split() for keyword in corporate_keywords):
            return True
        if "service provider" in user['description'].lower():
            return True
        if any(keyword in user['description'] for keyword in others):
            return True

    return False

def is_advertisement_tweet(tweet):
    # keywords filter
    ad_keywords = ["buy now", "limited offer", "subscribe", "discount", "sale", "promo", "special deal", "click the link", "order now", "get it now"]
    if any(keyword in tweet.lower() for keyword in ad_keywords):
        return True

    return False

def is_news_or_retweet(tweet):
    if tweet.startswith("RT @"):  # Pure retweets
        return True
    # if tweet.count("http") == 1 and len(tweet.split()) < 30:  # Only one link and very few words
    #     return True
    return False

def filter_users(users):
    filtered_users = {}
    
    for u_name, user in tqdm(users.items(), desc=f"Filtering users", total=len(users)):
        if is_corporate_account(user):
            continue #Skip Business Account
        
        filtered_tweets = []
        for tweet in user['behaviors']:
            tweet_text = tweet['tweet']
            if is_advertisement_tweet(tweet_text) or is_news_or_retweet(tweet_text):
                continue  # Filter low-quality tweets
            filtered_tweets.append(tweet)
        
        if len(filtered_tweets) > MIN_POST_COUNT:  
            user['behaviors'] = filtered_tweets
            user['history_count'] = len(filtered_tweets)
            filtered_users[u_name] = user
    
    return filtered_users

filtered_users = filter_users(user_data)
post_counts = [v['history_count'] for v in filtered_users.values()]
print("Average post count:", statistics.mean(post_counts))
print("Max post count:", max(post_counts))
print("Min post count:", min(post_counts))
print("User count:", len(filtered_users))


users_wo_behaviors = {}

for u_name, metas in filtered_users.items():
    users_wo_behaviors[u_name] = {k: metas[k] for k in metas if k != 'behaviors'}
    uid = metas['map_id']
    output_file = os.path.join(output_dir, f"{uid}.json")
    with open(output_file, 'w') as f:
        for tweet in metas['behaviors']:
            f.write(json.dumps(tweet) + "\n")

with open(uid_map_file, 'w') as f:
    json.dump(users_wo_behaviors, f)

