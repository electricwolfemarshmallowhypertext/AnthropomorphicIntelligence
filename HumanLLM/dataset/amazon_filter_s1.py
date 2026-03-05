import os
import re
import json
import gzip
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import copy
import argparse

'''
Set seeds
'''
seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--full_data_name", type=str, help="", default="Amazon"
    )
    parser.add_argument(
        "--meta_file", type=str, help="", default=""
    )
    parser.add_argument(
        "--review_file", type=str, help="", default=""
    )
    parser.add_argument(
        "--save_data_file", type=str, default='', help=""
    )
    parser.add_argument(
        "--save_metadata_file", type=str, default='', help=""
    )
    args = parser.parse_args()
    return args

def get_value_by_key(keys, d):
    for key in keys:
        if key in d:
            return d[key]
    return ""

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def Amazon(rating_score=-1, args=None):
    '''
    return (user, item, timestamp) sort in get_interaction
    '''
    items_with_title = defaultdict(int)
    title_dedup = set()
    with gzip.open(args.meta_file, "r") as fr:
        for line in tqdm(fr, desc="load meta data"):
            line = json.loads(line)
            if "title" not in line or line['title'] is None or len(line['title'])==0:
                continue
            title = line['title'].strip().lower()
            if title in title_dedup or len(title) == 0:
                continue
            title_dedup.add(title)
            items_with_title[line['parent_asin']] += 1 

    print(f"items with title: {add_comma(len(items_with_title))}")
    datas = []
    data_dict = {}
    with gzip.open(args.review_file, "r") as fr:
        for line in tqdm(fr, desc="load all interactions"):
            # try:
            line = json.loads(line)
            user = line['user_id']
            item = line['parent_asin']
            if not line['rating'] or float(line['rating']) <= rating_score or item not in items_with_title: # remove low rating
                continue
            if (user, item) in data_dict:
                continue
            if not line['text'] or len(line['text'].strip().split()) <= 5: 
                continue
            time = line['timestamp']
            text = line['text'].strip()
            title = line['title'].strip() if line['title'] else ''
            rating = line['rating']
            data_dict[(user, item)] = int(time) # merge duplicate interactions, keep the first record
            datas.append((user, item, int(time), rating, title, text))
    print(f"total interactions: {add_comma(len(datas))}")
    return datas

def Amazon_meta(item_ids, args):
    meta_datas = {}
    with gzip.open(args.meta_file, "r") as fr:
        for line in tqdm(fr, desc="load meta data"):
            line = json.loads(line)
            if line['parent_asin'] not in item_ids:
                continue
            # if "title" in line:
            line['title'] = re.sub(r'\n\t', ' ', line['title']).encode('UTF-8', 'ignore').decode('UTF-8')
            assert len(line['title'].strip()) > 0
                # line['title'] = line['title'].split(",")[0]
            if "description" in line:
                if type(line['description']) == str:
                    line['description'] = re.sub(r'\n\t', ' ', line['description']).encode('UTF-8', 'ignore').decode('UTF-8')
                elif type(line['description']) == list:
                    descs = []
                    for desc in line['description']:
                        desc = re.sub(r'\n\t', ' ', desc).encode('UTF-8', 'ignore').decode('UTF-8')
                        descs.append(desc)
                    line['description'] = descs
            if 'images' in line:
                del line['images']
            if 'videos' in line:
                del line['videos']

            for key in ['title', 'average_rating', 'rating_number', 'price']:
                value = get_value_by_key([key], line)
                if value is None or value=="":
                    line[key] = ""
            for key in ['description', 'categories', 'features']:
                value = get_value_by_key([key], line)
                if value is None or value=="":
                    line[key] = []
            value = get_value_by_key(['details'], line)
            if value is None or value=="":
                line['details'] = {}

            if 'details' in line and 'Pricing' in line['details']:
                del line['details']['Pricing']
            # for key in ['Genre', 'Format', 'Director', 'Release date', 'Actors', 'Producers', 'Studio']:
            #     value = get_value_by_key([key], line['details'])
            #     if value is None or value=="":
            #         line['details'][key] = ""

            meta_datas[line['parent_asin']] = line
    return meta_datas
        
def add_comma(num): # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def get_interaction(datas):
    # get user interaction sequence for sequential recommendation
    user_seq = {}
    for data in datas:
        user, item, time, rating, title, text = data
        if user in user_seq:
            user_seq[user].append((item, time, rating, title, text))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time, rating, title, text))

    item_freq = defaultdict(int)
    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # sort by time
        items = []
        for t in item_time:
            items.append({'item_id': t[0], 'timestamp': t[1], 'rating': t[2], 'title': t[3], 'text': t[4]})
            item_freq[t[0]] += 1
        user_seq[user] = items
    return user_seq, item_freq

def check_Kcore(user_items, user_core, item_core):
    # K-core user_core item_core, return False if any user/item < core
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item['item_id']] += 1

    for user, _ in user_items.items():
        if user_count[user] < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # all user/item >= core

def filter_Kcore(user_items, user_core, item_core):
    # Filter the K-core in a loop
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        cur_user_items = copy.deepcopy(user_items)
        for user, _ in user_items.items():
            if user_count[user] < user_core: # remove user
                cur_user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item['item_id']] < item_core:
                        cur_user_items[user].remove(item)
        user_items = cur_user_items
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    total_interactions = 0
    for user, items in user_items.items():
        total_interactions += len(items)
    print("interactions: {0} after k-core filter".format(add_comma(total_interactions)))
    return user_items

def filter_and_dedup(user_items, item_freq, user_core):
    ### filter out low frequency items and deduplication user sequences
    user_items_f1 = {}
    # filter out low frequency items
    item_freq = dict(sorted(item_freq.items(), key=lambda x: x[1], reverse=True))
    top_items = set(list(item_freq.keys())[0:len(item_freq)//2]) # top 50% frequent items

    for idx, (user, items) in tqdm(enumerate(user_items.items()), desc='filter_and_dedup'):
        cur_items = [item for item in items if item['item_id'] in top_items]
        if len(cur_items) >= user_core:
            user_items_f1[user] = cur_items
    return user_items_f1

def shuffle_dict(d):
    """Shuffle a dictionary and return a new dictionary with shuffled key-value pairs."""
    items = list(d.items())
    random.shuffle(items)
    return dict(items) 

def main_process(data_name, args, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp', 'Steam'}
    rating_score = -1.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 8
    item_core = 3

    datas = Amazon(rating_score, args)  # list of [user, item, timestamp]

    user_items, item_freq = get_interaction(datas) # dict of {user: interaction list sorted by time} 
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    print(f'User Num: {len(user_items)}')

    user_items = filter_and_dedup(user_items, item_freq, user_core)
    print(f'filter_and_dedup completed!')
    
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    
    user_count, item_count, isKcore = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    assert isKcore==True
    user_count_list = list(user_count.values()) # user click count
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values()) # item click count
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])

    user_num = len(user_items)
    item_num = len(item_count)
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    print('Begin extracting meta infos...')
    
    meta_infos = Amazon_meta(set(item_count.keys()), args)

    print(f'{data_name} & {add_comma(user_num)} & {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f} & {add_comma(interact_num)} & {sparsity:.2f}% ')

    # -------------- Save Data ---------------
    shuffled_user_items = shuffle_dict(user_items)

    with open(args.save_metadata_file, "w", encoding="utf-8") as f:
        json.dump(meta_infos, f, ensure_ascii=False, indent=4)

    with open(args.save_data_file, "w", encoding="utf-8") as f:
        json.dump(shuffled_user_items, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main_process(args.full_data_name, args=args, data_type='Amazon')