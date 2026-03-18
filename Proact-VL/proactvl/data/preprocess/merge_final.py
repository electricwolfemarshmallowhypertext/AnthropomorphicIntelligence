# Merge all annotations
import os
import json
import argparse

TAG_MAP = {
    'cyberpunk_2077': 'Solo commentators',
    'starcraft2': 'Solo commentators',
    'baldurs_gate_3': 'Solo commentators',
    # 'black_myth_wukong': 'Solo commentators',
    'elden_ring': 'Solo commentators',
    'tears_of_the_kingdom': 'Solo commentators',
    'yu_gi_oh': 'Multiple commentators',
    'lol': 'Multiple commentators',
    'csgo': 'Multiple commentators',
    'streetfighter6': 'Multiple commentators',
    'minecraft': 'Guidance',
    'ego4d': 'Ego4D',
    'soccernet': 'SoccerNet',
    'black_myth_wukong': 'Wukong',
}
# game
dataset_name_list = ['cyberpunk_2077', 'starcraft2', 'baldurs_gate_3',
                    'elden_ring', 'tears_of_the_kingdom', 'yu_gi_oh', 'lol', 'csgo',
                    'streetfighter6', 'minecraft']
parser = argparse.ArgumentParser()
parser.add_argument('--ann_dir', type=str, default='/home/v-weicaiyan/ds/DATA/ann')
args = parser.parse_args()

ann_dir = args.ann_dir
save_file = os.path.join(ann_dir, 'gaming_all_val.jsonl')
anns = []
for file in os.listdir(ann_dir):
    if file.endswith('_val.jsonl'):
        ann_path = os.path.join(ann_dir, file)
        dataset_name = file.replace('_final_val.jsonl', '')
        if dataset_name not in dataset_name_list:
            continue
        with open(ann_path, 'r') as f:
            for idx, line in enumerate(f):
                ann = json.loads(line)
                ann['dataset_name'] = dataset_name
                if dataset_name not in TAG_MAP:
                    raise ValueError(f'Cannot find tag mapping for dataset {dataset_name}')
                ann['tag'] = TAG_MAP[dataset_name]
                ann['idx'] = idx
                anns.append(ann)
        print(f'Loaded {len(anns)} annotations from {ann_path}')

with open(save_file, 'w') as f:
    for ann in anns:
        f.write(json.dumps(ann) + '\n')

# wukong
ann_path = os.path.join(ann_dir, 'black_myth_wukong_final_val.jsonl')
save_file = os.path.join(ann_dir, 'black_myth_wukong_val.jsonl')
dataset_name = 'black_myth_wukong'
with open(ann_path, 'r') as f:
    anns = []
    for idx, line in enumerate(f):
        ann = json.loads(line)

        ann['dataset_name'] = dataset_name
        ann['idx'] = idx
        
        if dataset_name not in TAG_MAP:
            raise ValueError(f'Cannot find tag mapping for dataset {dataset_name}')
        ann['tag'] = TAG_MAP[dataset_name]
        anns.append(ann)
    print(f'Loaded {len(anns)} annotations from {ann_path}')

with open(save_file, 'w') as f:
    for ann in anns:
        f.write(json.dumps(ann) + '\n')

# ego4d
ann_path = os.path.join(ann_dir, 'ego4d_final_val.jsonl')
save_file = os.path.join(ann_dir, 'ego4d_val.jsonl')
dataset_name = 'ego4d'
with open(ann_path, 'r') as f:
    anns = []
    for idx, line in enumerate(f):
        ann = json.loads(line)

        ann['dataset_name'] = dataset_name
        ann['idx'] = idx
        if dataset_name not in TAG_MAP:
            raise ValueError(f'Cannot find tag mapping for dataset {dataset_name}')
        ann['tag'] = TAG_MAP[dataset_name]
        anns.append(ann)
    print(f'Loaded {len(anns)} annotations from {ann_path}')

with open(save_file, 'w') as f:
    for ann in anns:
        f.write(json.dumps(ann) + '\n')



ann_path_list = [
    os.path.join(ann_dir, 'gaming_all_val.jsonl'),
    os.path.join(ann_dir, 'black_myth_wukong_val.jsonl'),
    os.path.join(ann_dir, 'ego4d_val.jsonl'),
]
save_file = os.path.join(ann_dir, 'all_in_one.jsonl')
all_anns = []
for ann_path in ann_path_list:
    with open(ann_path, 'r') as f:
        for line in f:
            ann = json.loads(line)
            all_anns.append(ann)
    print(f'Loaded {len(all_anns)} annotations from {ann_path}')
with open(save_file, 'w') as f:
    for ann in all_anns:
        f.write(json.dumps(ann) + '\n')
print(f'Saved all annotations to {save_file}')