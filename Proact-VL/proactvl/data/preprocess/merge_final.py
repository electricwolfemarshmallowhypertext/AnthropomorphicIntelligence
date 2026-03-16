# 合并所有的ann
import os
import json

TAG_MAP = {
    'cyberpunk_2077': 'Solo commentators',
    'starcraft2': 'Solo commentators',
    'baldurs_gate_3': 'Solo commentators',
    'black_myth_wukong': 'Solo commentators',
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
ann_dir = '/home/v-weicaiyan/ds/DATA/ann'
save_file = '/home/v-weicaiyan/ds/DATA/ann/gaming_all_val.jsonl'
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
ann_path = '/home/v-weicaiyan/ds/DATA/ann/black_myth_wukong_final_val.jsonl'
save_file = '/home/v-weicaiyan/ds/DATA/ann/black_myth_wukong_val.jsonl'
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
ann_path = '/home/v-weicaiyan/ds/DATA/ann/ego4d_final_val.jsonl'
save_file = '/home/v-weicaiyan/ds/DATA/ann/ego4d_val.jsonl'
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
    '/home/v-weicaiyan/ds/DATA/ann/gaming_all_val.jsonl',
    '/home/v-weicaiyan/ds/DATA/ann/black_myth_wukong_val.jsonl',
    '/home/v-weicaiyan/ds/DATA/ann/ego4d_val.jsonl',
]
save_file = '/home/v-weicaiyan/ds/DATA/ann/all_in_one.jsonl'
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