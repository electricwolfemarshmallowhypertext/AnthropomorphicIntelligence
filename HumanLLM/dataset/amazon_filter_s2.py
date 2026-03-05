import json
import os
import shutil
import argparse
from tqdm import tqdm


amazon_categories = [
    'Arts_Crafts_and_Sewing', 'Automotive', 'Baby_Products', 'Beauty_and_Personal_Care', 'Books', 
    'CDs_and_Vinyl', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Electronics', 
    'Grocery_and_Gourmet_Food', 'Health_and_Household', 'Home_and_Kitchen',
    'Industrial_and_Scientific', 'Sports_and_Outdoors', 'Video_Games',
]


def process_category(category, input_dir, output_root):
    behavior_file = os.path.join(input_dir, f'amazon_{category}_behavior.json')

    # each category has its own output directory
    output_cat_dir = os.path.join(output_root, category)
    os.makedirs(output_cat_dir, exist_ok=True)

    uid_map_file = os.path.join(input_dir, f'amazon_{category}_uid_map.json')

    behavior_data = json.load(open(behavior_file, 'r'))

    uid_map = {}

    for user_id, user_data in tqdm(behavior_data.items(), desc=f'Processing {category}'):
        uid_map[user_id] = len(uid_map)

        user_file = os.path.join(output_cat_dir, f'{uid_map[user_id]}.json')

        with open(user_file, 'w') as f:
            for item in user_data:
                f.write(json.dumps(item) + '\n')

    with open(uid_map_file, 'w') as f:
        json.dump(uid_map, f)

    print(f'Processed {category}. {len(uid_map)} users.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    args = parser.parse_args()

    for category in amazon_categories:
        process_category(category, args.input_dir, args.output_root)

    print('All categories processed.')


if __name__ == '__main__':
    main()
