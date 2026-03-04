#!/bin/bash
set -euo pipefail

cd ~/AnthropomorphicIntelligence/Human_LLM/dataset

raw_data_dir="$HOME/HumanLLM_data/raw"
save_data_dir="$HOME/HumanLLM_data/amazon/raw_data"
mkdir -p "${save_data_dir}"

amazon_categories=(
    Arts_Crafts_and_Sewing
    Automotive
    Baby_Products
    Beauty_and_Personal_Care
    Books
    CDs_and_Vinyl
    Cell_Phones_and_Accessories
    Clothing_Shoes_and_Jewelry
    Electronics
    Grocery_and_Gourmet_Food
    Health_and_Household
    Home_and_Kitchen
    Industrial_and_Scientific
    Sports_and_Outdoors
    Video_Games
)

for category in "${amazon_categories[@]}"; do
    echo "Processing ${category}..."

    python amazon_filter_s1.py \
        --meta_file "${raw_data_dir}/meta_${category}.jsonl.gz" \
        --review_file "${raw_data_dir}/${category}.jsonl.gz" \
        --save_data_file "${save_data_dir}/amazon_${category}_behavior.json" \
        --save_metadata_file "${save_data_dir}/amazon_${category}_content.json"
done

echo "All categories processed. Next step: filter s2"

python amazon_filter_s2.py --input_dir "${save_data_dir}" --output_root "${save_data_dir}"

echo "All categories processed."