#!/bin/bash

# target directory
SAVE_DIR=~/HumanLLM_data/raw
mkdir -p ${SAVE_DIR}

BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

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
    echo "Downloading ${category}..."

    # review file
    wget -c "${BASE_URL}/review_categories/${category}.jsonl.gz" -P ${SAVE_DIR}

    # meta file
    wget -c "${BASE_URL}/meta_categories/meta_${category}.jsonl.gz" -P ${SAVE_DIR}

done

echo "All downloads completed."
