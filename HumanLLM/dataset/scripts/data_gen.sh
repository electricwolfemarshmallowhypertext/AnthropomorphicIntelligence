#!/bin/bash

cd ~/AnthropomorphicIntelligence/HumanLLM/dataset


# data filtering
python reddit_data_filter.py
python twitter_data_filter.py
bash scripts/download_amazon.sh
bash scripts/amazon_filter.sh
bash scripts/blogger_filter.sh


# data synthesis
python reddit_data_gen.py
python twitter_data_gen.py
python amazon_data_gen.py
python blogger_data_gen.py


# data quality control
python data_judger.py
python blogger_data_judger.py


# generate sft data
python generate_sft_data.py