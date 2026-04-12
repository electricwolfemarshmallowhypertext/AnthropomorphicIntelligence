#!/bin/bash
set -e

python -m radar.sft.generate \
    --input_file data/raw/train_2class.jsonl \
    --output_file data/cot/train_2class.jsonl \
    --num_classes 2 \
    --model_name o3 \
    --base_url https://api.openai.com/v1

python -m radar.sft.preprocess \
    --input_file data/cot/train_2class.jsonl \
    --output_file data/sft/train_2class.jsonl

accelerate launch radar/sft/train.py \
    --model_name Qwen/Qwen3-8B \
    --train_data_path data/sft/train_2class.jsonl \
    --output_dir saves/sft_2class_8b \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 4096 \
    --save_steps 100 \
    --logging_steps 10 \
    --bf16 \
    --deepspeed configs/ds.json \
    --num_classes 2 \
    2>&1 | tee train.log
