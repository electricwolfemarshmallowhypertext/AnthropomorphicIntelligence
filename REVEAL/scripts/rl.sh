#!/bin/bash
set -e

python -m radar.rl.preprocess \
    --input_dir data/aigcbank \
    --model_name saves/sft_2class_8b \
    --output_file data/rl/train_2class_8b.jsonl \
    --sft_train_file data/sft/train_2class.jsonl \
    --sft_test_file data/test/test_2class.jsonl \
    --num_classes 2

accelerate launch \
    --config_file configs/ds.yaml \
    radar/rl/train.py \
    --dataset_name data/rl/train_2class_8b.jsonl \
    --model_name_or_path saves/sft_2class_8b \
    --output_dir saves/rl_2class_8b \
    --num_train_epochs 1.0 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --num_generations 8 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --attn_implementation "flash_attention_2"\
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" \
    2>&1 | tee train.log

python -m radar.rl.merge \
    --model_path saves/sft_2class_8b \
    --lora_path saves/rl_2class_8b \
    --output_path saves/model_2class_8b