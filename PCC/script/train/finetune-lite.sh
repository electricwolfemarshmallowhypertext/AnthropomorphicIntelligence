#!/bin/bash

LEARNING_RATE=5e-5
COMPRESS_RATIO=4

COMPRESS_MODEL=BroAlanTaps/Stage1-PCC-Lite-4x
CONVERTER_MODEL=${COMPRESS_MODEL}
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct

TRAIN_DATA_DIR=BroAlanTaps/Stage2-PCC-Lite-SFT-Squad
VALID_DATA_DIR=BroAlanTaps/Stage2-PCC-Lite-SFT-Squad

PROJECT_NAME=PCC-Lite-Stage2-Compress:${COMPRESS_RATIO}x-Lr:${LEARNING_RATE}
OUTPUT_DIR=train/${PROJECT_NAME}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
EMBED_LEN=$((256 / COMPRESS_RATIO))

echo "Embed length: $EMBED_LEN"
deepspeed --include node-0:0,1,2,3 --master_port $MASTER_PORT train.py \
    --stage 2 \
    --deepspeed config/ds_config_zero1_bf16.json \
    --use_lora False \
    --do_train \
    --do_eval  \
    --compress_model ${COMPRESS_MODEL} \
    --converter_model ${CONVERTER_MODEL} \
    --decoder_model ${LLM_MODEL} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --train_data_dir ${TRAIN_DATA_DIR} \
    --valid_data_dir ${VALID_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir train/log_dir \
    --overwrite_output_dir True \
    --per_device_train_batch_size 4 \
    --bf16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 6 \
    --save_steps 500 \
    --eval_steps 250 \
    --metric_for_best_model eval_rag_val_loss \
    --warmup_steps 100 \
    --logging_steps 1 \
    --max_grad_norm 0.5 \
    --embed_len ${EMBED_LEN} \
    --remove_unused_columns False \
    --dataloader_num_workers 32 \
    --save_total_limit 8 \
    --save_strategy steps \
    --run_name ${PROJECT_NAME} \
    --skip_memory_metrics False