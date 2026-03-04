#!/bin/bash

# ==========settings ==========
BASE_DIR=~/AnthropomorphicIntelligence/Human_LLM
CUDA_DEVICES=0,1,2,3
API_KEY=token-abc123
PORT=8001
GPU_MEM_UTIL=0.94
DTYPE=bfloat16
TP_SIZE=4
MAX_MODEL_LEN=8192
TRUST_REMOTE_CODE="--trust-remote-code"
DATASET_FILE=~/HumanLLM_data/sft_dataset/test.json
MAX_SAMPLES=500
DOMAINS="amazon.Arts_Crafts_and_Sewing.item_selection amazon.Automotive.item_selection amazon.Baby_Products.item_selection amazon.Beauty_and_Personal_Care.item_selection amazon.Books.item_selection amazon.CDs_and_Vinyl.item_selection amazon.Cell_Phones_and_Accessories.item_selection amazon.Clothing_Shoes_and_Jewelry.item_selection amazon.Electronics.item_selection amazon.Grocery_and_Gourmet_Food.item_selection amazon.Health_and_Household.item_selection amazon.Home_and_Kitchen.item_selection amazon.Industrial_and_Scientific.item_selection amazon.Sports_and_Outdoors.item_selection amazon.Video_Games.item_selection"
TEMPERATURE=0.1
MAX_SEQ_LENGTH=8192


MODEL_LIST=(
  xxxx/Llama-3.1-8B-Instruct_lr5e-6_seq8192_batch64_zero3_3epoch/merged_model
  xxxx/Qwen2.5-3B-Instruct_lr5e-6_seq8192_batch64_zero2_3epoch/merged_model
)

OUTPUT_DIR=~/AnthropomorphicIntelligence/Human_LLM/output/test_indomain
INFER_LOG=infer.log


cd $BASE_DIR
OUTPUT_JSON_LIST=()

for MODEL_PATH in "${MODEL_LIST[@]}"
do
    MODEL_DIR=$(basename $(dirname "$MODEL_PATH"))
    MODEL_SUBDIR=$(basename "$MODEL_PATH")
    MODEL_NAME="${MODEL_DIR}_${MODEL_SUBDIR}"
    OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}.json"
    OUTPUT_JSON_LIST+=("$OUTPUT_FILE")

    LOGFILE="server_${PORT}.log"
    echo "============================================"
    echo "Launching vllm serve for $MODEL_NAME ..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve "$MODEL_PATH" \
      --api-key $API_KEY \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --port $PORT \
      --dtype $DTYPE \
      --tensor-parallel-size $TP_SIZE \
      --max-model-len $MAX_MODEL_LEN \
      $TRUST_REMOTE_CODE > $LOGFILE 2>&1 &
    SERVER_PID=$!

    # vllm serve ready
    echo "Waiting for vllm server to be ready..."
    while ! (grep -q "Uvicorn running on" $LOGFILE || grep -q "Application startup complete" $LOGFILE); do
        sleep 2
        if ! ps -p $SERVER_PID > /dev/null; then
            echo "vllm serve crashed! Check $LOGFILE"
            exit 1
        fi
    done
    echo "vllm serve ready!"

    # inference
    export OPENAI_API_KEY=$API_KEY;
    export OPENAI_API_BASE="http://localhost:${PORT}/v1";
    python training/inference.py \
        --dataset_file "$DATASET_FILE" \
        --max_samples_per_task $MAX_SAMPLES \
        --domains $DOMAINS \
        --temperature $TEMPERATURE \
        --model_name_or_path "$MODEL_PATH" \
        --output_file "$OUTPUT_FILE" \
        --max_seq_length $MAX_SEQ_LENGTH \
        --cal_acc "item selection" > $INFER_LOG 2>&1

    #kill server
    echo "Killing vllm server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null

    rm -f $LOGFILE

    echo "Done with $MODEL_NAME"
done