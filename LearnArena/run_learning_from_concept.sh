#!/bin/bash

# Default values
GAMES="TicTacToe-v0,Poker-v0,Checkers-v0,Stratego-v0,TruthAndDeception-v0,UltimateTicTacToe-v0"
OUTPUT_DIR="results"
NUM_ROUNDS=20
GPU=4

# Fixed Player-0 configuration
PLAYER0_MODEL="qwen2.5-32b-chat"
PLAYER0_PATH="/path/to/qwen2.5-32b"

# Player-1 model configurations
declare -A PLAYER1_CONFIGS=(
    ["qwen2.5-1.5b"]="/path/to/qwen2.5-1.5b"
    ["qwen2.5-7b"]="/path/to/qwen2.5-7b"
    ["qwen2.5-14b"]="/path/to/qwen2.5-14b"
    ["qwen2.5-32b"]="/path/to/qwen2.5-32b"
)

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run experiment for each Player-1 model
for model_name in "${!PLAYER1_CONFIGS[@]}"; do
    echo "Running experiment with Player-1 model: $model_name"
    
    # Set output file for this model
    OUTPUT_FILE="$OUTPUT_DIR/model_scale_${model_name}_results.json"
    
    # Run the experiment
    python model_scale_experiment.py \
        --games "$GAMES" \
        --output-file "$OUTPUT_FILE" \
        --num-rounds "$NUM_ROUNDS" \
        --player0-model "$PLAYER0_MODEL" \
        --player0-path "$PLAYER0_PATH" \
        --player1-model "$model_name" \
        --player1-path "${PLAYER1_CONFIGS[$model_name]}" \
        --gpu "$GPU"
    
    echo "Completed experiment for $model_name"
    echo "Results saved to $OUTPUT_FILE"
    echo "----------------------------------------"
done

# Print summary of all results
echo -e "\nExperiment Summary:"
echo "==================="
for model_name in "${!PLAYER1_CONFIGS[@]}"; do
    OUTPUT_FILE="$OUTPUT_DIR/model_scale_${model_name}_results.json"
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "\nResults for $model_name:"
        python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    results = json.load(f)
    for result in results:
        print(f'Game: {result[\"game\"]}')
        print(f'With Concept: {result[\"with_concept\"]}')
        print(f'Win Rate: {result[\"win_rate\"]:.2%}')
        print(f'Wins: {result[\"wins\"]}, Losses: {result[\"losses\"]}, Draws: {result[\"draws\"]}')
        print('---')
"
    fi
done 