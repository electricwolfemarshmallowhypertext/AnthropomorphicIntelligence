#!/bin/bash

# =============================================
# USER CONFIGURATION - PLEASE UPDATE THESE VALUES
# =============================================

# Model paths - Update these with your actual model paths
MODELS_BASE_DIR="PLACEHOLDER"  # Update this to your models directory
FIXED_PLAYER0_MODEL_PATH="${MODELS_BASE_DIR}/Qwen2.5-32B-Instruct"  # Fixed opponent model
PLAYER1_MODEL_PATHS=(
    "${MODELS_BASE_DIR}/Qwen2.5-1.5B-Instruct"
    "${MODELS_BASE_DIR}/Qwen2.5-7B-Instruct"
    "${MODELS_BASE_DIR}/Qwen2.5-14B-Instruct"
    "${MODELS_BASE_DIR}/Qwen2.5-32B-Instruct"
)

# Model names for vLLM
FIXED_PLAYER0_MODEL_NAME="Qwen2.5-32B-Instruct"
PLAYER1_MODEL_NAMES=(
    "Qwen2.5-1.5B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Qwen2.5-14B-Instruct"
    "Qwen2.5-32B-Instruct"
)

# Games to evaluate - Update this with your game list
GAMES="TicTacToe-v0,Poker-v0,Checkers-v0,Stratego-v0,TruthAndDeception-v0,UltimateTicTacToe-v0"


# Output directory - Update this if you want a different output location
BASE_RESULTS_DIR="results/experience_experiments"

# Fixed Player-0 (Opponent) Model Configuration
FIXED_PLAYER0_GPU_COUNT="8"
FIXED_PLAYER0_PORT="8000"

# Variable Player-1 (Learning Model) Configuration
PLAYER1_PORT="8001"
THREADS="8"

# =============================================
# FIXED CONFIGURATION - Usually no need to change
# =============================================

PYTHON_EXE="python"
EXPERIMENT_SCRIPT_PATH="experience_driven_experiment.py"

# === Script Execution ===
echo "Starting Scale Experience-Driven Experiments..."
echo "Fixed Player-0 (Opponent): ${FIXED_PLAYER0_MODEL_NAME} (GPU: ${FIXED_PLAYER0_GPU_COUNT})"

# Create base results directory if it doesn't exist
mkdir -p "${BASE_RESULTS_DIR}"

num_models=${#PLAYER1_MODEL_PATHS[@]}

for (( i=0; i<${num_models}; i++ )); do
    PLAYER1_PATH=${PLAYER1_MODEL_PATHS[$i]}
    PLAYER1_NAME=${PLAYER1_MODEL_NAMES[$i]}

    echo "----------------------------------------------------------------------"
    echo "Running with Player-1 (Learning Model): ${PLAYER1_NAME}"
    echo "----------------------------------------------------------------------"

    # Define a unique output folder for this specific model pair
    SANITIZED_PLAYER1_NAME=$(echo "${PLAYER1_NAME}" | tr '/' '_')
    SANITIZED_PLAYER0_NAME=$(echo "${FIXED_PLAYER0_MODEL_NAME}" | tr '/' '_')
    OUTPUT_FOLDER="${BASE_RESULTS_DIR}/Player0_${SANITIZED_PLAYER0_NAME}__Player1_${SANITIZED_PLAYER1_NAME}"
    mkdir -p "${OUTPUT_FOLDER}"

    # Run the experience-driven experiment
    echo "Running experience_driven_experiment.py..."
    ${PYTHON_EXE} "${EXPERIMENT_SCRIPT_PATH}" \
        --player0-model "${FIXED_PLAYER0_MODEL_NAME}" \
        --player0-path "${FIXED_PLAYER0_MODEL_PATH}" \
        --player1-model "${PLAYER1_NAME}" \
        --player1-path "${PLAYER1_PATH}" \
        --games "${GAMES}" \
        --output-file "${OUTPUT_FOLDER}/experience_results_${SANITIZED_PLAYER1_NAME}.jsonl" \
        --num-rounds 20 \
        --gpu "${FIXED_PLAYER0_GPU_COUNT}"

    if [ $? -eq 0 ]; then
        echo "Successfully completed run for Player-1: ${PLAYER1_NAME}"
    else
        echo "Error during run for Player-1: ${PLAYER1_NAME}" >&2
    fi
    echo "Output and logs for this run are in: ${OUTPUT_FOLDER}"
done

echo "----------------------------------------------------------------------"
echo "All Scale Experience-Driven Experiments finished."
echo "Base results directory: ${BASE_RESULTS_DIR}"
echo "----------------------------------------------------------------------" 