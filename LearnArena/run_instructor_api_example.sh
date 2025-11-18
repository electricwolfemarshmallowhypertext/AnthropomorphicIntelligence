#!/bin/bash
set -euo pipefail

# ===============================================
# Learning-from-Instructor Math Evaluation (API)
# ===============================================

DATA_DIR="./data"
GEN_OUTPUT_DIR="learnarena_results_api/gen"
EVAL_OUTPUT_DIR="learnarena_results_api/eval"
NUM_THREADS=32
EVAL_THREADS=16
MAX_TOKENS=512
EVAL_MAX_TOKENS=256
TEMPERATURE=0.7
EVAL_TEMPERATURE=0.7

# Datasets collected from the MAmmoTH math_eval release
DATASETS=(
    "aqua.jsonl"
    "gsm8k.jsonl"
    "math-500.jsonl"
    "mmlu_math.jsonl"
    "numglue.jsonl"
    "sat.jsonl"
    "simuleq.jsonl"
    "svamp.jsonl"
)

# API configuration
PLAYER0_API_KEY="${API_KEY_0:-your-player0-api-key}"
PLAYER1_API_KEY="${API_KEY_1:-your-player1-api-key}"
PLAYER0_API_BASE="https://api.openai.com/v1"
PLAYER1_API_BASE="https://api.openai.com/v1"
PLAYER0_MODEL="gpt-4o-mini"
PLAYER1_MODEL="gpt-4o"

mkdir -p "${GEN_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}" logs

GENERATE_DATASETS=$(IFS=,; echo "${DATASETS[*]}")

echo "========================================="
echo "Learning-from-Instructor Math (API Mode)"
echo "========================================="
echo "Student model : ${PLAYER1_MODEL}"
echo "Instructor model: ${PLAYER0_MODEL}"
echo "Datasets: ${GENERATE_DATASETS}"
echo "Output: ${EVAL_OUTPUT_DIR}"
echo "========================================="
echo ""

python learning_from_instructor_experiment.py \
    --mode api \
    --datasets "${GENERATE_DATASETS}" \
    --data_dir "${DATA_DIR}" \
    --gen_output_dir "${GEN_OUTPUT_DIR}" \
    --eval_output_dir "${EVAL_OUTPUT_DIR}" \
    --stage pipeline \
    --gen_model_name "${PLAYER1_MODEL}" \
    --gen_api_base "${PLAYER1_API_BASE}" \
    --gen_api_key "${PLAYER1_API_KEY}" \
    --gen_max_tokens "${MAX_TOKENS}" \
    --gen_temperature "${TEMPERATURE}" \
    --gen_threads "${NUM_THREADS}" \
    --eval_model_name "${PLAYER0_MODEL}" \
    --eval_api_base "${PLAYER0_API_BASE}" \
    --eval_api_key "${PLAYER0_API_KEY}" \
    --eval_max_tokens "${EVAL_MAX_TOKENS}" \
    --eval_temperature "${EVAL_TEMPERATURE}" \
    --eval_threads "${EVAL_THREADS}" \
    --overwrite

STATUS=$?

if [[ ${STATUS} -eq 0 ]]; then
    echo ""
    echo "✓ API evaluation completed successfully!"
    echo "Generation output : ${GEN_OUTPUT_DIR}"
    echo "Evaluation output : ${EVAL_OUTPUT_DIR}"
else
    echo ""
    echo "✗ API evaluation failed. Check logs and HTTP responses for details." >&2
    exit ${STATUS}
fi
