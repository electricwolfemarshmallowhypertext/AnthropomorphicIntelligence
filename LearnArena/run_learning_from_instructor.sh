#!/bin/bash
set -euo pipefail

###############################################
#           User Configuration               #
###############################################

# Paths to local vLLM-ready weights
GEN_MODEL_PATH="${1:-}"          # Player-1 (student) model path - required
EVAL_MODEL_PATH="${2:-}"         # Instructor/Judge model path - required

# GPU allocation (override via command line if needed)
GEN_GPU_COUNT="${3:-2}"
EVAL_GPU_COUNT="${4:-2}"

# Model identifiers (used by vLLM)
GEN_MODEL_NAME="learnarena-student"
EVAL_MODEL_NAME="learnarena-instructor"

# Server ports
GEN_PORT=8000
EVAL_PORT=8001

# Generation hyper-parameters
MAX_TOKENS=512
TEMPERATURE=0.7
GEN_THREADS=32

# Evaluation hyper-parameters
EVAL_MAX_TOKENS=256
EVAL_TEMPERATURE=0.7
EVAL_THREADS=16

# Output directories
GEN_OUTPUT_DIR="./gen_output"
EVAL_OUTPUT_DIR="./eval_output"

# Dataset configuration (collected from the MAmmoTH math_eval release)
DATA_DIR="./data"
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

###############################################
#         Input Validation                   #
###############################################

if [[ -z "${GEN_MODEL_PATH}" || -z "${EVAL_MODEL_PATH}" ]]; then
    echo "Usage: $0 <student_model_path> <instructor_model_path> [student_gpus] [instructor_gpus]"
    exit 1
fi

###############################################
#         Derived Parameters                 #
###############################################

mkdir -p "${GEN_OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

GENERATE_DATASETS=$(IFS=,; echo "${DATASETS[*]}")

###############################################
#         Pipeline Execution                 #
###############################################

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Learning-from-Instructor math pipeline (vLLM mode)..."

python learning_from_instructor_experiment.py \
    --mode vllm \
    --datasets "${GENERATE_DATASETS}" \
    --data_dir "${DATA_DIR}" \
    --gen_output_dir "${GEN_OUTPUT_DIR}" \
    --eval_output_dir "${EVAL_OUTPUT_DIR}" \
    --stage pipeline \
    --gen_model_name "${GEN_MODEL_NAME}" \
    --gen_api_base "http://localhost:${GEN_PORT}" \
    --gen_model_path "${GEN_MODEL_PATH}" \
    --gen_port "${GEN_PORT}" \
    --gen_gpu "${GEN_GPU_COUNT}" \
    --gen_max_tokens "${MAX_TOKENS}" \
    --gen_temperature "${TEMPERATURE}" \
    --gen_threads "${GEN_THREADS}" \
    --eval_model_name "${EVAL_MODEL_NAME}" \
    --eval_api_base "http://localhost:${EVAL_PORT}" \
    --eval_model_path "${EVAL_MODEL_PATH}" \
    --eval_port "${EVAL_PORT}" \
    --eval_gpu "${EVAL_GPU_COUNT}" \
    --eval_max_tokens "${EVAL_MAX_TOKENS}" \
    --eval_temperature "${EVAL_TEMPERATURE}" \
    --eval_threads "${EVAL_THREADS}"

PIPELINE_STATUS=$?

if [[ ${PIPELINE_STATUS} -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline completed successfully. Results in ${EVAL_OUTPUT_DIR}."
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline failed. Check logs for details." >&2
    exit ${PIPELINE_STATUS}
fi
