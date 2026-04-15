#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="config/evaluate.yaml"

# Record root. If TITLE/TITLES is set, script will evaluate INPUT_ROOT/<title>.
INPUT_ROOT="output/record"

# Output root for async evaluator.
# If path ends with by_model, results are written to by_model/<character_llm>/...
OUTPUT_DIR="output/evaluation/by_model"

# Optional: evaluate only one title folder under INPUT_ROOT.
TITLE=""

# Optional batch titles; if non-empty, these override TITLE.
TITLES=(
  # "autogen_scene_xxx"
)

# Optional file with one title per line (non-empty lines only).
TITLES_FILE=""

# Evaluator runtime options.
GLOB="**/persona_detail/*_character.*"
CONCURRENCY=4
JUDGE_TIMEOUT=120
RETRY=2
MAX_FILES=0
INDEX_RANGE=""
RESUME=true
TARGET_CHARACTER_ID=""

if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Missing CONFIG_FILE."
  exit 1
fi

run_eval() {
  local input_dir="$1"
  echo "Running evaluation: input_dir=${input_dir} output_dir=${OUTPUT_DIR}"

  local cmd=(
    python -u evaluate_arena.py
    --config "${CONFIG_FILE}"
    --input_dir "${input_dir}"
    --output_dir "${OUTPUT_DIR}"
    --glob "${GLOB}"
    --concurrency "${CONCURRENCY}"
    --judge_timeout "${JUDGE_TIMEOUT}"
    --retry "${RETRY}"
  )

  if [[ "${MAX_FILES}" != "0" ]]; then
    cmd+=(--max_files "${MAX_FILES}")
  fi
  if [[ -n "${INDEX_RANGE}" ]]; then
    cmd+=(--index_range "${INDEX_RANGE}")
  fi
  if [[ "${RESUME}" == "true" ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "${TARGET_CHARACTER_ID}" ]]; then
    cmd+=(--target_character_id "${TARGET_CHARACTER_ID}")
  fi

  "${cmd[@]}"
}

if [[ -n "${TITLES_FILE}" ]]; then
  if [[ ! -f "${TITLES_FILE}" ]]; then
    echo "TITLES_FILE not found: ${TITLES_FILE}"
    exit 1
  fi
  mapfile -t TITLES < <(rg -v "^\s*$" "${TITLES_FILE}" || true)
fi

if [[ ${#TITLES[@]} -gt 0 ]]; then
  for t in "${TITLES[@]}"; do
    run_eval "${INPUT_ROOT}/${t}"
  done
elif [[ -n "${TITLE}" ]]; then
  run_eval "${INPUT_ROOT}/${TITLE}"
else
  run_eval "${INPUT_ROOT}"
fi
