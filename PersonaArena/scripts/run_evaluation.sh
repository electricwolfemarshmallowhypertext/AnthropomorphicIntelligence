#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="config/evaluate.yaml"
# TITLE maps to the subfolder name under output/record/<TITLE>/...
TITLE="autogen_scene_xx_xx"
# eg:TITLE="autogen_scene_Kyle_Adams_q3-32-N"
# Batch titles; if non-empty, these override TITLE.
TITLES=(
  # "autogen_scene_Henry_Long_q3-32-N"
)
# Optional file with one title per line (non-empty lines only).
TITLES_FILE=""
# These must match the <narrator>_<character> prefix in the record filename.
NARRATOR_LLM="narrator_name"
CHARACTER_LLM="character_name"

if [[ -z "$CONFIG_FILE" || -z "$NARRATOR_LLM" || -z "$CHARACTER_LLM" ]]; then
  echo "Missing evaluation inputs."
  echo "Set CONFIG_FILE/TITLE/NARRATOR_LLM/CHARACTER_LLM in the script and try again."
  exit 1
fi
if [[ ${#TITLES[@]} -eq 0 && -z "$TITLE" && -z "$TITLES_FILE" ]]; then
  echo "Missing evaluation TITLE."
  echo "Set TITLE, populate TITLES, or set TITLES_FILE in the script and try again."
  exit 1
fi

run_eval() {
  local title="$1"
  echo "Running evaluation for TITLE=$title"
  python -u quick_start_arena.py \
    --config_file "$CONFIG_FILE" \
    --title "$title" \
    --narrator_llm "$NARRATOR_LLM" \
    --character_llm "$CHARACTER_LLM"
}

if [[ -n "$TITLES_FILE" ]]; then
  if [[ ! -f "$TITLES_FILE" ]]; then
    echo "TITLES_FILE not found: $TITLES_FILE"
    exit 1
  fi
  mapfile -t TITLES < <(rg -v "^\s*$" "$TITLES_FILE" || true)
fi

if [[ ${#TITLES[@]} -gt 0 ]]; then
  for t in "${TITLES[@]}"; do
    run_eval "$t"
  done
else
  run_eval "$TITLE"
fi
