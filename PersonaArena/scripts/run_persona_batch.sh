#!/usr/bin/env bash
set -euo pipefail

PERSONA_FILE="${PERSONA_FILE:-persona_data/1000_persona.en.jsonl}"
LOG_DIR="${LOG_DIR:-output/log}"

# test
INDICES=(19)
# INDICES=(6 19 29 39 49 59 69 78 89 96)
CONFIGS=(
  config/play_qwen3_32b.yaml
  # config/play_mistral_small3.2.yaml
  # config/play_phi4.yaml
  # config/play_llama3.1_8b.yaml
)

mkdir -p output/log

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    exit 1
  fi
  cfg_base="$(basename "$cfg" .yaml)"
  for idx in "${INDICES[@]}"; do
    log_path="batch_${cfg_base}_p${idx}.log"
    python -u simulator.py \
      --config_file "$cfg" \
      --persona_jsonl_path "$PERSONA_FILE" \
      --persona_index "$idx" \
      --autogen_scene_from_persona \
      --log_file "$log_path"
  done
done
