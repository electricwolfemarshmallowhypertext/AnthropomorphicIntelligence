#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# =========================
# User Config (edit here)
# =========================

# 1) Persona data file path
PERSONA_FILE="persona_data/1000_persona.en.jsonl"

# 2) Persona indices to run: supports "0-12", "0,2,5", "0-3,8,10-12"
PERSONA_INDICES="100"

# 3) Parallelism (controlled by simulator.py --experiment_parallelism)
EXPERIMENT_PARALLELISM=1

# 4) Skip completed tasks: 1=skip, 0=rerun
SKIP_EXISTING=1

# 5) Model configs to run
CONFIGS=(
  # config/play_gpt-5.1.yaml
  # config/play.yaml
  config/play_qwen3_14b.yaml
)

# =========================
# Run
# =========================

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    exit 1
  fi

  echo "[RUN] cfg=$cfg persona_indices=$PERSONA_INDICES parallelism=$EXPERIMENT_PARALLELISM"

  cmd=(
    python -u simulator.py
    -c "$cfg"
    --persona_jsonl_path "$PERSONA_FILE"
    --autogen_scene_from_persona
    --persona_indices "$PERSONA_INDICES"
    --experiment_parallelism "$EXPERIMENT_PARALLELISM"
  )

  if ((SKIP_EXISTING == 1)); then
    cmd+=(--skip_existing)
  fi

  "${cmd[@]}"
done

echo "All configs finished."
