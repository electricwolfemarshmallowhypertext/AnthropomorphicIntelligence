#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Ego4D GoalStep
WORKERS=8

python main.py \
    --data_name ego4d_goalstep \
    --func load_dataset \
    --json_path dataset/ego4d_goalstep/goalstep_train.json \
    --system_prompt prompt/ego4d_goalstep/new_polish_system_prompt.txt \
    --workers $WORKERS

python main.py \
    --data_name ego4d_goalstep \
    --func load_dataset \
    --json_path dataset/ego4d_goalstep/goalstep_val.json \
    --system_prompt prompt/ego4d_goalstep/new_polish_system_prompt.txt \
    --workers $WORKERS

python main.py \
    --data_name ego4d_goalstep \
    --func post_process \
    --json_path dataset/ego4d_goalstep/ego4d_goalstep_train_data_polished.json \
    --system_prompt prompt/ego4d_goalstep/repolish_system_prompt.txt \
    --workers $WORKERS

python main.py \
    --data_name ego4d_goalstep \
    --func post_process \
    --json_path dataset/ego4d_goalstep/ego4d_goalstep_val_data_polished.json \
    --system_prompt prompt/ego4d_goalstep/repolish_system_prompt.txt \
    --workers $WORKERS
