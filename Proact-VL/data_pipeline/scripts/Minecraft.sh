#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Minecraft
WORKERS=16

python main.py \
    --data_name minecraft \
    --func process_vision \
    --video_path dataset/video/MineCraft/survive \
    --vision_model qwen3-vl-plus \
    --system_prompt prompt/minecraft/analyse_system_prompt.txt \
    --output_dir dataset/minecraft/analyse_video \
    --segment_minutes 5 \
    --workers $WORKERS

python main.py \
    --data_name minecraft \
    --func extract_atom_action \
    --json_path dataset/minecraft/analyse_video \
    --output_dir dataset/minecraft/action_infer \
    --vision_model qwen3-vl-plus \
    --system_prompt prompt/minecraft/extract_atom_action_system_prompt.txt \
    --segment_minutes 5 \
    --workers $WORKERS

python main.py \
    --data_name minecraft \
    --func refine_atom_action \
    --json_path dataset/minecraft/action_infer \
    --output_dir dataset/minecraft/refine_atom \
    --system_prompt prompt/minecraft/refine_atom_action_system_prompt.txt \
    --workers $WORKERS

python main.py \
    --data_name minecraft \
    --func post_process \
    --json_path dataset/minecraft/refine_atom \
    --output_dir dataset/minecraft/post_process \
    --system_prompt prompt/minecraft/repolish_system_prompt.txt \
    --workers $WORKERS

python main.py \
    --data_name minecraft \
    --func extract_role \
    --polish_dir dataset/minecraft/post_process/final \
    --system_prompt prompt/extract_persona/role_analyse_system_prompt.txt \
    --workers $WORKERS
