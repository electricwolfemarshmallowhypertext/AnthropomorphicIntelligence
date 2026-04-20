#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# LOL
WORKERS=16
# 2025MSI_AL_vs_FLY

python main.py \
    --data_name lol \
    --func extract_audio \
    --video_path dataset/video/LOL/2025MSI_AL_vs_FLY \
    --output_dir dataset/audio/LOL/2025MSI_AL_vs_FLY \
    --workers $WORKERS

python main.py \
    --data_name lol \
    --func asr \
    --audio_path dataset/audio/LOL/2025MSI_AL_vs_FLY \
    --save_asr_dir dataset/asr/LOL/2025MSI_AL_vs_FLY \
    --language zh \
    --min_speakers 2 \
    --max_speakers 2

python main.py \
    --data_name lol \
    --func merge_asr_punctuation \
    --asr_path dataset/asr/LOL/2025MSI_AL_vs_FLY \
    --save_asr_dir dataset/asr/LOL/2025MSI_AL_vs_FLY_merged \
    --workers $WORKERS

python main.py \
    --data_name lol \
    --func polish \
    --asr_path dataset/asr/LOL/2025MSI_AL_vs_FLY_merged \
    --save_polish_dir dataset/polish/LOL/2025MSI_AL_vs_FLY \
    --system_prompt prompt/lol/polish/polish_system_prompt_al_fly.txt \
    --workers $WORKERS

python main.py \
    --data_name lol \
    --func extract_role \
    --polish_dir dataset/polish/LOL/2025MSI_AL_vs_FLY/final \
    --system_prompt prompt/extract_persona/role_analyse_system_prompt.txt \
    --workers $WORKERS
