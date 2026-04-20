#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Cyberpunk_2077
WORKERS=16
python main.py \
    --data_name cyberpunk \
    --func extract_audio \
    --video_path dataset/video/Cyberpunk_2077/walkthrough \
    --output_dir dataset/audio/Cyberpunk_2077 \
    --workers $WORKERS

python main.py \
    --data_name cyberpunk \
    --func asr \
    --audio_path dataset/audio/Cyberpunk_2077 \
    --save_asr_dir dataset/asr/Cyberpunk_2077 \
    --language en \
    --min_speakers 1 \
    --max_speakers 10 \
    --workers $WORKERS

python main.py \
    --data_name cyberpunk \
    --func clean_speaker \
    --json_path dataset/asr/Cyberpunk_2077 \
    --output_dir dataset/asr/Cyberpunk_2077/clean_speaker \
    --workers $WORKERS

python main.py \
    --data_name cyberpunk \
    --func tone_analysis \
    --asr_path dataset/asr/Cyberpunk_2077/clean_speaker \
    --video_path dataset/video/Cyberpunk_2077/walkthrough \
    --save_asr_dir dataset/asr/Cyberpunk_2077/merge_asr_punctuation_speed_analysis \
    --speed_threshold 1.5 \
    --system_prompt prompt/tone_analysis_system_prompt.txt \
    --workers $WORKERS

python main.py \
    --data_name cyberpunk \
    --func polish \
    --asr_path dataset/asr/Cyberpunk_2077/clean_speaker \
    --save_polish_dir dataset/polish/Cyberpunk_2077 \
    --system_prompt prompt/cyberpunk_2077/polish_system_prompt.txt \
    --workers $WORKERS

python main.py \
    --data_name cyberpunk \
    --func extract_role \
    --polish_dir dataset/polish/Cyberpunk_2077/final \
    --system_prompt prompt/extract_persona/role_analyse_system_prompt.txt \
    --workers $WORKERS
