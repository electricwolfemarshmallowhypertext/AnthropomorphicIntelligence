#!/bin/bash
set -e

huggingface-cli download \
  --repo-type dataset \
  bmbgsj/AIGC-text-bank \
  --local-dir data/aigc-text-bank

python -m radar.split_data \
  --num_classes 2 \
  --human_data "data/aigc-text-bank/human_data/human_data_sample.jsonl" \
  --ai_data "data/aigc-text-bank/ai_native/ai_data" \
  --polish_data "data/aigc-text-bank/ai_polish" \
  --output_dir "data/raw" \
  --target_per_class 9000