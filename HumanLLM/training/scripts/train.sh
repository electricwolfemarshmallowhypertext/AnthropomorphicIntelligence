export TOKENIZERS_PARALLELISM=false;
export WANDB_RUN_ID=$(date +"%Y%m%d_%H%M%S")

cd ~/LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_8b_full_sft.yaml > ~/LLaMA-Factory/output/train_qwen3_8b_full_sft.log 2>&1