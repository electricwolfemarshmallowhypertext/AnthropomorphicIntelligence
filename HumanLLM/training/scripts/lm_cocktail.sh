cd ~/AnthropomorphicIntelligence/Human_LLM//training

python lm_cocktail.py \
    --model_names_or_paths "xxx/Llama-3.1-8B-Instruct_lr5e-6_seq8192_batch64_zero1_3epoch,Llama-3.1-8B-Instruct" \
    --model_type decoder \
    --weights 0.5,0.5 \
    --output_path xxx/Llama-3.1-8B-Instruct_lr5e-6_seq8192_batch64_zero1_3epoch/merged_model
