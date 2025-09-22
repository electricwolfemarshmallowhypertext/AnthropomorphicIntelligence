export CUDA_VISIBLE_DEVICES=0

#---------------PCC Lite Configuration---------------#
COMPRESS_MODEL_PATH=Stage2-PCC-Lite-4x
CONVERTER_MODEL_PATH=Stage2-PCC-Lite-4x
LLM_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
COMPRESS_RATIO=4

python -m experience.qa.evaluate_qa  \
    --dataset nq \
    --compress_model_path ${COMPRESS_MODEL_PATH} \
    --converter_model_path ${CONVERTER_MODEL_PATH} \
    --decoder_model ${LLM_MODEL_PATH} \
    --compress_ratio ${COMPRESS_RATIO} \
    --write True \
    --segment_length 256


#---------------PCC Large Configuration---------------#
# COMPRESS_MODEL_PATH=PCC-Large-Encoder-Llama3-8B-Instruct
# ADAPTER_MODEL_PATH=Stage2-PCC-Large-4x
# CONVERTER_MODEL_PATH=Stage2-PCC-Large-4x
# LLM_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
# COMPRESS_RATIO=4

# python -m experience.qa.evaluate_qa  \
#     --dataset nq \
#     --use_lora True \
#     --adapter_model ${ADAPTER_MODEL_PATH} \
#     --compress_model_path ${COMPRESS_MODEL_PATH} \
#     --converter_model_path ${CONVERTER_MODEL_PATH} \
#     --decoder_model ${LLM_MODEL_PATH} \
#     --compress_ratio ${COMPRESS_RATIO} \
#     --write True \
#     --segment_length 256