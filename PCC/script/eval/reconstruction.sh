export CUDA_VISIBLE_DEVICES=0
DATA_PATH=GPT2-Large-Llama3-8B-fineweb-256-5Btokens
COMPRESS_MODEL_PATH=Stage1-PCC-Lite-4x
CONVERTER_MODEL_PATH=Stage1-PCC-Lite-4x
LLM_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct
COMPRESS_RATIO=4
python -m experience.reconstruction.evaluate_ae  \
    --data_path ${DATA_PATH} \
    --compress_model_path ${COMPRESS_MODEL_PATH} \
    --converter_model_path ${CONVERTER_MODEL_PATH} \
    --decoder_model ${LLM_MODEL_PATH} \
    --compress_ratio ${COMPRESS_RATIO} \
    --write True \
    --segment_length 256