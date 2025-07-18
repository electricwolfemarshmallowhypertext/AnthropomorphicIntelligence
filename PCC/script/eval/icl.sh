export CUDA_VISIBLE_DEVICES=0
dataset=wsc

COMPRESS_MODEL_PATH=PCC-Large-Encoder-Llama3-8B-Instruct
CONVERTER_MODEL_PATH=PCC-Large-Encoder-Llama3-8B-Instruct
ADAPTER_MODEL_PATH=Stage2-PCC-Large-4x
LLM_MODEL_PATH=meta-llama/Meta-Llama-3-8B-Instruct

COMPRESS_RATIO=4

num_softprompt_demonstrations=30
num_plaintext_demonstrations=0
seed=10
use_calib=True


python -m experience.icl.evaluate_icl \
    --dataset ${dataset} \
    --num_softprompt_demonstrations ${num_softprompt_demonstrations} \
    --num_plaintext_demonstrations ${num_plaintext_demonstrations} \
    --seed ${seed} \
    --use_calibration \
    --compress_model ${COMPRESS_MODEL_PATH} \
    --converter_model ${CONVERTER_MODEL_PATH} \
    --decoder_model ${LLM_MODEL_PATH} \
    --adapter_model ${ADAPTER_MODEL_PATH} \
    --segment_length 256 \
    --compress_ratio ${COMPRESS_RATIO} \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --compressor_gradient_checkpoint False \
    --decoder_gradient_checkpoint False
    