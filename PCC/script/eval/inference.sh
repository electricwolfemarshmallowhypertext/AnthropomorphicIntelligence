export CUDA_VISIBLE_DEVICES=0
python inference.py \
    --compress_model Stage2-PCC-Lite-4x \
    --converter_model Stage2-PCC-Lite-4x \
    --decoder_model meta-llama/Meta-Llama-3-8B-Instruct \
    --stage 2 \
    --segment_length 256 \
    --ratio 4 \
    --compressor_gradient_checkpoint False \
    --decoder_gradient_checkpoint False
