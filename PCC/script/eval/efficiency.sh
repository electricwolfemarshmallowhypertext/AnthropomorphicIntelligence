export CUDA_VISIBLE_DEVICES=0

python -m experience.efficiency.evaluate_efficiency  \
    --ratio 4 \
    --batch_size 8 \
    --input_length 1024 \
    --generate_length 32
