conda create -n proactvl python=3.11 -y
conda activate proactvl

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install transformers==4.57.1
pip install qwen-omni-utils==0.0.8
pip install qwen-vl-utils==0.0.14
pip install ipykernel
pip install peft==0.17.1

# train
pip install deepspeed==0.18.2
pip install wandb

conda install "ffmpeg<8" -c conda-forge -y
pip install torchcodec==0.6 --index-url=https://download.pytorch.org/whl/cu128

pip install -U "huggingface_hub[cli]"

# audio generation
pip install kokoro
# web server
pip install uvicorn
pip install fastapi
pip install uvicorn[standard]
# test
pip install datasets==4.4.1
pip install openai==2.7.2
pip install azure-identity==1.25.1
# debug
pip install debugpy

# flash attention for cuda 12.*
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install ./flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
rm ./flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# cuda12.8
conda install -c nvidia -c conda-forge cuda-toolkit=12.8 -y

# data pipeline
pip install ffmpeg-python
