

## cd to the folder of this script
cd "$(dirname "$0")"

MODEL=Qwen/Qwen2.5-7B-Instruct
echo $MODEL 
python server_vllm.py  --model=$MODEL  --port=8014  
  
