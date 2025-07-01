 
 


## cd to the folder of this script
cd "$(dirname "$0")"

MODEL=Qwen/Qwen2.5-7B-Instruct
## DATA can be  Amazon, Persona, Blog
DATA=Blog
echo $MODEL
echo $DATA 
python client_eval.py  --dataset=$DATA  --llm=$MODEL   --port=8014 

