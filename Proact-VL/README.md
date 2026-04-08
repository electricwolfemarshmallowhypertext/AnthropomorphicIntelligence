# Proact-VL: A Proactive VideoLLM for Real-Time AI Companions

<a href="https://proact-vl.github.io" target="_blank"><img alt="Homepage" src="https://img.shields.io/badge/🌍 Homepage-d35400?color=d35400" /></a>
<a href="https://arxiv.org/abs/2603.03447" target="_blank"><img alt="Paper" src="https://img.shields.io/badge/📄 Paper-28a745?color=28a745" /></a>
<!-- <a href="" target="_blank"><img alt="Checkpoint" src="https://img.shields.io/badge/🤗 Model-2980b9?color=2980b9" /></a>
<a href="" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Dataset-8e44ad?color=8e44ad" /></a> -->
![Proact-VL](asset/proact-vl.jpg)

## TLDR
We provide Proact-VL,  a general framework that shapes multimodal language models into proactive, real-time interactive agents capable of human-like environment perception
and interaction.


## Key Features

- 🎮 **Real-Time Processing**: Handles infinite video streams with low latency
- 🚀 **Proactive Understanding**: Goes beyond reactive responses to provide contextual insights
- 💬 **Multi-Scenario Application**: Supports single-speaker, multi-speaker, and guidance commentary scenarios
- 🔧 **Flexible Architecture**: Built on multiple backbone models (Qwen2-VL, chenjoya/LiveCC-7B-Base, Qwen2.5-VL, Qwen3-VL)
- 📊 **Comprehensive Evaluation**: Includes gaming scenario evaluation with LLM-based judging

## 📢 News
- **[2026.03.16]** 🎉 Proact-VL code released!

## TODO List
- [ ] Release the model zoo (pretrained checkpoints).
- [ ] Release the training dataset and training scripts.
- [ ] Release the test dataset and evaluation scripts.


## Installation
### Conda Environment Setup
Environment for basic usage.
```
conda create -n proactvl python=3.11 -y
conda activate proactvl
sh script/env/prepare_env.sh
```

## Quick Start
1) Solo commentary, co-commentary, and user guidance scenarios

We provide a quick inference script in `quickstart.py` which support SOLO commentary, co-commentary and user guidance scenario.


2) multi-assistant commentary

Another interesting application is to initialize multiple assistants and let them converse with each other. We provide a simple code in `quickstart_multi_assistant.py`.

## Demo
![Demo](asset/demo.png)

Lauch the server, only support qwen2vl, qwen2.5vl, livecc-base model. Set the --port parameter to an unused port. For better presentation, we use [kokoro](https://github.com/hexgrad/kokoro) for audio generation.
```
python -m proactvl.app.cli
```
We recommend include the following content in the system prompts:
<details>
<summary>General/Default System Prompt</summary>
You are a helpful assistant. Provide comprehensive and accurate responses to the user based on the context provided.
</details>

<details>
<summary>Solo Commentary</summary>
Your role is to independently analyze and narrate the game, delivering insightful, engaging, and natural commentary just like a human expert. Focus on key plays, tactics, player actions, and exciting moments to keep viewers informed and entertained. It is not necessary to speak continuously—during uneventful or transitional parts of the match, you may remain silent. Always maintain a lively yet professional tone, and adapt your commentary to the real-time action shown in the video.
</details>

<details>
<summary>Co-Commentary</summary>
Working alongside a human co-caster in a live broadcasting scenario, your role is to analyze, interpret, and explain the in-game action, highlight exciting plays, and engage viewers with insightful and entertaining commentary. You should respond naturally to your co-caster’s remarks, support their analysis, or introduce new perspectives, just like a professional esports commentator team. Always keep your tone lively, professional, and audience-friendly. Rely on real-time video and your co-caster’s speech to guide your commentary, and make sure your responses are timely, relevant, and complementary to your co-caster.
</details>

<details>
<summary>Guidance</summary>
When a player asks a question, use the real-time game visuals to provide clear, step-by-step guidance to help the player accomplish their goal. Only respond when the player asks for help or completes current sub-action and prepare for the next; otherwise, remain silent. Your instructions should be concise, accurate, and easy for players to follow. Continue to guide the player until the task is completed.
</details>

## Evaluation

### Live Gaming Benchmark & Live gaming Benchmark-Streaming
To evaluate on our benchmark, please follow the steps below.

First download our dataset.

**LiveGamingBenchmark**
- Use our model (or your custom model) to generate the output file.
- Compute the metrics (LLM Score, CC, F1, Time-Diff, and PAUC) following the instructions below.

**LiveGamingBenchmark-Streaming**
- Generate the inference results.
- Run our script to slice the results into segments (one segment every 30 seconds).
- Compute the metrics on the segmented results.

#### Download dataset
Download our Live Gaming Benchmark (to be released), and get the annotations and other useful files (to be released).  
Download the Ego4D Goal-Step dataset following the instructions in this GitHub repository [Ego4D Goal-Step](https://github.com/facebookresearch/ego4d-goalstep), and place the video directory under `DATA/`.  

The expected data directory structure is shown below:
```
DATA
├── game_commentary
│   ├── LOL
|   |   ├── videos
│   ├── Minecraft
|   |   ├── videos
│   ├── Cyberpunk_2077
│   ├── CSGO
│   ├── Black_Myth_Wukong
│   ├── Baldurs_Gate_3
│   ├── Starcraft2
│   ├── Streetfighter6
│   ├── Tears_of_the_Kingdom
│   ├── Yu_Gi_Oh
│   ├── Elden_Ring
├── ego4d
│   ├── v2
|   |   ├── annotations
|   |   ├── full_scale
│   ├── ego4d.json
├── anns
│   ├── all_in_one.jsonl
│   ├── ego4d.jsonl
│   ├── main_games.jsonl
│   ├── streaming_games.jsonl
│   ├── wukong.jsonl
```
`all_in_one.jsonl` merges annotations in `main_games.jsonl`, `ego4d.jsonl` and `wukong.jsonl`.
`streaming_games.jsonl` is used for evaluate the streaming ability.

#### Infer LiveGamingBenchmark
Set video_dir to [LOCAL_DATA_DIR].
```
# livecc-base
CKPT_PATH='to_be_released'
BASE_MODEL='chenjoya/LiveCC-7B-Base'
MODEL_ID='proactvl_base_liveccbase'
python -m evaluation.gaming.distributed_generate_gaming --model_name_or_path ${BASE_MODEL} \
    --ckpt_path ${CKPT_PATH} --num_workers ${NGPUS} --model_id ${MODEL_ID} \
    --state_threshold 0.3 \
    --dataset_name [TO_BE_RELEASED] \
    --test_name 'all_in_one' \
    --video_dir $HOME/ds/DATA \
    --output_dir ./results/proactvl \
    --max_kv_tokens 16384
```

#### Infer LiveGamingBenchmark-Streaming
```
# livecc-base
CKPT_PATH='to_be_released'
BASE_MODEL='chenjoya/LiveCC-7B-Base'
MODEL_ID='proactvl_base_liveccbase'
python -m evaluation.gaming.distributed_generate_gaming --model_name_or_path ${BASE_MODEL} \
    --ckpt_path ${CKPT_PATH} --num_workers $(get_gpu_count) --model_id ${MODEL_ID} \
    --state_threshold 0.3 \
    --dataset_name [TO_BE_RELEASED] \
    --test_name 'streaming_games' \
    --video_dir $HOME/ds/DATA \
    --output_dir ./results/proactvl/streaming \
    --max_kv_tokens 16384

python -m evaluation.gaming.label_streaming2standard \
    --ann_path ./results/proactvl/streaming_${MODEL_ID}_30_16384.jsonl \
    --save_path ./results/proactvl/streaming_${MODEL_ID}_30_16384_standard.jsonl --mode pred
```
#### Judge
First, replace the code you use to initialize GPT, and configure envrioment variable. 

```
export OPENAI_AUTH_MODE=api_key
export OPENAI_API_KEY='your_api_key'
export OPENAI_API_BASE='your_api_base'
```
Then run as follow:

**CC(win rate)** 
```
# live gaming benchmark
python -m evaluation.gaming.llm_judge --model_id liveccbase_30_16384 \
    --prediction_jsonl results/proactvl/liveccbase_30_16384.jsonl \
    --num_workers 16 \
    --baseline_id gemini2.5-pro --baseline_jsonl results/baseline/captions/gemini-2.5-pro.jsonl \
    --asr_jsonl results/anns/all_in_one.jsonl \
    --output_dir results/evaluation/cc/proactvl
```
For Live Gaming Benchmark-Streaming:
```
# live gaming benchmark-streaming
python -m evaluation.gaming.llm_judge --model_id streaming_${MODEL_ID}_30_16384_standard \
    --prediction_jsonl results/proactvl/streaming_${MODEL_ID}_30_16384_standard.jsonl \
    --num_workers 16 \
    --baseline_id streamingvlm_streaming --baseline_jsonl results/baseline/streaming/StreamingVLM_standard.jsonl \
    --asr_jsonl results/anns/streaming_video_commentary_val_standard.jsonl
```
**LLM Score: LiveU, FinalQ**
```
python -m evaluation.gaming.llm_score --model_id liveccbase_30_16384 \
    --prediction_jsonl results/proactvl/liveccbase_30_16384.jsonl \
    --num_workers 16 \
    --asr_jsonl results/anns/all_in_one.jsonl \
    --output_dir results/evaluation/llm_score/proactvl
```
**F1, Time-Diff**
```
python3 -m evaluation.gaming.f1_timediff \
    results/proactvl/liveccbase_30_16384.jsonl  \
    --reference results/anns/all_in_one_val_proactive.jsonl \
    --output results/evaluation/f1/proactvl/liveccbase_30_16384.json \
    --alpha 0.2 \
    --verbose
```
**PAUC**
```
python -m evaluation.gaming.pauc \
  --func one_step \
  --pred_fname results/proactvl/liveccbase_30_16384.jsonl \
  --reference  results/anns/all_in_one_val_proactive.jsonl \
  --output_fname results/evaluation/pauc/proactvl/liveccbase_30_16384.json \
  --openai_model gpt-5.1_2025-11-13 \
  --concurrency 16 \
  --start_score 0 \
  --judge_limit -1 \
  --resume
```


## Train
### Data Preparation
1) Download training data from huggingface (to be released)

2) Download the Ego4D Goal-Step dataset following the instructions in this GitHub repository [Ego4D Goal-Step](https://github.com/facebookresearch/ego4d-goalstep), and place the video directory under `DATA/`.
3) Download Live-WhisperX-526K(https://huggingface.co/datasets/chenjoya/Live-WhisperX-526K) and place the video directory under `DATA`, we only use the first 32000 sample to finetune the model.

The expected data directory structure is shown below:
```
DATA
├── game_commentary
│   ├── LOL
|   |   ├── videos
│   ├── Minecraft
|   |   ├── videos
│   ├── Cyberpunk_2077
│   ├── CSGO
│   ├── Black_Myth_Wukong
│   ├── Baldurs_Gate_3
│   ├── Starcraft2
│   ├── Streetfighter6
│   ├── Tears_of_the_Kingdom
│   ├── Yu_Gi_Oh
│   ├── Elden_Ring
├── ego4d
│   ├── v2
|   |   ├── annotations
|   |   ├── full_scale
│   ├── ego4d.json
├── live_sft
│   ├── videos
|   |   ├── *.json
|   |   ├── *.mp4
├── anns
│   ├── *_final_train.jsonl
│   ├── *_final_val.jsonl
```
### full parameter fine-tuning
We freeze the visual tower, it takes about 24hours to train 2000 steps using 8*H100 with gradient_accumulation_steps set to 8(batch size=64 in total).
```
N_GPUS=8
GRADIENT_ACC_STEPS=4

RUN_ID=$(date +"%Y%m%d-%H%M%S")
RUN_NAME='proactvl_fulltuning_base_liveccbase'

STAGE="strategy3"
ACTIVE_LAYER=-2

deepspeed --num_gpus=$N_GPUS --master_port 8848 finetune.py \
    --deepspeed ./config/deepspeed_zero2.json \
    --do_train \
    --do_eval \
    --train_dataset_names baldurs_gate_3 csgo cyberpunk_2077 elden_ring lol minecraft starcraft2 streetfighter6 tears_of_the_kingdom yu_gi_oh livecc ego4d \
    --val_dataset_names baldurs_gate_3 csgo cyberpunk_2077 elden_ring lol starcraft2 streetfighter6 tears_of_the_kingdom yu_gi_oh \
    --data_dir_path $HOME/ds/DATA \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --num_train_epochs 2 \
    --max_steps 2000 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --bf16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --eval_accumulation_steps 4 \
    --logging_steps 5 \
    --model_name_or_path "chenjoya/LiveCC-7B-Base" \
    --enable_audio_output False \
    --state_threshold 0.5 \
    --loss_active_scale 0.2 \
    --use_lora False \
    --freeze_audio True \
    --freeze_visual True \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 3 \
    --eval_steps 1 \
    --eval_strategy "steps" \
    --report_to "wandb" \
    --run_name $RUN_NAME \
    --gradient_checkpointing True \
    --finetune_strategy ${STAGE} \
    --label_names labels active_labels \
    --active_layer_id ${ACTIVE_LAYER} \
    --output_dir trainer_output/${RUN_NAME}
```

## Related Projects
- [VideoLLM-online](https://github.com/showlab/videollm-online)
- [StreamMind](https://github.com/xinding-sys/StreamMind)
- [MMDuet](https://github.com/yellow-binary-tree/mmduet)
- [LiveStar](https://github.com/sotayang/LiveStar)
- [LiveCC](https://github.com/showlab/livecc)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM-V)
- [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm/tree/main)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

## Citation
```BibTeX
@article{yan2026proact,
  title={Proact-VL: A Proactive VideoLLM for Real-Time AI Companions},
  author={Yan, Weicai and Dai, Yuhong and Ran, Qi and Li, Haodong and Lin, Wang and Liao, Hao and Xie, Xing and Jin, Tao and Lian, Jianxun},
  journal={arXiv preprint arXiv:2603.03447},
  year={2026}
}
```

## Contact
If you would like early access to the model weights and dataset, or if you have any questions or would like to discuss this work, please contact the authors at [yanweicai@zju.edu.cn](mailto:yanweicai@zju.edu.cn), [broalantaps123@gmail.com](mailto:broalantaps123@gmail.com), or [jianxun.lian@microsoft.com](mailto:jianxun.lian@microsoft.com).