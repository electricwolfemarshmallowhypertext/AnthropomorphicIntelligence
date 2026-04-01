# LiveGamingBenchmark

LiveGamingBenchmark is a dataset construction pipeline for live gaming videos and transcripts.

This README only covers the current way to run the scripts in [`scripts/`](scripts/).

The current repository highlights three representative data categories:

- `Cyberpunk 2077` as an example of `Solo Commentary`
- `LOL` as an example of `Co-Commentary`
- `Minecraft` as an example of `Guidance`

## Requirements

Use Python 3.11.

Install dependencies:

```bash
cd LiveGamingBenchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

You will usually also need:

- `ffmpeg`
- a working PyTorch / CUDA environment for ASR
- a valid Hugging Face token for `whisperx` diarization

Recommended checks:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
ffmpeg -version
```

## Environment Variables

The code loads environment variables from [`client.env`](client.env) at the repository root.

Fill this file before running any script:

```bash
OPENROUTER_API_KEY=
DASHSCOPE_API_KEY=
OPENAI_API_KEY=
UNIAPI_API_KEY=
AZURE_API_KEY=
HF_TOKEN=

POLISH_MODEL_NAME=deepseek/deepseek-v3.2-exp
POLISH_BASE_URL=https://openrouter.ai/api/v1
```

Notes:

- `HF_TOKEN` is required for ASR diarization in `whisperx`.
- `DASHSCOPE_API_KEY` is needed for the Minecraft vision pipeline.
- `OPENROUTER_API_KEY` or another compatible API key is needed for `polish`, `tone_analysis`, and `extract_role`.

## How To Run

All current runnable scripts are in [`scripts/`](scripts/):

- [`scripts/LOL.sh`](scripts/LOL.sh)
- [`scripts/Cyberpunk.sh`](scripts/Cyberpunk.sh)
- [`scripts/Minecraft.sh`](scripts/Minecraft.sh)
- [`scripts/Ego4D.sh`](scripts/Ego4D.sh)

Run them from the repository root:

```bash
cd LiveGamingBenchmark
bash scripts/LOL.sh
bash scripts/Cyberpunk.sh
bash scripts/Minecraft.sh
bash scripts/Ego4D.sh
```

You can also make them executable:

```bash
chmod +x scripts/*.sh
./scripts/LOL.sh
./scripts/Cyberpunk.sh
./scripts/Minecraft.sh
./scripts/Ego4D.sh
```

## What Each Script Does

### LOL

[`scripts/LOL.sh`](scripts/LOL.sh) runs the LOL text pipeline for `dataset/video/LOL/2025MSI_AL_vs_FLY`.

Current steps:

1. `extract_audio`
2. `asr`
3. `merge_asr_punctuation`
4. `polish`
5. `extract_role`

Main outputs:

- `dataset/audio/LOL/2025MSI_AL_vs_FLY`
- `dataset/asr/LOL/2025MSI_AL_vs_FLY`
- `dataset/asr/LOL/2025MSI_AL_vs_FLY_merged`
- `dataset/polish/LOL/2025MSI_AL_vs_FLY`

If you want to run another LOL match, edit the paths inside [`scripts/LOL.sh`](scripts/LOL.sh).

### Cyberpunk 2077

[`scripts/Cyberpunk.sh`](scripts/Cyberpunk.sh) runs the Cyberpunk pipeline for `dataset/video/Cyberpunk_2077/walkthrough`.

Current steps:

1. `extract_audio`
2. `asr`
3. `clean_speaker`
4. `tone_analysis`
5. `polish`
6. `extract_role`

Main outputs:

- `dataset/audio/Cyberpunk_2077`
- `dataset/asr/Cyberpunk_2077`
- `dataset/asr/Cyberpunk_2077/clean_speaker`
- `dataset/asr/Cyberpunk_2077/merge_asr_punctuation_speed_analysis`
- `dataset/polish/Cyberpunk_2077`

### Minecraft

[`scripts/Minecraft.sh`](scripts/Minecraft.sh) runs the Minecraft vision/action pipeline for `dataset/video/MineCraft/survive`.

Current steps:

1. `process_vision`
2. `extrac_atom_action`
3. `refine_atom_action`
4. `post_process`
5. `extract_role`

Main outputs:

- `dataset/minecraft/analyse_video`
- `dataset/minecraft/action_infer`
- `dataset/minecraft/refine_atom`
- `dataset/minecraft/post_process`

### Ego4D

[`scripts/Ego4D.sh`](scripts/Ego4D.sh) runs the Ego4D GoalStep data processing pipeline.

Current steps:

1. `load_dataset` for train
2. `load_dataset` for val
3. `post_process` for train
4. `post_process` for val

Main inputs and outputs:

- `dataset/ego4d_goalstep/goalstep_train.json`
- `dataset/ego4d_goalstep/goalstep_val.json`
- `dataset/ego4d_goalstep/ego4d_goalstep_train_data_polished.json`
- `dataset/ego4d_goalstep/ego4d_goalstep_val_data_polished.json`

## Important Behavior

These scripts are sequential scripts, not argument-driven wrappers.

That means:

- running a script will execute every command currently written in that file
- if you only want one step, comment out the other commands first
- if you want different input/output paths, edit the script directly

## Troubleshooting

### `HF_TOKEN` missing

If ASR fails during diarization, check that `HF_TOKEN` is set in [`client.env`](client.env).

### API errors during `polish`, `tone_analysis`, or `extract_role`

Check:

- the API keys in [`client.env`](client.env)
- `POLISH_MODEL_NAME`
- `POLISH_BASE_URL`

### `ffmpeg` not found

Install `ffmpeg` and confirm:

```bash
ffmpeg -version
```

### GPU / CUDA issues

If `whisperx` fails to load or runs too slowly, verify your CUDA / PyTorch environment first.
