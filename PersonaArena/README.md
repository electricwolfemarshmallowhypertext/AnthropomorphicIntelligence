# PersonaArena
The appendix of the paper can be found in the file [PersonaArena_Appendix.pdf](PersonaArena_Appendix.pdf).

<p align="center">
  <img src="asset/img/PersonaArena.png" alt="Character Arena Framework" width="100%">
</p>

## Installation

To install PersonaArena, follow these steps:

1. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

## Quick Start

Sample personas, scenes, and logs are stored under `persona_data`, `out/scenes`, and `output/`. Adjust the configuration that best matches your setup under `config/` (for example `config/play_gpt4o.yaml`).

Run a simulation with:

```shell
python -u simulator.py --config_file config/play_gpt4o.yaml --log_file simulation_gpt4o.log
```

## Run scripts

```shell
chmod +x scripts/run_persona_batch.sh
scripts/run_persona_batch.sh
```

The first run may need a proxy because it fetches:
https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/./modules.json

```shell
export http_proxy=xxx; export https_proxy=xxx
```

Common optional parameters:
- PERSONA_FILE: custom persona jsonl path
- PERSONA_INDICES: comma-separated persona indices (e.g., 0,1,2)
- CONFIG_FILES: comma-separated config list (e.g., config/play_qwen3_32b.yaml,config/play_gpt4o.yaml)

- `--config_file` points to a YAML file describing narrator, character, and optional NPC models.
- Provide API credentials per agent in the config (`character_api_key`, `character_api_base`, `npc_api_key`...). PersonaArena accepts any OpenAI-compatible endpoint, including self-hosted services.
- `npc_human_control`: when `true`, route NPC replies through manual input; `false` keeps NPCs fully model-driven.
- If `--log_file` is omitted, logs are written to `output/log/simulation/` automatically.

Simulation transcripts and agent memories are saved to `output/record/` according to the paths declared in the configuration file.

## Evaluation

Once simulations have produced records, run the evaluation driver to score a protagonist's performance during a complete interaction :

```shell
chmod +x scripts/run_evaluation.sh
scripts/run_evaluation.sh
```

- Populate `config/evaluate.yaml` with the `title`, `scene_id`, `judges`, and the `narrator_llm` / `character_llm` pairs you wish to assess. Each judge entry supports separate `api_key`, `api_base`, and provider settings.
- The evaluator reads interaction logs from `output/record/` and writes per-scene reports to `output/evaluation/detail/{title}/{character}_{narrator}_character_evaluation_detail.csv`.
- Aggregated summaries are stored in `output/evaluation/multi/{title}/`, and debate metadata (if enabled) is captured alongside the numeric metrics.

Optional parameters (environment variables):
- RECORD_FILE: directly specify a record file under `output/record`
- TITLE / NARRATOR_LLM / CHARACTER_LLM: manually set when auto-detection fails
