# PersonaArena
PersonaArena is a "role-play + evaluation" pipeline: it first runs an interaction (simulation) under a given persona, then uses evaluator models to score the outcome and output readable table metrics.

<p align="center">
  <img src="asset/img/PersonaArena.png" alt="Character Arena Framework" width="100%">
</p>

## What this repo does
- **Generate interaction records (simulation)**: a narrator builds a scene, the main character (the LLM under test) and NPCs interact based on their personas, and the system records full actions and dialogue.
- **Automatic evaluation**: score the main character's performance in each scene on 8 dimensions, producing detailed CSVs and summary CSVs.
- **Model comparison**: switch models and APIs in `config/play_*.yaml` to reproduce or compare results.

## Installation
```shell
pip install -r requirements.txt
```

## Quick start (single simulation)
1. Pick a config file and fill in model + API info (example: `config/play_gpt4o.yaml`).
   - `character_llm` / `narrator_llm` / `npc_llm`: model names or aliases.
   - `*_api_key` / `*_api_base`: API credentials for each model.
   - Supports any OpenAI-compatible endpoint (including self-hosted services).
2. Run a single simulation:
```shell
python -u simulator.py --config_file config/play_gpt4o.yaml --log_file simulation_gpt4o.log
```
3. Outputs:
   - Interaction records: `output/record/<title>/...`
   - Logs: `output/log/simulation/<title>/...`
   - Auto-generated scenes (if autogen enabled): `output/scenes/` or `generated_scenes/`

> "simulation" means: auto-generate a scene from a persona + run a full interaction (actions + dialogue) + record everything.

## Batch simulation (multiple personas / models)
Script: `scripts/run_persona_batch.sh`  
**Note**: the script hard-codes persona indices and config lists (`INDICES`, `CONFIGS`). Edit the script before running.

```shell
chmod +x scripts/run_persona_batch.sh
scripts/run_persona_batch.sh
```

Common edits:
- `PERSONA_FILE`: persona jsonl path (default: `persona_data/1000_persona.en.jsonl`)
- `INDICES`: persona indices to run
- `CONFIGS`: model config list to run

If you need a proxy on first run (to fetch the HuggingFace embedding model):
```shell
export http_proxy=xxx; export https_proxy=xxx
```

## Evaluation
After simulations finish, run the evaluation script to output metrics.

Script: `scripts/run_evaluation.sh`  
**Note**: you must set `TITLE` / `NARRATOR_LLM` / `CHARACTER_LLM`, and they must match the record file names.

```shell
chmod +x scripts/run_evaluation.sh
scripts/run_evaluation.sh
```

Outputs:
- Details: `output/evaluation/detail/<title>/<character>_<narrator>_character_evaluation_detail.csv`
- Summary: `output/evaluation/multi/<title>/<narrator>_<scene_id>_character_evaluation_avg.csv`

Metrics (8 dimensions):
1. Knowledge Accuracy
2. Emotional Expression
3. Personality Traits
4. Behavioral Accuracy
5. Immersion
6. Adaptability
7. Behavioral Coherence
8. Interaction Richness

### Result table (see the final CSV at `output/evaluation/multi/<title>/<narrator>_<scene_id>_character_evaluation_avg.csv`)
```text
| Title | Judger | Narrator | Model | SceneID | Round | Knowledge Accuracy | Emotional Expression | Personality Traits | Behavioral Accuracy | Immersion | Adaptability | Behavioral Coherence | Interaction Richness | Average | DebateCount |
| ...   | ...    | ...      | ...   | ...     | ...   | ...                | ...                  | ...               | ...                | ...       | ...         | ...                 | ...                 | ...     | ...        |
```
