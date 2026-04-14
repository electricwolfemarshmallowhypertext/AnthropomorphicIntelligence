# PersonaArena

PersonaArena is a "role-playing + evaluation" workflow: it first runs interactions (simulation) under a given persona, then uses an evaluation model to score the results and outputs easy-to-understand tabular metrics.

<p align="center">

<img src="asset/img/PersonaArena.png" alt="PersonaArena Framework" width="100%">

</p>

## What This Repository Does

- **Generate interaction records (simulation)**: The narrator constructs scenarios, the protagonist (LLM under test) and NPCs interact according to their respective personas, and the system records complete actions and dialogues.

- **Automatic evaluation**: Scores the protagonist's performance in each scenario across 8 dimensions, generating detailed CSV files and summary CSV files.

- **Model comparison**: Switch models and APIs in `config/play.yaml` to reproduce or compare results.

## Installation

```shell

pip install -r requirements.txt

```

## Offline / Self-hosted LLM API Configuration

PersonaArena supports any OpenAI-compatible endpoint. The 4 methods below are common examples: `Ollama`, `vLLM`, `LiteLLM`, `SGLang`.

### Minimal rules

- `api_base` format: `http://<host>:<port>/v1` (local or remote server both work)
- Local example: `http://localhost:<port>/v1`
- Remote server example: `http://<server-ip>:<port>/v1`
- `api_key` is your self-defined token (example: `my-vllm-key`)
- API check command:
```bash
curl http://<host>:<port>/v1/models
curl http://<host>:<port>/v1/models -H "Authorization: Bearer <api_key>"
```

### Model choice (important)

- Prefer non-thinking/chat models for simulation stability.
- If you must use reasoning models, set this in `config/play.yaml`:
```yaml
model_kwargs:
  extra_body:
    # Disable chain-of-thought style "thinking" output if backend supports it
    enable_thinking: false
    chat_template_kwargs:
      # For compatible chat templates
      enable_thinking: false
# Print verbose logs
verbose: true
```

### 1) Ollama (`qwen3-32b`)

```bash
ollama pull qwen3:32b
cat > Modelfile <<'EOF'
FROM qwen3:32b
EOF
ollama create qwen3-32b -f Modelfile
ollama serve
curl http://localhost:11434/v1/models
curl http://<server-ip>:11434/v1/models
```

`config/play.yaml`:
```yaml
character_llm: qwen3-32b
character_provider: openai
character_api_key: empty
character_api_base: http://localhost:11434/v1   # local
# character_api_base: http://<server-ip>:11434/v1   # remote server
```

### 2) vLLM (`qwen3-32b`)

```bash
export VLLM_KEY="my-vllm-key"
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --host 0.0.0.0 --port 8000 --api-key "$VLLM_KEY"
curl http://localhost:8000/v1/models -H "Authorization: Bearer $VLLM_KEY"
curl http://<server-ip>:8000/v1/models -H "Authorization: Bearer $VLLM_KEY"
```

`config/play.yaml`:
```yaml
character_llm: qwen3-32b
character_provider: openai
character_api_key: my-vllm-key
character_api_base: http://localhost:8000/v1   # local
# character_api_base: http://<server-ip>:8000/v1   # remote server
```

### 3) LiteLLM (`qwen3-32b`)

`litellm.config.yaml`:
```yaml
model_list:
  - model_name: qwen3-32b
    litellm_params:
      model: openai/Qwen/Qwen3-32B
      api_base: http://localhost:8000/v1  # upstream local vLLM
      # api_base: http://<server-ip>:8000/v1  # upstream remote vLLM
      api_key: my-vllm-key
```

```bash
export LITELLM_MASTER_KEY="my-litellm-gateway-key"
litellm --config litellm.config.yaml --host 0.0.0.0 --port 4000
curl http://localhost:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
curl http://<server-ip>:4000/v1/models -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

`config/play.yaml`:
```yaml
character_llm: qwen3-32b
character_provider: openai
character_api_key: my-litellm-gateway-key
character_api_base: http://localhost:4000/v1   # local
# character_api_base: http://<server-ip>:4000/v1   # remote server
```

### 4) SGLang (`qwen3-32b`)

```bash
export SGLANG_KEY="my-sglang-key"
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --served-model-name qwen3-32b \
  --host 0.0.0.0 --port 30000 --api-key "$SGLANG_KEY"
curl http://localhost:30000/v1/models -H "Authorization: Bearer $SGLANG_KEY"
curl http://<server-ip>:30000/v1/models -H "Authorization: Bearer $SGLANG_KEY"
```

`config/play.yaml`:
```yaml
character_llm: qwen3-32b
character_provider: openai
character_api_key: my-sglang-key
character_api_base: http://localhost:30000/v1   # local
# character_api_base: http://<server-ip>:30000/v1   # remote server
```


## Quick Start (Single Simulation)

1. Fill in model and API information in the configuration file (for example: `config/play.yaml`).

- `character_llm` / `narrator_llm` / `npc_llm`: model name or alias (including locally deployed model names).

- `*_api_key` / `*_api_base`: API credentials for each model.

- Detailed parameter descriptions are available in the comments of `config/play.yaml`.

- Supports any OpenAI-compatible endpoint (including self-hosted services).

2. Run a single simulation:

```shell

python -u simulator.py --config_file config/play.yaml --log_file simulation.log

```
3. Outputs:

- Interaction records: `output/record/<title>/...`

- Logs: `output/log/simulation/<title>/...`

- Auto-generated scenarios (if auto-generation is enabled): `output/scenes/` or `generated_scenes/`

> "Simulation" means: automatically generating scenarios based on persona + running complete interactions (actions + dialogue) + recording all content.

## Batch Simulation (Multiple Personas/Models)

Script: `scripts/run_persona_batch.sh`

**Note**: This script hardcodes persona indices and configuration lists (`PERSONA_INDICES`, `CONFIGS`). Please edit the script and prepare related configuration files before running.

```shell

chmod +x scripts/run_persona_batch.sh

scripts/run_persona_batch.sh

```

### Configuration Parameters

Edit the following parameters in `scripts/run_persona_batch.sh`:

#### 1. Persona Configuration

- **`PERSONA_FILE`**: Path to persona data file (default: `persona_data/1000_persona.en.jsonl`)

- **`PERSONA_INDICES`**: Which personas to run (supports multiple formats)
  - Single index: `"100"`
  - Range: `"0-12"`
  - Multiple: `"0,2,5"`
  - Mixed: `"0-3,8,10-12"`

#### 2. Parallelism & Concurrency

- **`EXPERIMENT_PARALLELISM`**: Number of concurrent personas to simulate
  - `1` = sequential (one at a time)
  - `3` = simulate 3 personas concurrently
  - **Recommended**: Set based on your GPU/CPU resources and model API rate limits

**How it works**:
- Each persona gets its own simulation process
- Multiple personas run in parallel using Python's `concurrent.futures.ThreadPoolExecutor`
- Each persona creates a `.done` marker file in `output/batch_done/` when completed
- If `SKIP_EXISTING=1`, already-completed personas are skipped

#### 3. Execution Control

- **`SKIP_EXISTING`**: Skip completed tasks (1=skip, 0=rerun)
  - When set to `1`, the script checks for `.done` markers and completed logs
  - Useful for resuming interrupted batch runs

- **`CONFIGS`**: Array of model config files to run
  ```bash
  CONFIGS=(
    config/play_qwen3_14b.yaml
    config/play_gpt-4o-mini.yaml
  )
  ```
  - All personas will be simulated for each config
  - Example: 10 personas × 2 configs = 20 total simulations

#### 4. Output Organization

Simulations are organized as:
```
output/
├── record/
│   ├── autogen_scene_<persona_name>_<model>_0/
│   ├── autogen_scene_<persona_name>_<model>_1/
│   └── ...
└── batch_done/
    ├── play_qwen3_14b_p0.done  # Persona 0 completed
    ├── play_qwen3_14b_p1.done  # Persona 1 completed
    └── ...
```

## Evaluation

After simulation is complete, run the evaluation script to output metrics.

Script: `scripts/run_evaluation.sh`

- The script directly runs `evaluate_arena.py` (async evaluator).
- Detailed parameter descriptions are available in the comments of `config/evaluate.yaml`.

```shell

chmod +x scripts/run_evaluation.sh

scripts/run_evaluation.sh

```

### Configuration Parameters

Edit the following parameters in `scripts/run_evaluation.sh`:

#### 1. Basic Configuration

- **`CONFIG_FILE`**: Evaluation config file (default: `config/evaluate.yaml`)
  - Defines judge models, debate settings, and evaluation parameters

- **`INPUT_ROOT`**: Record root directory (default: `output/record`)
  - Where simulation results are stored

- **`OUTPUT_DIR`**: Output root directory (default: `output/evaluation/by_model`)
  - If path ends with `by_model`, results are organized by character model
  - Results: `by_model/<character_llm>/evaluation_results.jsonl`

#### 2. Concurrency & Performance

- **`CONCURRENCY`**: Number of files to evaluate concurrently (default: `4`)
  - `1` = sequential evaluation (safer, slower)
  - `4` = evaluate 4 files simultaneously (recommended)
  - Higher values = faster but more API load

#### 3. File Selection Control

Choose one of the following to select which titles to evaluate:

- **`TITLE`**: Single title (e.g., `"autogen_scene_Anna_Castillo_Qwen3-14B-Q4"`)
  - Evaluates only files under `INPUT_ROOT/<TITLE>/`

- **`TITLES`**: Multiple titles array
  ```bash
  TITLES=(
    "autogen_scene_Anna_Castillo_Qwen3-14B-Q4"
    "autogen_scene_Henry_Long_gpt-4o-mini"
  )
  ```

- **`TITLES_FILE`**: File containing title list (one per line)
  - Useful for large batches
  - Empty lines are ignored

If none are set, evaluates all titles under `INPUT_ROOT`.

- **`GLOB`**: File pattern to match (default: `"**/persona_detail/*_character.*"`)
  - Matches character record files
  - Supports both `.jsonl` and `.jsnol` extensions

### Outputs

#### Main Results (Per Model)
- **JSONL**: `output/evaluation/by_model/<character_llm>/evaluation_results.jsonl`
  - One line per evaluated file
  - Contains all scores, judges, critiques

- **Average CSV**: `output/evaluation/by_model/<character_llm>/evaluation_results_avg.csv`
  - Summary table with average scores per file
  - Easy to compare different models

#### Detail CSVs (Per Scene)
- **Detail**: `output/evaluation/by_model/<character_llm>/detail/<title>/*_evaluation_detail.csv`
  - Per-judge scores and critiques
  - Useful for detailed analysis

#### Organization Example
```
output/evaluation/by_model/
├── Qwen3-14B/
│   ├── evaluation_results.jsonl
│   ├── evaluation_results_avg.csv
│   └── detail/
│       ├── autogen_scene_Anna_Castillo_Qwen3-14B/
│           └── openai_qwen3-32b_Qwen3-14B_evaluation_detail.csv
│       
└── gpt-4o-mini/
    ├── evaluation_results.jsonl
    └── evaluation_results_avg.csv
```

Metrics (8 dimensions):

1. Knowledge accuracy

2. Emotional expression

3. Personality traits

4. Behavioral accuracy

5. Immersion

6. Adaptability

7. Behavioral coherence

8. Interaction richness

### Results Table (see the final CSV file) (`output_all/evaluation/by_model/<character_llm>/evaluation_results_avg.csv`)

```table
| file | title | character_name | error | Knowledge Accuracy | Emotional Expression | Personality Traits | Behavioral Accuracy | Immersion | Adaptability | Behavioral Coherence | Interaction Richness |

| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

```
