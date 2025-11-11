# Unveiling the Learning Mind of Language Models: A Cognitive Framework and Empirical Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Arxiv](https://img.shields.io/badge/arXiv-2506.13464-b31b1b.svg)](https://arxiv.org/abs/2506.13464)


LearnArena is a cognitively inspired benchmark for evaluating LLMs’ general learning abilities across three dimensions, Learning from Instructor, Learning from Concept, and Learning from Experience. It integrates interactive, conceptual, and experiential settings to test how models acquire, abstract, and adapt knowledge. By enabling consistent comparisons across open- and closed-source models, LearnArena provides a unified framework for assessing and advancing the learning capabilities of large language models.




## 📁 Project Structure

```
learnarena/
├── textarena/                            # Core game environment package
│   ├── __init__.py
│   ├── api.py
│   ├── core.py
│   ├── agents/                           # Agent implementations
│   ├── envs/                             # Game environments
│   │   ├── Checkers/
│   │   ├── Poker/
│   │   ├── SpellingBee/
│   │   ├── TicTacToe/
│   │   └── ... (other games)
│   ├── utils/                            # Utility functions
│   └── wrappers/                         # Environment wrappers
│
├── utils/                                # Shared utilities
│   ├── __init__.py
│   └── utils.py
│
├── learnarena_benchmark.py               # Main integrated benchmark
├── run_learnarena_benchmark.sh           # Benchmark runner script
│
├── model_scale_experiment.py             # Learning from Concept
├── run_learning_from_concept.sh          # LfC runner script
│
├── experience_driven_experiment.py       # Learning from Experience
├── run_learning_from_experience.sh       # LfE runner script
│
├── setup.py                              # Package setup
├── pyproject.toml                        # Modern package config
├── requirements.txt                      # Dependencies
├── install_env.sh                        # Installation script
├── restructure.sh                        # Restructuring helper
│
└── logs/ results/ environment_summary/   # Output directories
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- CUDA-capable GPU (for running vLLM servers)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/zxia545/nips-learnarena.git
cd nips-learnarena

# Run the installation script
bash install_env.sh
```

### Manual Installation

```bash
# Create conda environment
conda create -n learnarena python=3.10 -y
conda activate learnarena

# Install the package
pip install -e .
```

### Verify Installation

```python
import textarena as ta
print(ta.__version__)
```

## 🎯 Quick Start

### 1. Start vLLM Servers

Before running experiments, start vLLM servers for the models:

```bash
# Terminal 1: Start Player-0 (Teacher/Evaluator)
vllm serve qwen2.5-32b-chat --port 8000 --gpu-memory-utilization 0.9

# Terminal 2: Start Player-1 (Student Model)
vllm serve your-model-name --port 8001 --gpu-memory-utilization 0.9
```

### 2. Run the Main Benchmark

```bash
# Make sure you're in the root directory
bash run_learnarena_benchmark.sh
```

Or run a specific configuration:

```bash
python learnarena_benchmark.py \
    --player0-model "qwen2.5-32b-chat" \
    --player0-path "/path/to/qwen2.5-32b" \
    --player1-model "qwen2.5-7b-chat" \
    --player1-path "/path/to/qwen2.5-7b" \
    --games "TicTacToe-v0,Checkers-v0,Poker-v0" \
    --output-file "results/benchmark_results.json" \
    --num-rounds 20 \
    --gpu 4
```

### 3. Explore Individual Experiments

Each experiment type has its own Python script and shell runner in the root directory:

- **Main Benchmark**: `learnarena_benchmark.py` + `run_learnarena_benchmark.sh`
- **Learning from Concept**: `model_scale_experiment.py` + `run_learning_from_concept.sh`
- **Learning from Experience**: `experience_driven_experiment.py` + `run_learning_from_experience.sh`

## 🎮 Experiment Types and Scripts

### 1. LearnArena Benchmark (Integrated)

The main benchmark that evaluates all three learning dimensions simultaneously in competitive game scenarios.

#### Python Script: `learnarena_benchmark.py`

**Purpose:** Main integrated benchmark implementation evaluating Learning from Instructor (LfI), Learning from Concept (LfC), and Learning from Experience (LfE).

**Key Features:**

- Player-0 (Qwen2.5-32B) acts as instructor/evaluator providing post-game feedback
- Player-1 (evaluated model) learns from feedback, concepts, and experience
- 20 rounds per game with progressive learning
- Automatic game quality scoring (win=10, draw=5, loss=0)
- Generates game summaries and strategic guidance
- Selects top-3 relevant past experiences for in-context learning

**Command-Line Arguments:**

```bash
python learnarena_benchmark.py \
    --player0-model "qwen2.5-32b-chat"              # Instructor model name
    --player0-path "/path/to/qwen2.5-32b"           # Path to Player-0 model
    --player1-model "qwen2.5-7b-chat"               # Student model name
    --player1-path "/path/to/qwen2.5-7b"            # Path to Player-1 model
    --games "TicTacToe-v0,Checkers-v0,Poker-v0"     # Comma-separated games
    --output-file "results/benchmark.jsonl"          # Output file path
    --num-rounds 20                                  # Rounds per game (default: 20)
    --gpu 4                                          # Number of GPUs to use
```

**Output Format:** JSONL with fields including `game`, `round`, `player0_model`, `player1_model`, `learning_enabled`, `rewards`, `outcome`, `moves`, `instructor_feedback`, `score`, `game_summary`, `selected_experience_info`.

#### Shell Script: `run_learnarena_benchmark.sh`

**Purpose:** Automated batch execution of LearnArena benchmark across multiple models.

**Features:**

- Runs experiments for multiple Player-1 models automatically
- Manages vLLM server startup and shutdown
- Creates organized output directories with timestamps
- Generates comprehensive results summaries
- Color-coded console output for easy monitoring
- Error handling and progress tracking

**Configuration Variables:**

```bash
GAMES="TicTacToe-v0,Checkers-v0,Stratego-v0,..."   # Games to evaluate
OUTPUT_DIR="learnarena_results"                     # Results directory
NUM_ROUNDS=20                                        # Rounds per game
GPU=8                                                # GPU count
PLAYER0_MODEL="qwen2.5-32b-chat"                    # Fixed instructor
PLAYER0_PATH="/path/to/qwen2.5-32b"                 # Instructor path

# Multiple Player-1 configurations
declare -A PLAYER1_CONFIGS=(
    ["qwen2.5-1.5b"]="/path/to/qwen2.5-1.5b"
    ["qwen2.5-7b"]="/path/to/qwen2.5-7b"
    ["qwen2.5-14b"]="/path/to/qwen2.5-14b"
    ["qwen2.5-32b"]="/path/to/qwen2.5-32b"
)
```

**How to Run:**

```bash
# Edit configuration in the script first, then:
bash run_learnarena_benchmark.sh

# Or customize via command line:
GAMES="TicTacToe-v0,Poker-v0" NUM_ROUNDS=10 bash run_learnarena_benchmark.sh
```


**Output:**

- Individual results: `learnarena_results/{model_name}/benchmark_{model_name}.jsonl`
- Summary: `learnarena_results/comprehensive_summary.txt`
- Logs: `logs/learnarena_benchmark_*.log`


---

### 2. Learning from Concept (LfC)

Evaluates models' ability to use structured knowledge and abstract concepts with and without concept guidance.

#### Python Script: `model_scale_experiment.py`

**Purpose:** Tests how models of different scales utilize conceptual knowledge (game rules and strategic advice).

**Key Features:**

- Runs paired experiments: with concept vs. without concept
- Generates or loads game summaries with strategic guidance
- Tests across different model scales (1.5B, 7B, 14B, 32B parameters)
- Evaluates concept integration in competitive games
- Automatic comparison of performance with/without guidance

**Command-Line Arguments:**

```bash
python model_scale_experiment.py \
    --player0-model "qwen2.5-32b-chat"              # Concept provider model
    --player0-path "/path/to/qwen2.5-32b"           # Path to Player-0
    --player1-model "qwen2.5-7b-chat"               # Model to evaluate
    --player1-path "/path/to/qwen2.5-7b"            # Path to Player-1
    --games "TicTacToe-v0,Poker-v0,Checkers-v0"     # Games list
    --output-file "results/lfc_results.json"         # Output path
    --num-rounds 20                                  # Rounds per game
    --gpu 4                                          # GPU count
```

**Experimental Conditions:**

- **Without Concept**: Baseline performance without strategic guidance
- **With Concept**: Performance with game rules and strategic advice

**Output Format:** JSON with fields including `game`, `player0_model`, `player1_model`, `with_concept`, `wins`, `losses`, `draws`, `win_rate`, `total_games`.

#### Shell Script: `run_learning_from_concept.sh`

**Purpose:** Batch execution of Learning from Concept experiments across model scales.

**Features:**

- Tests multiple model scales automatically
- Compares performance with/without concept guidance
- Generates summary statistics for each model
- Organized output structure

**Configuration Variables:**

```bash
GAMES="TicTacToe-v0,Poker-v0,Checkers-v0,..."      # Games to test
OUTPUT_DIR="results"                                 # Output directory
NUM_ROUNDS=20                                        # Rounds per game
GPU=4                                                # GPU count
PLAYER0_MODEL="qwen2.5-32b-chat"                    # Concept provider
PLAYER0_PATH="/path/to/qwen2.5-32b"                 # Provider path

# Models to evaluate at different scales
declare -A PLAYER1_CONFIGS=(
    ["qwen2.5-1.5b"]="/path/to/qwen2.5-1.5b"
    ["qwen2.5-7b"]="/path/to/qwen2.5-7b"
    ["qwen2.5-14b"]="/path/to/qwen2.5-14b"
    ["qwen2.5-32b"]="/path/to/qwen2.5-32b"
)
```

**How to Run:**

```bash
# Edit paths in the script, then:
bash run_learning_from_concept.sh

# Results are automatically summarized at the end
```

**Output:**

- Per-model results: `results/model_scale_{model_name}_results.json`
- Automatic summary printed to console with win rates

**Environments Tested:**

- Competitive: Checkers, Poker, Stratego, TicTacToe, TruthAndDeception, UltimateTicTacToe
- Logic/Planning: LogicGrid, NaLogic, Plan, AlfWorld, ScienceWorld, BabyAI (for conceptual generalization tasks)

---

### 3. Learning from Experience (LfE)

Evaluates models' ability to learn from their own interaction history through experience-driven adaptation.

#### Python Script: `experience_driven_experiment.py`

**Purpose:** Tests models' ability to select, analyze, and learn from past game experiences.

**Key Features:**

- Score-based history selection (top-3 most relevant experiences)
- Direct history usage without summarization for authentic learning
- Progressive experience accumulation across rounds
- Self-analysis of successes and failures
- Optional history size limiting for controlled experiments

**Command-Line Arguments:**

```bash
python experience_driven_experiment.py \
    --player0-path "/path/to/qwen2.5-32b"           # Evaluator model path
    --player1-model "qwen2.5-7b-chat"               # Model to evaluate
    --player1-path "/path/to/qwen2.5-7b"            # Path to Player-1
    --games "TicTacToe-v0,Stratego-v0,Tak-v0"       # Games list
    --output-file "results/lfe_results.jsonl"        # Output path
    --num-rounds 20                                  # Rounds per game
    --history-limit 50                               # Max history size (optional)
    --gpu 4                                          # GPU count
```

**Experience Selection Mechanism:**

- Scores each past game based on outcome quality (win=10, draw=5, loss=0)
- Selects top-3 highest-scoring experiences
- Provides full game histories for in-context learning
- No external summarization to preserve authentic experience

**Output Format:** JSONL with fields including `game`, `round`, `player0_model`, `player1_model`, `experience_enabled`, `selected_experiences`, `rewards`, `outcome`, `moves`, `experience_analysis`.

#### Shell Script: `run_learning_from_experience.sh`

**Purpose:** Batch execution of Learning from Experience experiments with organized output management.

**Features:**

- Manages vLLM servers for both players
- Tests multiple model scales in sequence
- Creates organized output folders per model pair
- Automatic server cleanup after experiments
- Progress tracking and error reporting

**Configuration Variables:**

```bash
MODELS_BASE_DIR="PLACEHOLDER"                       # Base path for models
FIXED_PLAYER0_MODEL_PATH="${MODELS_BASE_DIR}/..."  # Evaluator path
PLAYER1_MODEL_PATHS=(                               # Models to test
    "${MODELS_BASE_DIR}/Qwen2.5-1.5B-Instruct"
    "${MODELS_BASE_DIR}/Qwen2.5-7B-Instruct"
    "${MODELS_BASE_DIR}/Qwen2.5-14B-Instruct"
    "${MODELS_BASE_DIR}/Qwen2.5-32B-Instruct"
)
PLAYER1_MODEL_NAMES=(                               # Model names
    "Qwen2.5-1.5B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Qwen2.5-14B-Instruct"
    "Qwen2.5-32B-Instruct"
)
GAMES="TicTacToe-v0,Poker-v0,Checkers-v0,..."      # Games to test
BASE_RESULTS_DIR="results/experience_experiments"   # Output directory
FIXED_PLAYER0_GPU_COUNT="8"                         # GPU for evaluator
PLAYER1_PORT="8001"                                 # Player-1 port
THREADS="8"                                         # Thread count
```

**How to Run:**

```bash
# IMPORTANT: Update MODELS_BASE_DIR and model paths in the script first
bash run_learning_from_experience.sh

# Results are organized by model pair
```

**Output Structure:**

```
results/experience_experiments/
├── Player0_Qwen2.5-32B-Instruct__Player1_Qwen2.5-1.5B-Instruct/
│   └── experience_results_Qwen2.5-1.5B-Instruct.jsonl
├── Player0_Qwen2.5-32B-Instruct__Player1_Qwen2.5-7B-Instruct/
│   └── experience_results_Qwen2.5-7B-Instruct.jsonl
└── ... (other model pairs)
```

**Experimental Settings:**

- **Without Experience**: First few rounds establish baseline
- **With Experience**: Later rounds use top-3 past experiences for learning
- **Progressive Learning**: Experience pool grows with each game

---

### 4. Installation and Setup Script

#### Shell Script: `install_env.sh`

**Purpose:** Automated installation of LearnArena and all dependencies.

**Features:**

- Creates conda environment with Python 3.10
- Installs textarena package in editable mode
- Sets up all required dependencies
- Provides usage instructions after installation

**How to Run:**

```bash
bash install_env.sh
```

**What It Does:**

1. Activates base conda environment
2. Creates new `learnarena` conda environment
3. Installs the `textarena` package from `./textarena` directory
4. Displays usage instructions for all experiment scripts

**Post-Installation:**

```bash
conda activate learnarena
python -c "import textarena as ta; print(ta.__version__)"
```

---

## 📊 Quick Reference: All Scripts


| Script                            | Type   | Purpose                              | Input                   | Output                               |
| --------------------------------- | ------ | ------------------------------------ | ----------------------- | ------------------------------------ |
| `learnarena_benchmark.py`         | Python | Integrated benchmark (LfI+LfC+LfE)   | Model paths, games list | JSONL with all learning metrics      |
| `run_learnarena_benchmark.sh`     | Shell  | Batch runner for main benchmark      | Edit config in script   | Results per model + summary          |
| `model_scale_experiment.py`       | Python | Learning from Concept experiments    | Model paths, games list | JSON with/without concept comparison |
| `run_learning_from_concept.sh`    | Shell  | Batch LfC across model scales        | Edit config in script   | Results per model                    |
| `experience_driven_experiment.py` | Python | Learning from Experience experiments | Model paths, games list | JSONL with experience selection      |
| `run_learning_from_experience.sh` | Shell  | Batch LfE across model scales        | Edit config in script   | Organized results by model pair      |
| `install_env.sh`                  | Shell  | Environment setup                    | None                    | Installed conda environment          |



## 🎲 Available Games

LearnArena includes 8+ competitive text-based games from the TextArena suite:


| Game                     | Category   | Description                                 |
| ------------------------ | ---------- | ------------------------------------------- |
| **TicTacToe-v0**         | Classic    | 3x3 grid game requiring tactical planning   |
| **Checkers-v0**          | Board Game | Traditional checkers with complex strategy  |
| **Poker-v0**             | Card Game  | Texas Hold'em variant with betting          |
| **Stratego-v0**          | Strategy   | Military board game with hidden information |
| **TruthAndDeception-v0** | Social     | Deduction game requiring theory of mind     |
| **SpellingBee-v0**       | Word Game  | Word spelling challenge                     |
| **SpiteAndMalice-v0**    | Card Game  | Competitive patience/solitaire variant      |
| **Tak-v0**               | Abstract   | Modern abstract strategy game               |
| **WordChains-v0**        | Word Game  | Word association challenge                  |
| **UltimateTicTacToe-v0** | Classic    | Meta-game variant of TicTacToe              |

Each game tests different cognitive abilities:

- **Strategic Planning**: Checkers, Stratego, Tak
- **Hidden Information**: Poker, Stratego
- **Social Reasoning**: TruthAndDeception
- **Language Skills**: SpellingBee, WordChains
- **Tactical Thinking**: TicTacToe, UltimateTicTacToe

## 🀽 Configuration

### Key Configuration Options

**Model Configuration:**

```python
--player0-model "qwen2.5-32b-chat"  # Teacher/evaluator model
--player0-path "/path/to/model"      # Path to model weights
--player1-model "your-model"         # Student model to evaluate
--player1-path "/path/to/model"      # Path to model weights
```

**Experiment Configuration:**

```python
--games "Game1-v0,Game2-v0"  # Comma-separated game list
--num-rounds 20               # Rounds per game (default: 20)
--gpu 4                       # Number of GPUs to use
--output-file "results.json"  # Output file path
```

**Learning Configuration:**

```python
--disable-learning            # Disable learning for baseline
--player0-port 8000          # Player-0 server port
--player1-port 8001          # Player-1 server port
```

### Output Format

Results are saved in JSON/JSONL format:

```json
{
  "game": "TicTacToe-v0",
  "round": 1,
  "player0_model": "qwen2.5-32b-chat",
  "player1_model": "qwen2.5-7b-chat",
  "learning_enabled": true,
  "rewards": [0, 1],
  "outcome": "Player 1 won",
  "player1_won": true,
  "moves": [...],
  "instructor_feedback": "...",
  "game_summary": "...",
  "score": 7
}
```




##  Key Differences Between Experiments

Understanding which script to use for your research question:


| Aspect                  | LearnArena Benchmark          | Learning from Concept       | Learning from Experience          |
| ----------------------- | ----------------------------- | --------------------------- | --------------------------------- |
| **Script**              | `learnarena_benchmark.py`     | `model_scale_experiment.py` | `experience_driven_experiment.py` |
| **Learning Dimensions** | All three (LfI+LfC+LfE)       | LfC only                    | LfE only                          |
| **Player-0 Role**       | Instructor/Evaluator          | Concept Provider            | Evaluator only                    |
| **Player-1 Receives**   | Feedback + Concepts + History | Strategic guidance only     | Past game experiences only        |
| **Best For**            | Complete evaluation           | Testing concept utilization | Testing experience-based learning |
| **Comparison**          | Learning vs. No learning      | With concept vs. Without    | With experience vs. Baseline      |
| **Output Type**         | JSONL (per round)             | JSON (aggregated)           | JSONL (per round)                 |
| **Typical Runtime**     | Longest (all features)        | Medium                      | Medium                            |
| **Model Requirements**  | 2 models + servers            | 2 models + servers          | 2 models + servers                |



## 🀽 Citation

If you use LearnArena in your research, please cite:

```bibtex
@article{hu2025unveiling,
  title={Unveiling the Learning Mind of Language Models: A Cognitive Framework and Empirical Study},
  author={Hu, Zhengyu and Lian, Jianxun and Xiao, Zheyuan and Zhang, Seraphina and Wang, Tianfu and Yuan, Nicholas Jing and Xie, Xing and Xiong, Hui},
  journal={arXiv preprint arXiv:2506.13464},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




## 🙏 Acknowledgments

- Original [TextArena](https://github.com/LeonGuertler/TextArena) framework by Leon Guertler
- Built on top of [vLLM](https://github.com/vllm-project/vllm) for efficient model serving
