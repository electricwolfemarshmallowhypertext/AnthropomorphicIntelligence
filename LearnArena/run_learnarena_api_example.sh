#!/bin/bash

# LearnArena Benchmark Script - API Mode Example
# ===============================================
# This script demonstrates how to run LearnArena benchmark using external API endpoints
# instead of local vLLM servers. This is useful for:
# - Using commercial API providers (OpenAI, Anthropic, etc.)
# - Testing against remote model endpoints
# - Running experiments without local GPU requirements

# Configuration
GAMES="TicTacToe-v0,Checkers-v0,Poker-v0"
OUTPUT_DIR="learnarena_results_api"
NUM_ROUNDS=20

# API Configuration
# You can either set API keys here or export them as environment variables
# Option 1: Export environment variables (recommended for security)
# export API_KEY_0="your-player0-api-key"
# export API_KEY_1="your-player1-api-key"

# Option 2: Pass directly as arguments (less secure, visible in process list)
PLAYER0_API_KEY="${API_KEY_0:-your-player0-api-key}"
PLAYER1_API_KEY="${API_KEY_1:-your-player1-api-key}"

# API endpoints
PLAYER0_API_BASE="https://api.openai.com/v1"  # Example: OpenAI API
PLAYER1_API_BASE="https://api.openai.com/v1"  # Example: OpenAI API

# Model names
PLAYER0_MODEL="gpt-4"
PLAYER1_MODEL="gpt-3.5-turbo"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "========================================="
echo "LearnArena Benchmark - API Mode"
echo "========================================="
echo "Player-0 Model: $PLAYER0_MODEL"
echo "Player-1 Model: $PLAYER1_MODEL"
echo "API Base URLs:"
echo "  Player-0: $PLAYER0_API_BASE"
echo "  Player-1: $PLAYER1_API_BASE"
echo "Games: $GAMES"
echo "Output: $OUTPUT_DIR"
echo "========================================="
echo ""

# Run the benchmark in API mode
python learnarena_benchmark.py \
    --mode api \
    --player0-model "$PLAYER0_MODEL" \
    --player0-api-base "$PLAYER0_API_BASE" \
    --player0-api-key "$PLAYER0_API_KEY" \
    --player1-model "$PLAYER1_MODEL" \
    --player1-api-base "$PLAYER1_API_BASE" \
    --player1-api-key "$PLAYER1_API_KEY" \
    --games "$GAMES" \
    --output-file "$OUTPUT_DIR/benchmark_${PLAYER1_MODEL//\//_}.json" \
    --num-rounds "$NUM_ROUNDS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Benchmark completed successfully!"
    echo "Results saved to: $OUTPUT_DIR/benchmark_${PLAYER1_MODEL//\//_}.json"
else
    echo ""
    echo "✗ Benchmark failed. Check logs for details."
    exit 1
fi
