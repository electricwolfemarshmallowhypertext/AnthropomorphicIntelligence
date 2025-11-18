#!/bin/bash

# Learning from Concept Experiment - API Mode Example
# ====================================================

# Configuration
GAMES="TicTacToe-v0,Poker-v0,Checkers-v0"
OUTPUT_DIR="results/concept_api"
NUM_ROUNDS=20

# API Configuration
# Export environment variables for API keys (recommended)
# export API_KEY_0="your-player0-api-key"
# export API_KEY_1="your-player1-api-key"

PLAYER0_API_KEY="${API_KEY_0:-your-player0-api-key}"
PLAYER1_API_KEY="${API_KEY_1:-your-player1-api-key}"

# API endpoints
PLAYER0_API_BASE="https://api.openai.com/v1"
PLAYER1_API_BASE="https://api.openai.com/v1"

# Model names
PLAYER0_MODEL="gpt-4"
PLAYER1_MODEL="gpt-3.5-turbo"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "========================================="
echo "Learning from Concept - API Mode"
echo "========================================="
echo "Player-0: $PLAYER0_MODEL"
echo "Player-1: $PLAYER1_MODEL"
echo "Games: $GAMES"
echo "========================================="
echo ""

# Run the experiment
python model_scale_experiment.py \
    --mode api \
    --player0-model "$PLAYER0_MODEL" \
    --player0-api-base "$PLAYER0_API_BASE" \
    --player0-api-key "$PLAYER0_API_KEY" \
    --player1-model "$PLAYER1_MODEL" \
    --player1-api-base "$PLAYER1_API_BASE" \
    --player1-api-key "$PLAYER1_API_KEY" \
    --games "$GAMES" \
    --output-file "$OUTPUT_DIR/concept_${PLAYER1_MODEL//\//_}.json" \
    --num-rounds "$NUM_ROUNDS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Experiment completed successfully!"
else
    echo ""
    echo "✗ Experiment failed. Check logs for details."
    exit 1
fi
