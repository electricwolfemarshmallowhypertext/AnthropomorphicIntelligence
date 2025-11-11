#!/bin/bash

# LearnArena Installation Script
# ===============================

echo "Installing LearnArena and its dependencies..."

# Activate base conda environment
source activate base

# Create conda environment
echo "Creating conda environment 'learnarena' with Python 3.10..."
conda create -n learnarena python=3.10 -y

# Activate the environment
conda activate learnarena

# Install the package in editable mode
echo "Installing learnarena package..."

pip install -e .

pip install -r requirements.txt