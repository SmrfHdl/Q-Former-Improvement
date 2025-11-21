#!/bin/bash
# Script to install wandb for the project

echo "Installing wandb package..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to install wandb..."
    uv add wandb
else
    echo "uv not found, using pip..."
    pip install wandb
fi

echo "wandb installation completed!"
echo ""
echo "To login to wandb, run: wandb login"
echo "You can get your API key from: https://wandb.ai/authorize"
