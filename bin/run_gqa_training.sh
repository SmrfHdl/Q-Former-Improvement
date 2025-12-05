#!/bin/bash

# Run GQA Visual Reasoning training script
# Usage: ./bin/run_gqa_training.sh [options]
#
# Examples:
#   # Run QFormer Improved with default settings
#   ./bin/run_gqa_training.sh
#
#   # Run specific model with limited samples (for debugging)
#   ./bin/run_gqa_training.sh --model_name qformer_base_gqa --max_train_samples 1000 --max_val_samples 500

cd "$(dirname "$0")/.."

# Default: Run QFormer Improved with CLIP encoder
uv run src/trainers/trainer_gqa.py \
    --model_name qformer_improved_gqa \
    --use_clip_for_text True \
    --gpu_device 0 \
    --results_dir results_gqa \
    --models_dir saved_models_gqa \
    --config_dir configs \
    --seed 42 \
    --run_id 0 \
    --use_wandb False \
    "$@"

