# Weights & Biases Integration

## Setup

1. **Install wandb package:**
   ```bash
   cd scripts
   chmod +x install_wandb.sh
   ./install_wandb.sh
   ```

   Or manually:
   ```bash
   uv add wandb
   # or
   pip install wandb
   ```

2. **Login to wandb:**
   ```bash
   wandb login
   ```
   Get your API key from: https://wandb.ai/authorize

## Usage

### Single Experiment
```bash
# Enable wandb (default)
python trainers/trainer.py --model_name qformer_base --use_wandb True

# Disable wandb
python trainers/trainer.py --model_name qformer_base --use_wandb False

# Custom project and entity
python trainers/trainer.py --model_name qformer_base \
    --wandb_project "my-project" \
    --wandb_entity "my-username"
```

### Multiple Experiments
```bash
# Enable wandb for all experiments (default)
python scripts/run.py --models all --encoders all --use_wandb True

# Disable wandb
python scripts/run.py --models all --encoders all --use_wandb False

# Custom wandb settings
python scripts/run.py --models all --encoders all \
    --wandb_project "q-former-experiments" \
    --wandb_entity "your-username"
```

## What Gets Logged

### Metrics
- Training/Validation/Test accuracy
- Training/Validation/Test losses:
  - Total loss
  - ITC loss (Image-Text Contrastive)
  - ITM loss (Image-Text Matching) 
  - IGT loss (Image Grounded Text Generation)
  - Answer prediction loss

### System Info
- Model architecture details
- Hyperparameters
- Training configuration
- Device information
- Random seeds

### Artifacts
- Model checkpoints (best models)
- Learning curves
- Configuration files

## View Results

1. Go to https://wandb.ai
2. Navigate to your project
3. View experiment runs, compare metrics, and analyze results

## Project Structure

- **Project Name:** `q-former-improvement` (default)
- **Run Names:** `{model}_{encoder}_run{id}_seed{seed}`
- **Tags:** `[model_name, encoder_type, run_id]`

## Troubleshooting

1. **Import Error:** Make sure wandb is installed
2. **Login Issues:** Check API key and internet connection
3. **Quota Exceeded:** Check your wandb account limits
4. **Offline Mode:** Set environment variable `WANDB_MODE=offline`
