"""
Trainer for Generative VQA models.

Key differences from classification approach:
- No vocabulary needed
- Model generates text directly  
- Accuracy = exact match between generated and ground truth
"""

import torch
import os
import sys
import json
import csv
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse
import numpy as np
import random
from loguru import logger

src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from datasets.dataset_generative import load_vqa_dataset_from_hf, create_generative_dataloader
from model_training.q_former_generative import (
    QFormerBaseGenerativeLightning,
    QFormerImprovedGenerativeLightning
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_generative(
    model_name: str,
    use_clip_for_text: bool = True,
    gpu_device: int = 0,
    num_gpus: int = None,
    results_dir: str = "../results_generative",
    models_dir: str = "../saved_models_generative",
    config_dir: str = "../configs",
    seed: int = 42,
    run_id: int = 0,
    use_wandb: bool = True,
    wandb_project: str = "q-former-generative",
    wandb_entity: str = None,
):
    """Train generative VQA model."""
    
    logger.info(f"Starting Generative VQA training: {model_name}")
    
    # Device setup
    if num_gpus and num_gpus > 1:
        device = torch.device("cuda:0")
    else:
        gpu_device = gpu_device if torch.cuda.is_available() else None
        device = torch.device(f"cuda:{gpu_device}" if gpu_device is not None else "cpu")

    set_seed(seed)

    # Load config
    config_dir_path = Path(config_dir).resolve()
    config_path = config_dir_path / f"config_{model_name.lower()}.json"
    
    if not config_path.exists():
        config_path = config_dir_path / "config_qformer_improved.json"
    
    with open(config_path, 'r') as f:
        hyperparams = json.load(f)

    hyperparams['use_clip_for_text'] = use_clip_for_text
    hyperparams.setdefault('max_answer_length', 10)
    
    # Directories
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    encoder_type = "clip" if use_clip_for_text else "bert"

    # Wandb
    wandb_logger = None
    if use_wandb:
        try:
            wandb_run_name = f"{model_name}_{encoder_type}_run{run_id}"
            wandb_logger = WandbLogger(
                project=wandb_project, entity=wandb_entity,
                name=wandb_run_name, save_dir=str(results_dir),
                tags=[model_name, encoder_type, "generative"]
            )
            wandb_logger.experiment.config.update({**hyperparams, "model_name": model_name})
        except Exception as e:
            logger.warning(f"Wandb init failed: {e}")
            wandb_logger = None

    # Load data
    logger.info("Loading dataset...")
    train_df, val_df, test_df = load_vqa_dataset_from_hf()
    
    train_loader, val_loader, test_loader = create_generative_dataloader(
        train_df, val_df, test_df,
        batch_size=hyperparams['batch_size'],
        device=device,
        use_augmentation=hyperparams.get('use_data_augmentation', False),
        num_workers=hyperparams.get('num_workers', 0),
        pin_memory=hyperparams.get('pin_memory', False),
    )

    # Create model
    if model_name.lower() == "qformer_base_generative":
        model = QFormerBaseGenerativeLightning(hyperparams=hyperparams, device=device)
    elif model_name.lower() == "qformer_improved_generative":
        model = QFormerImprovedGenerativeLightning(hyperparams=hyperparams, device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(results_dir),
        filename=f"{model_name}_{encoder_type}_run{run_id}_best",
        monitor="val_answer_accuracy",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    early_stopping = EarlyStopping(
        monitor="val_answer_accuracy",
        patience=hyperparams.get('patience', 10),
        mode="max",
        verbose=True
    )

    logger_ts = TensorBoardLogger(
        save_dir="logs",
        name=f"{model_name}_{encoder_type}_run{run_id}"
    )

    loggers = [logger_ts]
    if wandb_logger:
        loggers.append(wandb_logger)

    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    
    precision = hyperparams.get('precision', 'bf16-mixed')
    
    # Multi-GPU
    config_num_gpus = hyperparams.get('num_gpus', 1)
    actual_num_gpus = num_gpus if num_gpus else config_num_gpus
    strategy = 'ddp' if actual_num_gpus > 1 else 'auto'
    devices = actual_num_gpus if actual_num_gpus > 1 else ([gpu_device] if gpu_device is not None else None)
    
    trainer = pl.Trainer(
        max_epochs=hyperparams.get('num_epochs', 50),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        strategy=strategy,
        logger=loggers,
        callbacks=[early_stopping, checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        precision=precision,
    )

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    if os.path.exists(checkpoint_callback.best_model_path):
        best_model = type(model).load_from_checkpoint(
            checkpoint_callback.best_model_path,
            hyperparams=hyperparams, device=device
        )
        best_model.to(device)
        test_results = trainer.test(best_model, dataloaders=test_loader)
    else:
        test_results = trainer.test(model, dataloaders=test_loader)

    # Log results
    test_accuracy = test_results[0].get('test_answer_accuracy_epoch', 0.0) if test_results else 0.0
    if isinstance(test_accuracy, torch.Tensor):
        test_accuracy = test_accuracy.item()

    logger.info(f"Test Exact Match Accuracy: {test_accuracy:.4f}")

    # Save summary
    summary_file = results_dir / f"{model_name}_{encoder_type}_summary.csv"
    with open(summary_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id', 'seed', 'test_accuracy'])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({'run_id': run_id, 'seed': seed, 'test_accuracy': test_accuracy})

    if wandb_logger:
        try:
            wandb_logger.experiment.finish()
        except:
            pass

    return {'test_accuracy': test_accuracy, 'best_model_path': checkpoint_callback.best_model_path}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Generative VQA")

    parser.add_argument("--model_name", type=str, required=True,
                        choices=["qformer_base_generative", "qformer_improved_generative"])
    parser.add_argument("--use_clip_for_text", type=str2bool, default=True)
    parser.add_argument("--gpu_device", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="../results_generative")
    parser.add_argument("--models_dir", type=str, default="../saved_models_generative")
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--use_wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="q-former-generative")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    train_generative(
        model_name=args.model_name,
        use_clip_for_text=args.use_clip_for_text,
        gpu_device=args.gpu_device,
        num_gpus=args.num_gpus,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        config_dir=args.config_dir,
        seed=args.seed,
        run_id=args.run_id,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

