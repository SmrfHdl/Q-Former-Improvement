"""
Trainer for Open-Ended VQA models.

This trainer:
1. Loads data from HuggingFace (soumyasj/vqa-dataset)
2. Builds answer vocabulary from training data
3. Supports both base and improved open-ended models
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
import wandb
import argparse
import numpy as np
import random
from loguru import logger

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from datasets.dataset_openended import (
    load_vqa_dataset_from_hf,
    create_openended_dataloader,
    AnswerVocabulary
)
from model_training.q_former_base_openended import QFormerBaseOpenEndedLightning
from model_training.q_former_improved_openended import QFormerImprovedOpenEndedLightning


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_openended(
    model_name: str,
    use_clip_for_text: bool,
    gpu_device: int = 0,
    num_gpus: int = None,
    results_dir: str = "../results_openended",
    models_dir: str = "../saved_models_openended",
    config_dir: str = "../configs",
    vocab_dir: str = "../vocab",
    seed: int = 42,
    run_id: int = 0,
    save_learning_curves: bool = False,
    use_wandb: bool = True,
    wandb_project: str = "q-former-openended",
    wandb_entity: str = None,
    vocab_min_count: int = 9,
    vocab_max_size: int = 3129,
):
    """
    Train open-ended VQA model.
    
    Args:
        model_name: 'qformer_base_openended' or 'qformer_improved_openended'
        use_clip_for_text: Use CLIP or BERT for text encoding
        gpu_device: GPU device index
        num_gpus: Number of GPUs for multi-GPU training
        results_dir: Directory for results
        models_dir: Directory for model checkpoints
        config_dir: Directory containing configs
        vocab_dir: Directory for vocabulary files
        seed: Random seed
        run_id: Run identifier
        save_learning_curves: Save learning curves
        use_wandb: Use Weights & Biases
        wandb_project: Wandb project name
        wandb_entity: Wandb entity
        vocab_min_count: Minimum count for vocabulary inclusion
        vocab_max_size: Maximum vocabulary size
    """
    logger.info(f"Starting Open-Ended VQA training run {run_id} with seed {seed}")
    logger.info(f"Model: {model_name}, Use CLIP: {use_clip_for_text}")
    
    # Multi-GPU setup
    if num_gpus is not None and num_gpus > 1:
        logger.info(f"Multi-GPU training enabled: {num_gpus} GPUs")
        device = torch.device("cuda:0")
    else:
        logger.info(f"GPU device: {gpu_device}")
        gpu_device = gpu_device if torch.cuda.is_available() else None
        device = torch.device(f"cuda:{gpu_device}" if gpu_device is not None else "cpu")

    set_seed(seed)

    # Load config
    config_dir_path = Path(config_dir).resolve()
    
    # Map model names to config files
    config_mapping = {
        'qformer_base_openended': 'config_qformer_base_openended.json',
        'qformer_improved_openended': 'config_qformer_improved_openended.json',
    }
    
    config_filename = config_mapping.get(model_name.lower())
    if config_filename is None:
        # Try to use improved config as fallback
        config_filename = 'config_qformer_improved.json'
    
    config_path = config_dir_path / config_filename
    
    if not config_path.exists():
        # Fallback to base improved config
        config_path = config_dir_path / "config_qformer_improved.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        hyperparams = json.load(f)

    hyperparams['use_clip_for_text'] = use_clip_for_text
    
    # Create directories
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_dir = Path(vocab_dir).resolve()
    vocab_dir.mkdir(parents=True, exist_ok=True)

    encoder_type = "clip" if use_clip_for_text else "bert"

    # Initialize wandb
    wandb_logger = None
    if use_wandb:
        try:
            wandb_run_name = f"{model_name}_{encoder_type}_run{run_id}_seed{seed}"
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                save_dir=str(results_dir),
                log_model="all",
                tags=[model_name, encoder_type, "openended", f"run_{run_id}"]
            )
            
            wandb_logger.experiment.config.update({
                **hyperparams,
                "model_name": model_name,
                "encoder_type": encoder_type,
                "use_clip_for_text": use_clip_for_text,
                "seed": seed,
                "run_id": run_id,
                "vocab_min_count": vocab_min_count,
                "vocab_max_size": vocab_max_size,
            })
            
            logger.info(f"Wandb initialized: {wandb_project}/{wandb_run_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_logger = None

    # Load data from HuggingFace
    logger.info("Loading dataset from HuggingFace...")
    train_df, val_df, test_df = load_vqa_dataset_from_hf()
    
    # Load or create vocabulary
    vocab_path = vocab_dir / f"answer_vocab_min{vocab_min_count}_max{vocab_max_size}.json"
    
    if vocab_path.exists():
        logger.info(f"Loading existing vocabulary from {vocab_path}")
        answer_vocab = AnswerVocabulary()
        answer_vocab.load(str(vocab_path))
    else:
        logger.info("Building answer vocabulary from training data...")
        answer_vocab = AnswerVocabulary(
            min_count=vocab_min_count,
            max_vocab_size=vocab_max_size
        )
        train_answers = train_df['gt_answer'].to_list()
        answer_vocab.build_from_answers(train_answers)
        answer_vocab.save(str(vocab_path))
    
    num_answers = len(answer_vocab)
    logger.info(f"Answer vocabulary size: {num_answers}")

    # Create dataloaders
    train_loader, val_loader, test_loader, _ = create_openended_dataloader(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        answer_vocab=answer_vocab,
        batch_size=hyperparams['batch_size'],
        device=device,
        use_augmentation=hyperparams.get('use_data_augmentation', False),
        augmentation_prob=hyperparams.get('augmentation_prob', 0.5),
        num_workers=hyperparams.get('num_workers', 0),
        pin_memory=hyperparams.get('pin_memory', False),
    )

    # Create model
    if model_name.lower() == "qformer_base_openended":
        model = QFormerBaseOpenEndedLightning(
            hyperparams=hyperparams,
            num_answers=num_answers,
            device=device
        )
    elif model_name.lower() == "qformer_improved_openended":
        model = QFormerImprovedOpenEndedLightning(
            hyperparams=hyperparams,
            num_answers=num_answers,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Callbacks
    class MetricsCallBack(pl.Callback):
        def __init__(self, metrics_file: Path, model_name: str):
            super().__init__()
            self.metrics_file = metrics_file
            self.model_name = model_name
            
            if self.metrics_file is None:
                return
            
            headers = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
            with open(self.metrics_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            if self.metrics_file is None:
                return
            
            epoch = trainer.current_epoch
            try:
                train_acc = trainer.callback_metrics.get("train_answer_accuracy_epoch", 0.0)
                val_acc = trainer.callback_metrics.get("val_answer_accuracy_epoch", 0.0)
                train_loss = trainer.callback_metrics.get("train_total_loss_epoch", 0.0)
                val_loss = trainer.callback_metrics.get("val_total_loss_epoch", 0.0)

                train_acc_val = train_acc.item() if hasattr(train_acc, 'item') else float(train_acc)
                val_acc_val = val_acc.item() if hasattr(val_acc, 'item') else float(val_acc)
                train_loss_val = train_loss.item() if hasattr(train_loss, 'item') else float(train_loss)
                val_loss_val = val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)

                with open(self.metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss_val, train_acc_val, val_loss_val, val_acc_val])
            except Exception as e:
                logger.error(f"Error logging metrics at epoch {epoch}: {e}")

    metrics_file = results_dir / f"{model_name}_{encoder_type}_metrics_run{run_id}.csv" if save_learning_curves else None
    
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

    metrics_callback = MetricsCallBack(metrics_file, model_name)

    logger_ts = TensorBoardLogger(
        save_dir="logs",
        name=f"{model_name.lower()}_{encoder_type}_run{run_id}_logs",
        default_hp_metric=False
    )

    loggers = [logger_ts]
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    # H100 optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    
    precision = hyperparams.get('precision', 'bf16-mixed')
    
    # Multi-GPU configuration
    config_num_gpus = hyperparams.get('num_gpus', 1)
    actual_num_gpus = num_gpus if num_gpus is not None else config_num_gpus
    strategy = hyperparams.get('strategy', 'auto')
    
    if actual_num_gpus > 1:
        devices = actual_num_gpus
        if strategy == 'auto':
            strategy = 'ddp'
        logger.info(f"Using {actual_num_gpus} GPUs with {strategy} strategy")
    else:
        devices = [gpu_device] if gpu_device is not None else None
        strategy = 'auto'
    
    trainer = pl.Trainer(
        max_epochs=hyperparams.get('num_epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        strategy=strategy,
        logger=loggers,
        callbacks=[early_stopping, metrics_callback, checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        accumulate_grad_batches=1,
        precision=precision,
    )

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    if os.path.exists(checkpoint_callback.best_model_path):
        logger.info(f"Loading best model from {checkpoint_callback.best_model_path}")
        best_model = type(model).load_from_checkpoint(
            checkpoint_callback.best_model_path,
            hyperparams=hyperparams,
            num_answers=num_answers,
            device=device
        )
        best_model.to(device)
        test_results = trainer.test(best_model, dataloaders=test_loader)
    else:
        logger.info("Using current model state for testing")
        test_results = trainer.test(model, dataloaders=test_loader)

    # Extract test accuracy
    if test_results and len(test_results) > 0:
        test_accuracy = test_results[0].get('test_answer_accuracy_epoch', 0.0)
        if isinstance(test_accuracy, torch.Tensor):
            test_accuracy = test_accuracy.item()
    else:
        test_accuracy = 0.0

    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    # Save summary
    summary_results_file = results_dir / f"{model_name}_{encoder_type}_summary_results.csv"
    summary_exists = os.path.exists(summary_results_file)
    
    run_results = {
        'run_id': run_id,
        'seed': seed,
        'test_accuracy': test_accuracy,
        'num_answers': num_answers,
    }

    for key, value in test_results[0].items() if test_results else []:
        if isinstance(value, torch.Tensor):
            value = value.item()
        run_results[key] = value

    mode = 'a' if summary_exists else 'w'
    with open(summary_results_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=run_results.keys())
        if not summary_exists:
            writer.writeheader()
        writer.writerow(run_results)

    logger.info(f"Results saved to {summary_results_file}")

    # Cleanup wandb
    if use_wandb and wandb_logger is not None:
        try:
            wandb_logger.experiment.finish()
        except Exception as e:
            logger.warning(f"Error finishing wandb run: {e}")

    return {
        'model': model_name,
        'encoder': encoder_type,
        'run_id': run_id,
        'seed': seed,
        'test_accuracy': test_accuracy,
        'num_answers': num_answers,
        'best_model_path': checkpoint_callback.best_model_path
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Open-Ended VQA model")

    parser.add_argument("--model_name", type=str, required=True,
                        choices=["qformer_base_openended", "qformer_improved_openended"],
                        help="Which model to train.")

    parser.add_argument("--use_clip_for_text", type=str2bool, nargs='?', const=True, default=True,
                        help="Use CLIP encoder for text (default: True)")

    parser.add_argument("--gpu_device", type=int, default=0,
                        help="GPU device index (default: 0)")

    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs for multi-GPU training")

    parser.add_argument("--results_dir", type=str, default="../results_openended",
                        help="Directory to save results")

    parser.add_argument("--models_dir", type=str, default="../saved_models_openended",
                        help="Directory to save model checkpoints")

    parser.add_argument("--config_dir", type=str, default="../configs",
                        help="Directory containing configs")

    parser.add_argument("--vocab_dir", type=str, default="../vocab",
                        help="Directory for vocabulary files")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    parser.add_argument("--run_id", type=int, default=0,
                        help="Run identifier (default: 0)")

    parser.add_argument("--save_learning_curves", type=str2bool, nargs='?', const=True, default=False,
                        help="Save learning curves (default: False)")

    parser.add_argument("--use_wandb", type=str2bool, nargs='?', const=True, default=True,
                        help="Use Weights & Biases (default: True)")

    parser.add_argument("--wandb_project", type=str, default="q-former-openended",
                        help="Wandb project name")

    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity")

    parser.add_argument("--vocab_min_count", type=int, default=9,
                        help="Minimum count for vocabulary inclusion (default: 9)")

    parser.add_argument("--vocab_max_size", type=int, default=3129,
                        help="Maximum vocabulary size (default: 3129)")

    args = parser.parse_args()

    train_openended(
        model_name=args.model_name,
        use_clip_for_text=args.use_clip_for_text,
        gpu_device=args.gpu_device,
        num_gpus=args.num_gpus,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        config_dir=args.config_dir,
        vocab_dir=args.vocab_dir,
        seed=args.seed,
        run_id=args.run_id,
        save_learning_curves=args.save_learning_curves,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        vocab_min_count=args.vocab_min_count,
        vocab_max_size=args.vocab_max_size,
    )

