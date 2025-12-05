import subprocess
import argparse
from pathlib import Path
import time
import csv
import pandas as pd
import numpy as np
import threading
from loguru import logger
import os


def parse_boolean(value: str | bool):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def run_single_experiment(model: str, use_clip: bool, encoder_name: str, gpu_id: int, run_id: int, seed: int,
                          results_dir: str, models_dir: str, config_dir: str,
                          log_dir: str, all_runs_file: str, use_wandb: bool = True, 
                          wandb_project: str = "q-former-gqa", wandb_entity: str = None,
                          max_train_samples: int = None, max_val_samples: int = None, max_test_samples: int = None):
    logger.info(f"Starting experiment: {model} + {encoder_name}, run id {run_id}, seed {seed} on GPU {gpu_id}")

    with open(all_runs_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model, encoder_name, run_id, seed, gpu_id, "RUNNING", ''])

    # Get the absolute path to the trainer script
    script_dir = Path(__file__).parent
    trainer_path = script_dir.parent / "src" / "trainers" / "trainer_gqa.py"
    
    cmd = [
        "uv", "run", str(trainer_path),
        "--model_name", model,
        "--use_clip_for_text", str(use_clip),
        "--gpu_device", str(gpu_id),
        "--results_dir", str(results_dir),
        "--models_dir", str(models_dir),
        "--config_dir", str(config_dir),
        "--seed", str(seed),
        "--run_id", str(run_id),
        "--use_wandb", str(use_wandb),
        "--wandb_project", wandb_project
    ]
    
    if wandb_entity:
        cmd.extend(["--wandb_entity", wandb_entity])
    
    if max_train_samples:
        cmd.extend(["--max_train_samples", str(max_train_samples)])
    if max_val_samples:
        cmd.extend(["--max_val_samples", str(max_val_samples)])
    if max_test_samples:
        cmd.extend(["--max_test_samples", str(max_test_samples)])

    log_file = log_dir / f"{model}_{encoder_name}_run{run_id}_gpu{gpu_id}.log"

    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()

    status = "COMPLETED" if process.returncode == 0 else f"FAILED ({process.returncode})"

    tracking_df = pd.read_csv(all_runs_file)
    mask = (tracking_df['model'] == model) & \
           (tracking_df['encoder'] == encoder_name) & \
           (tracking_df['run_id'] == run_id)
    
    tracking_df.loc[mask, 'status'] = status

    if process.returncode == 0:
        summary_file = results_dir / f"{model}_{encoder_name}_summary.csv"
        if os.path.exists(summary_file):
            try:
                summary_df = pd.read_csv(summary_file)
                run_data = summary_df[summary_df['run_id'] == run_id]
                if not run_data.empty:
                    tracking_df.loc[mask, 'test_exact_match'] = run_data['test_exact_match'].values[0]
            except Exception as e:
                logger.error(f"Error reading summary results for {model} + {encoder_name}, run id {run_id}: {e}")

    tracking_df.to_csv(all_runs_file, index=False)

    if process.returncode == 0:
        logger.info(f"Experiment completed successfully: {model} + {encoder_name}, run id {run_id}")
    else:
        logger.error(f"Experiment failed: {model} + {encoder_name}, run id {run_id} with return code {process.returncode}")

    return process.returncode


def run_experiments_group(model: str, use_clip: bool, encoder_name: str, gpu_id: int, num_runs: int, base_seed: int,
                          results_dir: str, models_dir: str, config_dir: str,
                          log_dir: str, all_runs_file: str, use_wandb: bool, wandb_project: str, 
                          wandb_entity: str, max_train_samples: int, max_val_samples: int, max_test_samples: int):
    logger.info(f"Starting group of experiments for {model} + {encoder_name} on GPU {gpu_id}")

    for run_id in range(num_runs):
        seed = base_seed + run_id
        return_code = run_single_experiment(
            model, use_clip, encoder_name, gpu_id, run_id, seed,
            results_dir, models_dir, config_dir,
            log_dir, all_runs_file, use_wandb, wandb_project, wandb_entity,
            max_train_samples, max_val_samples, max_test_samples
        )

    logger.info(f"Completed group of experiments for {model} + {encoder_name} on GPU {gpu_id}")


def run_gpu_experiments_sequentially(groups: list[dict], num_runs: int, base_seed: int,
                                     results_dir: str, models_dir: str, config_dir: str,
                                     log_dir: str, all_runs_file: str, use_wandb: bool, wandb_project: str, 
                                     wandb_entity: str, max_train_samples: int, max_val_samples: int, 
                                     max_test_samples: int):
    for group in groups:
        run_experiments_group(
            group['model'],
            group['use_clip'],
            group['encoder_name'],
            group['gpu_id'],
            num_runs,
            base_seed,
            results_dir,
            models_dir,
            config_dir,
            log_dir,
            all_runs_file,
            use_wandb,
            wandb_project,
            wandb_entity,
            max_train_samples,
            max_val_samples,
            max_test_samples
        )


def main():
    parser = argparse.ArgumentParser(description="Run GQA Visual Reasoning experiments")

    parser.add_argument("--results_dir", type=str, default="results_gqa", 
                        help="Directory to save results (default: 'results_gqa')")
    parser.add_argument("--models_dir", type=str, default="saved_models_gqa",
                        help="Directory to save models (default: 'saved_models_gqa')")
    parser.add_argument("--config_dir", type=str, default="configs",
                        help="Directory containing model configs (default: 'configs')")
    
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated list of GPU device IDs to use (default: '0')")
    
    parser.add_argument("--models", type=str, default="all",
                        choices=["all", "qformer_base_gqa", "qformer_improved_gqa"],
                        help="Which models to run (default: 'all')")
    parser.add_argument("--encoders", type=str, default="clip",
                        choices=["all", "clip", "bert"],
                        help="Which text encoders to use (default: 'clip')")
    
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs per experiment (default: 3)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for random number generation (default: 42)")
    
    parser.add_argument("--use_wandb", type=parse_boolean, default=True,
                        help="Whether to use Weights & Biases logging (default: True)")
    parser.add_argument("--wandb_project", type=str, default="q-former-gqa",
                        help="Wandb project name (default: 'q-former-gqa')")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity/username (default: None, uses default entity)")
    
    # Dataset size limits (for debugging/fast iteration)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum training samples (default: all)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Maximum validation samples (default: all)")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Maximum test samples (default: all)")
    
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs_gqa")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.models == "all":
        models = ["qformer_base_gqa", "qformer_improved_gqa"]
    else:
        models = [args.models]

    if args.encoders == "all":
        encoders = [True, False]
        encoder_names = ["clip", "bert"]
    elif args.encoders == "clip":
        encoders = [True]
        encoder_names = ["clip"]
    elif args.encoders == "bert":
        encoders = [False]
        encoder_names = ["bert"]

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",")]

    experiment_groups = []
    gpu_assignments_idx = 0

    for model in models:
        for encoder, encoder_name in zip(encoders, encoder_names):
            gpu_id = gpu_ids[gpu_assignments_idx % len(gpu_ids)]
            experiment_groups.append({
                'model': model,
                'use_clip': encoder,
                'encoder_name': encoder_name,
                'gpu_id': int(gpu_id)
            })
            gpu_assignments_idx += 1

    total_experiments = len(experiment_groups) * args.num_runs

    logger.info(f"=== GQA Visual Reasoning Experiments ===")
    logger.info(f"Planning to run {total_experiments} experiments across {len(gpu_ids)} GPUs.")
    logger.info(f"Each model-encoder combination will be run {args.num_runs} times with different seeds.")
    logger.info(f"Evaluation metric: Normalized Exact Match")

    for group in experiment_groups:
        logger.info(f"Model: {group['model']}, Encoder: {group['encoder_name']}, GPU: {group['gpu_id']}")

    proceed = input("\nProceed with the experiments? (y/n): ")
    if proceed.lower() != 'y':
        logger.info("Experiment run cancelled by user.")
        return
    
    all_runs_file = results_dir / "all_runs_summary.csv"
    with open(all_runs_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'encoder', 'run_id', 'seed', 'gpu', 'status', 'test_exact_match'])

    # Convert config_dir to absolute path
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = (Path.cwd() / config_dir).resolve()

    gpu_groups = {}
    for group in experiment_groups:
        gpu_id = group['gpu_id']
        if gpu_id not in gpu_groups:
            gpu_groups[gpu_id] = []
        gpu_groups[gpu_id].append(group)

    threads = []
    for gpu_id, groups in gpu_groups.items():
        thread = threading.Thread(
            target=run_gpu_experiments_sequentially,
            args=(
                groups,
                args.num_runs,
                args.base_seed,
                results_dir,
                models_dir,
                config_dir,
                log_dir,
                all_runs_file,
                args.use_wandb,
                args.wandb_project,
                args.wandb_entity,
                args.max_train_samples,
                args.max_val_samples,
                args.max_test_samples
            )
        )
        thread.start()
        threads.append(thread)
        time.sleep(1)

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.warning("Experiment run interrupted by user.")
        logger.info("Press Ctrl+C again to force exit.")
        time.sleep(2)

    logger.info("All experiments completed.")
    
    # Generate summary statistics
    generate_summary_stats(results_dir, models, encoder_names)
    
    logger.info("Experiment run finished.")


def generate_summary_stats(results_dir: str, models: list[str], encoder_names: list[str]):
    """Generate summary statistics for all experiments."""
    results_dir = Path(results_dir)
    
    summary_stats = []

    for model in models:
        for encoder in encoder_names:
            summary_file = results_dir / f"{model}_{encoder}_summary.csv"

            if summary_file.exists():
                try:
                    df = pd.read_csv(summary_file)
                    
                    mean_exact_match = df['test_exact_match'].mean()
                    std_exact_match = df['test_exact_match'].std()

                    summary_stats.append({
                        'model': model,
                        'encoder': encoder,
                        'exact_match_mean': mean_exact_match,
                        'exact_match_std': std_exact_match,
                        'num_runs': len(df)
                    })
                    
                    logger.info(f"{model} + {encoder}: {mean_exact_match:.4f} ± {std_exact_match:.4f} ({len(df)} runs)")
                except Exception as e:
                    logger.error(f"Error processing {summary_file}: {e}")

    if summary_stats:
        stats_df = pd.DataFrame(summary_stats)
        stats_df.to_csv(results_dir / "experiments_summary_statistics.csv", index=False)
        logger.info(f"Summary statistics saved to {results_dir / 'experiments_summary_statistics.csv'}")


if __name__ == "__main__":
    main()

