import subprocess
import argparse
from pathlib import Path
import time
import csv
import pandas as pd
import numpy as np
import threading
from loguru import logger


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
                          save_learning_curves: bool, results_dir: str, models_dir: str, config_dir: str, data_dir: str,
                          log_dir: str, all_runs_file: str):
    logger.info(f"Starting experiment: {model} + {encoder_name}, run id {run_id}, seed {seed} on GPU {gpu_id}")

    with open(all_runs_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model, encoder_name, run_id, seed, gpu_id, "RUNNING", '', ''])

    cmd = [
        "python", "../trainers/trainer.py",
        "--model_name", model,
        "--use_clip_for_text", str(use_clip),
        "--gpu_device", str(gpu_id),
        "--results_dir", str(results_dir),
        "--models_dir", str(models_dir),
        "--config_dir", str(config_dir),
        "--data_dir", str(data_dir),
        "--seed", str(seed),
        "--run_id", str(run_id),
        "--save_learning_curves", str(save_learning_curves)
    ]

    log_file = log_dir / f"{model}_{encoder_name}_run{run_id}_gpu{gpu_id}.log"

    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()

    