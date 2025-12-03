import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import torch
import json
from pathlib import Path
from PIL import Image

from model.q_former_base import QFormerBase
from visualization.visualize_base import QFormerBaseVisualizer, visualize_base_model


def load_base_model(checkpoint_path, config_path, device):
    """Load trained Q-Former Base model."""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = QFormerBase(
        sequence_size=config.get('sequence_size', 32),
        qformer_hidden_size=config.get('qformer_hidden_size', 768),
        blocks_num=config.get('blocks_num', 6),
        num_heads=config.get('num_heads', 12),
        num_queries=config.get('num_queries', 32),
        dropout_rate=config.get('dropout_rate', 0.1),
        use_clip_for_text=config.get('use_clip_for_text', True),
        unfreeze_layers=config.get('unfreeze_layers', 0),
        device=device
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('q_former_base.'):
                    new_key = k.replace('q_former_base.', '')
                    state_dict[new_key] = v
            
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint at {checkpoint_path}")
        print("Using randomly initialized model")
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize Q-Former Base')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, default='configs/config_qformer_base.json')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--question', type=str, default='What is in the image?')
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading Q-Former Base model...")
    model = load_base_model(args.checkpoint, args.config, device)
    
    visualize_base_model(model, args.image, args.question, args.output_dir, device)


if __name__ == "__main__":
    main()

