"""
Demo script to visualize Q-Former Improved object detection and attention.

Usage:
    python scripts/demo_visualization.py --checkpoint <path> --image <path> --question "Is there a dog?"
    
Or run on multiple samples:
    python scripts/demo_visualization.py --checkpoint <path> --data_file <path> --num_samples 5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import torch
import json
from pathlib import Path
from PIL import Image

from model.q_former_improved import QFormerImproved
from visualization.visualize_attention import QFormerVisualizer, visualize_sample


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load trained Q-Former Improved model."""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = QFormerImproved(
        sequence_size=config['sequence_size'],
        qformer_hidden_size=config['qformer_hidden_size'],
        blocks_num=config['blocks_num'],
        num_heads=config['num_heads'],
        num_object_queries=config.get('num_object_queries', 32),
        num_relation_queries=config.get('num_relation_queries', 48),
        num_global_queries=config.get('num_global_queries', 32),
        dropout_rate=config['dropout_rate'],
        use_clip_for_text=config['use_clip_for_text'],
        unfreeze_layers=config.get('unfreeze_layers', 0),
        device=device
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            # Lightning checkpoint
            state_dict = {k.replace('q_former_improved.', ''): v 
                         for k, v in checkpoint['state_dict'].items() 
                         if k.startswith('q_former_improved.')}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Using randomly initialized model (no checkpoint provided)")
    
    model.eval()
    return model


def visualize_single(model, image_path: str, question: str, output_dir: str, device: torch.device):
    """Visualize a single image-question pair."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Process image
    image_input = model.vision_encoder.processor(images=image, return_tensors="pt")
    image_input = {k: v.to(device) for k, v in image_input.items()}
    
    # Create visualizer
    visualizer = QFormerVisualizer(model, device)
    
    # Extract features
    print(f"\nProcessing: {question}")
    print(f"Image: {image_path}")
    
    features = visualizer.extract_features(image_input, question)
    
    # Print summary
    confidence = features['object_confidence'].squeeze().cpu().numpy()
    print(f"\nObject Detection Summary:")
    print(f"   - Total queries: {len(confidence)}")
    print(f"   - High confidence (>0.5): {(confidence > 0.5).sum()}")
    print(f"   - Max confidence: {confidence.max():.3f}")
    print(f"   - Mean confidence: {confidence.mean():.3f}")
    
    # Generate visualizations
    sample_name = Path(image_path).stem
    
    print(f"\nGenerating visualizations...")
    
    # Object detection
    fig1 = visualizer.visualize_object_attention(
        image, features,
        save_path=os.path.join(output_dir, f'{sample_name}_objects.png'),
        top_k=8
    )
    
    # Attention heatmaps
    fig2 = visualizer.visualize_attention_heatmap(
        image, features,
        save_path=os.path.join(output_dir, f'{sample_name}_attention.png')
    )
    
    # Hierarchical features
    fig3 = visualizer.visualize_hierarchical_features(
        features,
        save_path=os.path.join(output_dir, f'{sample_name}_hierarchical.png')
    )
    
    import matplotlib.pyplot as plt
    plt.close('all')
    
    print(f"\nVisualizations saved to: {output_dir}/")
    print(f"   - {sample_name}_objects.png")
    print(f"   - {sample_name}_attention.png")
    print(f"   - {sample_name}_hierarchical.png")


def visualize_from_datafile(model, data_file: str, images_dir: str, 
                            output_dir: str, num_samples: int, device: torch.device):
    """Visualize multiple samples from a data file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data file
    samples = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            image_name, rest = line.split('\t', 1)
            image_name = image_name.split('#')[0]
            
            words = rest.strip().split()
            answer = words[-1].lower()
            question = ' '.join(words[:-1])
            
            if answer in ['yes', 'no']:
                samples.append({
                    'image_path': os.path.join(images_dir, image_name),
                    'question': question,
                    'answer': answer
                })
    
    print(f"Found {len(samples)} valid samples")
    
    # Visualize random samples
    import random
    random.seed(42)
    selected = random.sample(samples, min(num_samples, len(samples)))
    
    for i, sample in enumerate(selected):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(selected)}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        
        visualize_single(
            model,
            sample['image_path'],
            sample['question'],
            os.path.join(output_dir, f'sample_{i}'),
            device
        )


def main():
    parser = argparse.ArgumentParser(description='Visualize Q-Former Improved attention')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/config_qformer_improved.json',
                       help='Path to config file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image')
    parser.add_argument('--question', type=str, default='What is in the image?',
                       help='Question about the image')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to data file for multiple samples')
    parser.add_argument('--images_dir', type=str, default='data/images',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize from data file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, args.config, device)
    
    if args.image:
        # Single image visualization
        visualize_single(model, args.image, args.question, args.output_dir, device)
    elif args.data_file:
        # Multiple samples from data file
        visualize_from_datafile(
            model, args.data_file, args.images_dir, 
            args.output_dir, args.num_samples, device
        )
    else:
        print("\nError: Please provide either --image or --data_file")
        parser.print_help()


if __name__ == "__main__":
    main()

