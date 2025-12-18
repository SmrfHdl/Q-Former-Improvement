"""
Visualization script for bounding box predictions from QFormerImproved.

This script demonstrates the attention-based bounding box prediction capability
of Level 1 (Object Detection Path) in the hierarchical Q-Former architecture.

Usage:
    python scripts/visualize_bounding_boxes.py --image_path path/to/image.jpg --question "What is in the image?"
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.q_former_improved import QFormerImproved
from torchvision import transforms


def load_model(config_path: str = None, checkpoint_path: str = None, device: str = 'cuda'):
    """Load the QFormerImproved model."""
    # Default config
    model = QFormerImproved(
        sequence_size=32,
        qformer_hidden_size=768,
        blocks_num=6,
        num_heads=12,
        num_object_queries=32,
        device=torch.device(device)
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model.eval()
    return model


def preprocess_image(image_path: str, size: int = 224):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_image


def visualize_bounding_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    confidences: np.ndarray,
    attention_maps: np.ndarray = None,
    question: str = "",
    save_path: str = None,
    top_k: int = 5,
    confidence_threshold: float = 0.3
):
    """
    Visualize predicted bounding boxes on the image.
    
    Args:
        image: Original PIL image
        boxes: (num_obj, 4) normalized [x, y, w, h] boxes
        confidences: (num_obj,) confidence scores
        attention_maps: (num_obj, num_patches) attention weights
        question: The input question
        save_path: Path to save visualization
        top_k: Number of top boxes to show
        confidence_threshold: Minimum confidence to display
    """
    # Convert to numpy
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)[::-1]
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if attention_maps is not None else 1, figsize=(16, 8))
    if attention_maps is None:
        axes = [axes]
    
    # === Plot 1: Image with Bounding Boxes ===
    ax1 = axes[0]
    ax1.imshow(img_array)
    ax1.set_title(f"Predicted Bounding Boxes\nQ: {question}", fontsize=12)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, top_k))
    
    shown_count = 0
    for i, idx in enumerate(sorted_indices):
        if shown_count >= top_k:
            break
            
        conf = confidences[idx]
        if conf < confidence_threshold:
            continue
            
        # Convert normalized [x, y, w, h] to pixel coordinates
        x, y, w, h = boxes[idx]
        x_pixel = x * img_w
        y_pixel = y * img_h
        w_pixel = w * img_w
        h_pixel = h * img_h
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_pixel, y_pixel), w_pixel, h_pixel,
            linewidth=2, edgecolor=colors[shown_count], facecolor='none'
        )
        ax1.add_patch(rect)
        
        # Add label
        ax1.text(
            x_pixel, y_pixel - 5,
            f'Obj {idx}: {conf:.2f}',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[shown_count], alpha=0.7)
        )
        
        shown_count += 1
    
    ax1.axis('off')
    
    # === Plot 2: Attention Heatmap (if available) ===
    if attention_maps is not None and len(axes) > 1:
        ax2 = axes[1]
        
        # Average attention of top-k objects
        top_indices = sorted_indices[:top_k]
        avg_attention = attention_maps[top_indices].mean(axis=0)
        
        # Reshape to 2D grid (assuming 16x16 patches for 224x224 image)
        grid_size = int(np.sqrt(len(avg_attention)))
        attention_2d = avg_attention.reshape(grid_size, grid_size)
        
        # Resize attention map to image size
        attention_resized = np.array(Image.fromarray(attention_2d).resize(
            (img_w, img_h), resample=Image.BILINEAR
        ))
        
        # Overlay attention on image
        ax2.imshow(img_array)
        ax2.imshow(attention_resized, alpha=0.6, cmap='jet')
        ax2.set_title("Attention Heatmap (Top-k objects)", fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_individual_objects(
    image: Image.Image,
    boxes: np.ndarray,
    confidences: np.ndarray,
    attention_maps: np.ndarray,
    save_path: str = None,
    top_k: int = 4
):
    """
    Visualize each detected object with its attention map.
    
    Args:
        image: Original PIL image
        boxes: (num_obj, 4) normalized [x, y, w, h] boxes
        confidences: (num_obj,) confidence scores
        attention_maps: (num_obj, num_patches) attention weights
        save_path: Path to save visualization
        top_k: Number of top objects to show
    """
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)[::-1][:top_k]
    
    # Create subplot grid
    fig, axes = plt.subplots(2, top_k, figsize=(4*top_k, 8))
    
    grid_size = int(np.sqrt(attention_maps.shape[1]))
    colors = plt.cm.rainbow(np.linspace(0, 1, top_k))
    
    for i, idx in enumerate(sorted_indices):
        conf = confidences[idx]
        x, y, w, h = boxes[idx]
        
        # === Row 1: Image with this object's box ===
        ax_img = axes[0, i]
        ax_img.imshow(img_array)
        
        x_pixel = x * img_w
        y_pixel = y * img_h
        w_pixel = w * img_w
        h_pixel = h * img_h
        
        rect = patches.Rectangle(
            (x_pixel, y_pixel), w_pixel, h_pixel,
            linewidth=3, edgecolor=colors[i], facecolor='none'
        )
        ax_img.add_patch(rect)
        ax_img.set_title(f'Object {idx}\nConf: {conf:.3f}', fontsize=10)
        ax_img.axis('off')
        
        # === Row 2: Attention map ===
        ax_attn = axes[1, i]
        
        attention_2d = attention_maps[idx].reshape(grid_size, grid_size)
        attention_resized = np.array(Image.fromarray(attention_2d).resize(
            (img_w, img_h), resample=Image.BILINEAR
        ))
        
        ax_attn.imshow(img_array, alpha=0.3)
        im = ax_attn.imshow(attention_resized, cmap='hot', alpha=0.7)
        ax_attn.set_title(f'Attention Map', fontsize=10)
        ax_attn.axis('off')
    
    plt.suptitle('Individual Object Detections with Attention Maps', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved individual visualization to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Q-Former bounding box predictions')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--question', type=str, default='What objects are in this image?',
                        help='Question to ask about the image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top objects to show')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path=args.checkpoint, device=device)
    
    # Load and preprocess image
    print(f"Loading image from {args.image_path}...")
    image_tensor, original_image = preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)
    
    # Prepare input
    samples = {
        'image_input': image_tensor,
        'question': [args.question],
        'answer': ['yes']  # Dummy answer for forward pass
    }
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(samples)
    
    # Extract predictions
    boxes = outputs['spatial_info'][0].cpu().numpy()  # (num_obj, 4)
    confidences = outputs['object_confidence'][0].squeeze(-1).cpu().numpy()  # (num_obj,)
    attention_maps = outputs['object_attention_maps'][0].cpu().numpy()  # (num_obj, num_patches)
    
    print(f"Detected {len(boxes)} object proposals")
    print(f"Box statistics: min={boxes.min():.3f}, max={boxes.max():.3f}")
    print(f"Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    
    # Generate visualizations
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Main visualization
    save_path = os.path.join(args.output_dir, f'{image_name}_bbox.png')
    visualize_bounding_boxes(
        original_image, boxes, confidences, attention_maps,
        question=args.question, save_path=save_path, top_k=args.top_k
    )
    
    # Individual object visualization
    save_path_individual = os.path.join(args.output_dir, f'{image_name}_individual.png')
    visualize_individual_objects(
        original_image, boxes, confidences, attention_maps,
        save_path=save_path_individual, top_k=min(args.top_k, 4)
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
