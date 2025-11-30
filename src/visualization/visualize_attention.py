"""
Visualization module for Q-Former Improved architecture.
Visualizes object detection, attention maps, and hierarchical features.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import seaborn as sns
from typing import Optional, Tuple, List
import os


class QFormerVisualizer:
    """Visualize Q-Former Improved attention and object detection."""
    
    def __init__(self, model, device: torch.device = torch.device('cuda')):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Store intermediate outputs
        self.attention_maps = {}
        self.hooks = []
        
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # For transformer layers, capture attention weights
                if hasattr(output, '__len__') and len(output) > 1:
                    self.attention_maps[name] = output
            return hook
        
        # Hook into object path transformer
        if hasattr(self.model, 'object_path'):
            for i, layer in enumerate(self.model.object_path.object_transformer.layers):
                hook = layer.mhca.register_forward_hook(get_attention_hook(f'object_cross_attn_{i}'))
                self.hooks.append(hook)
                
        # Hook into relation path transformer
        if hasattr(self.model, 'relation_path'):
            for i, layer in enumerate(self.model.relation_path.relation_transformer.layers):
                hook = layer.mhca.register_forward_hook(get_attention_hook(f'relation_cross_attn_{i}'))
                self.hooks.append(hook)
                
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = {}
        
    @torch.no_grad()
    def extract_features(self, image_input: dict, question: str) -> dict:
        """
        Extract intermediate features from the model.
        
        Args:
            image_input: Dict with 'pixel_values' tensor
            question: Question string
            
        Returns:
            Dict containing various intermediate features
        """
        self.model.eval()
        
        # Prepare inputs
        samples = {
            'image_input': image_input,
            'question': [question],
            'answer': ['yes']  # Dummy answer for forward pass
        }
        
        # Get image features
        image_features = self.model.vision_encoder.encode(image_input)
        image_features = self.model.vision_projection(image_features)
        image_features = self.model.vision_norm(image_features)
        
        # Get text embeddings
        question_output, question_tokens = self.model.encode_text([question])
        text_embeddings = question_output['last_hidden_state']
        text_embeddings = self.model.text_projection(text_embeddings)
        text_embeddings = self.model.text_norm(text_embeddings)
        
        # Run object detection path
        attention_mask = self.model.generate_attention_mask(
            task='itc',
            query_len=32,
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        object_features, spatial_info, object_confidence = self.model.object_path(
            image_features, text_embeddings, attention_mask
        )
        
        # Run relation path
        attention_mask_l2 = self.model.generate_attention_mask(
            task='itc',
            query_len=64,
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        relation_features, relation_types = self.model.relation_path(
            object_features, spatial_info, image_features, text_embeddings, attention_mask_l2
        )
        
        return {
            'image_features': image_features,
            'object_features': object_features,
            'spatial_info': spatial_info,
            'object_confidence': object_confidence,
            'relation_features': relation_features,
            'relation_types': relation_types,
            'text_embeddings': text_embeddings
        }
    
    def visualize_object_attention(
        self,
        image: Image.Image,
        features: dict,
        save_path: Optional[str] = None,
        top_k: int = 8
    ) -> plt.Figure:
        """
        Visualize object detection results with attention.
        
        Args:
            image: Original PIL image
            features: Output from extract_features()
            save_path: Path to save the figure
            top_k: Number of top objects to visualize
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Object confidence scores
        confidence = features['object_confidence'].squeeze().cpu().numpy()
        top_indices = np.argsort(confidence)[-top_k:][::-1]
        
        axes[0, 1].bar(range(len(confidence)), confidence, color='steelblue', alpha=0.7)
        axes[0, 1].bar(top_indices, confidence[top_indices], color='crimson', alpha=0.9)
        axes[0, 1].set_xlabel('Object Query Index', fontsize=12)
        axes[0, 1].set_ylabel('Confidence Score', fontsize=12)
        axes[0, 1].set_title(f'Object Confidence (Top {top_k} highlighted)', fontsize=14, fontweight='bold')
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Spatial bounding boxes
        spatial = features['spatial_info'].squeeze().cpu().numpy()  # (num_queries, 4)
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        axes[0, 2].imshow(image)
        colors = plt.cm.rainbow(np.linspace(0, 1, top_k))
        
        for idx, (query_idx, color) in enumerate(zip(top_indices, colors)):
            x, y, box_w, box_h = spatial[query_idx]
            # Convert normalized coords to pixel coords
            x_pixel = x * w
            y_pixel = y * h
            w_pixel = box_w * w
            h_pixel = box_h * h
            
            rect = patches.Rectangle(
                (x_pixel - w_pixel/2, y_pixel - h_pixel/2),
                w_pixel, h_pixel,
                linewidth=2, edgecolor=color, facecolor='none',
                label=f'Obj {query_idx}: {confidence[query_idx]:.2f}'
            )
            axes[0, 2].add_patch(rect)
            
        axes[0, 2].legend(loc='upper right', fontsize=8)
        axes[0, 2].set_title('Predicted Bounding Boxes', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Object feature similarity matrix
        obj_features = features['object_features'].squeeze().cpu().numpy()
        obj_similarity = np.corrcoef(obj_features)
        
        sns.heatmap(obj_similarity, ax=axes[1, 0], cmap='RdBu_r', center=0,
                    square=True, cbar_kws={'shrink': 0.8})
        axes[1, 0].set_title('Object Feature Similarity', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Object Query', fontsize=12)
        axes[1, 0].set_ylabel('Object Query', fontsize=12)
        
        # Relation types distribution
        relation_types = features['relation_types'].squeeze().cpu().numpy()
        relation_probs = F.softmax(torch.tensor(relation_types), dim=-1).numpy()
        
        type_names = ['Spatial', 'Semantic', 'Functional']
        avg_probs = relation_probs.mean(axis=0)
        
        bars = axes[1, 1].bar(type_names, avg_probs, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[1, 1].set_ylabel('Average Probability', fontsize=12)
        axes[1, 1].set_title('Relation Type Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim(0, 1)
        
        for bar, prob in zip(bars, avg_probs):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{prob:.2f}', ha='center', fontsize=11, fontweight='bold')
        
        # Feature magnitude per layer
        feature_mags = {
            'Image': np.linalg.norm(features['image_features'].cpu().numpy().mean(axis=(0, 1))),
            'Object': np.linalg.norm(features['object_features'].cpu().numpy().mean(axis=(0, 1))),
            'Relation': np.linalg.norm(features['relation_features'].cpu().numpy().mean(axis=(0, 1))),
            'Text': np.linalg.norm(features['text_embeddings'].cpu().numpy().mean(axis=(0, 1)))
        }
        
        axes[1, 2].bar(feature_mags.keys(), feature_mags.values(), 
                       color=['#9b59b6', '#e67e22', '#1abc9c', '#34495e'])
        axes[1, 2].set_ylabel('Feature Magnitude (L2 Norm)', fontsize=12)
        axes[1, 2].set_title('Hierarchical Feature Magnitudes', fontsize=14, fontweight='bold')
        
        plt.suptitle('Q-Former Improved: Object Detection Visualization', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved visualization to {save_path}")
            
        return fig
    
    def visualize_attention_heatmap(
        self,
        image: Image.Image,
        features: dict,
        query_indices: List[int] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention heatmaps for specific object queries.
        
        Args:
            image: Original PIL image
            features: Output from extract_features()
            query_indices: Which queries to visualize (default: top 4 by confidence)
            save_path: Path to save the figure
        """
        confidence = features['object_confidence'].squeeze().cpu().numpy()
        
        if query_indices is None:
            query_indices = np.argsort(confidence)[-4:][::-1]
        
        num_queries = len(query_indices)
        fig, axes = plt.subplots(2, num_queries, figsize=(4*num_queries, 8))
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Get object features and compute attention-like scores
        obj_features = features['object_features'].squeeze().cpu().numpy()  # (num_queries, dim)
        img_features = features['image_features'].squeeze().cpu().numpy()  # (num_patches, dim)
        
        # Compute patch size (assuming square patches from ViT)
        num_patches = img_features.shape[0]
        patch_size = int(np.sqrt(num_patches))
        
        for idx, query_idx in enumerate(query_indices):
            # Top row: Original image with bbox
            ax_top = axes[0, idx] if num_queries > 1 else axes[0]
            ax_top.imshow(image)
            
            spatial = features['spatial_info'].squeeze().cpu().numpy()
            x, y, box_w, box_h = spatial[query_idx]
            
            rect = patches.Rectangle(
                (x * w - box_w * w/2, y * h - box_h * h/2),
                box_w * w, box_h * h,
                linewidth=3, edgecolor='red', facecolor='none'
            )
            ax_top.add_patch(rect)
            ax_top.set_title(f'Query {query_idx}\nConf: {confidence[query_idx]:.3f}',
                           fontsize=12, fontweight='bold')
            ax_top.axis('off')
            
            # Bottom row: Attention heatmap
            ax_bottom = axes[1, idx] if num_queries > 1 else axes[1]
            
            # Compute attention scores via dot product
            query_feat = obj_features[query_idx]
            attention_scores = np.dot(img_features, query_feat)
            attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
            
            # Reshape to spatial grid
            attention_map = attention_scores.reshape(patch_size, patch_size)
            
            # Resize to image size
            attention_map_resized = np.array(Image.fromarray(attention_map).resize((w, h), Image.BILINEAR))
            
            # Overlay on image
            ax_bottom.imshow(image)
            heatmap = ax_bottom.imshow(attention_map_resized, cmap='jet', alpha=0.5)
            ax_bottom.set_title(f'Attention Heatmap', fontsize=12)
            ax_bottom.axis('off')
            
        plt.colorbar(heatmap, ax=axes.ravel().tolist(), shrink=0.6, label='Attention Score')
        plt.suptitle('Object Query Attention Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved attention heatmap to {save_path}")
            
        return fig
    
    def visualize_hierarchical_features(
        self,
        features: dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize hierarchical feature representations using t-SNE or PCA.
        """
        from sklearn.decomposition import PCA
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract features
        obj_feat = features['object_features'].squeeze().cpu().numpy()
        rel_feat = features['relation_features'].squeeze().cpu().numpy()
        
        # PCA for object features
        if obj_feat.shape[0] > 2:
            pca = PCA(n_components=2)
            obj_2d = pca.fit_transform(obj_feat)
            
            confidence = features['object_confidence'].squeeze().cpu().numpy()
            scatter = axes[0].scatter(obj_2d[:, 0], obj_2d[:, 1], 
                                      c=confidence, cmap='viridis', s=100, alpha=0.7)
            plt.colorbar(scatter, ax=axes[0], label='Confidence')
            axes[0].set_title('Object Features (PCA)', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
        
        # PCA for relation features
        if rel_feat.shape[0] > 2:
            pca = PCA(n_components=2)
            rel_2d = pca.fit_transform(rel_feat)
            
            rel_types = features['relation_types'].squeeze().cpu().numpy()
            rel_labels = np.argmax(rel_types, axis=-1)
            
            scatter = axes[1].scatter(rel_2d[:, 0], rel_2d[:, 1],
                                      c=rel_labels, cmap='Set1', s=50, alpha=0.7)
            axes[1].set_title('Relation Features (PCA)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('PC1')
            axes[1].set_ylabel('PC2')
            
            # Add legend
            type_names = ['Spatial', 'Semantic', 'Functional']
            handles = [plt.scatter([], [], c=plt.cm.Set1(i/3), label=name) 
                      for i, name in enumerate(type_names)]
            axes[1].legend(handles=handles, title='Relation Type')
        
        # Feature norm comparison
        norms = {
            'Object': np.linalg.norm(obj_feat, axis=1).mean(),
            'Relation': np.linalg.norm(rel_feat, axis=1).mean(),
        }
        
        bars = axes[2].bar(norms.keys(), norms.values(), color=['#3498db', '#e74c3c'])
        axes[2].set_ylabel('Average L2 Norm')
        axes[2].set_title('Feature Norms by Level', fontsize=12, fontweight='bold')
        
        for bar, val in zip(bars, norms.values()):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', fontsize=10)
        
        plt.suptitle('Hierarchical Feature Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved hierarchical visualization to {save_path}")
            
        return fig


def visualize_sample(
    model,
    image_path: str,
    question: str,
    output_dir: str = 'visualizations',
    device: torch.device = torch.device('cuda')
):
    """
    Convenience function to visualize a single sample.
    
    Args:
        model: QFormerImproved model
        image_path: Path to the image
        question: Question about the image
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Get processor from model
    image_input = model.vision_encoder.processor(images=image, return_tensors="pt")
    image_input = {k: v.to(device) for k, v in image_input.items()}
    
    # Create visualizer
    visualizer = QFormerVisualizer(model, device)
    
    # Extract features
    print(f"Processing: {question}")
    features = visualizer.extract_features(image_input, question)
    
    # Generate visualizations
    sample_name = Path(image_path).stem
    
    # Object detection visualization
    fig1 = visualizer.visualize_object_attention(
        image, features,
        save_path=os.path.join(output_dir, f'{sample_name}_objects.png')
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
    
    plt.close('all')
    print(f"Visualizations saved to {output_dir}/")
    
    return features


if __name__ == "__main__":
    print("Visualization module loaded. Use visualize_sample() to generate visualizations.")

