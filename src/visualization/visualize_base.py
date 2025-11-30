"""
Visualization module for Q-Former Base architecture.
Visualizes attention maps and query representations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import seaborn as sns
from typing import Optional, List
import os


class QFormerBaseVisualizer:
    """Visualize Q-Former Base attention and learned queries."""
    
    def __init__(self, model, device: torch.device = torch.device('cuda')):
        self.model = model
        self.device = device
        self.model.eval()
        
    @torch.no_grad()
    def extract_features(self, image_input: dict, question: str) -> dict:
        """
        Extract intermediate features from the Q-Former Base model.
        """
        self.model.eval()
        
        batch_size = 1
        
        # Get image features
        image_features = self.model.vision_encoder.encode(image_input)
        image_features_proj = self.model.vision_projection(image_features)
        image_features_proj = self.model.vision_norm(image_features_proj)
        
        # Get text embeddings  
        question_output, question_tokens = self.model.encode_text([question])
        text_embeddings_raw = question_output['last_hidden_state']
        text_embeddings_proj = self.model.text_projection(text_embeddings_raw)
        text_embeddings_proj = self.model.text_norm(text_embeddings_proj)
        
        # Get EOS token embedding for ITC
        if self.model.use_clip_for_text:
            eos_token_id = 49407
            eos_positions = (question_tokens['input_ids'] == eos_token_id).int().argmax(dim=-1)
            batch_indices = torch.arange(batch_size, device=self.device)
            cls_text_embedding = text_embeddings_proj[batch_indices, eos_positions, :]
        else:
            cls_text_embedding = text_embeddings_proj[:, 0, :]
        
        cls_text_normalized = F.normalize(cls_text_embedding, p=2, dim=-1)
        
        # Get learned queries
        queries = self.model.learned_queries.expand(batch_size, -1, -1).clone()
        
        # Generate attention mask
        attention_mask = self.model.generate_attention_mask(
            task='itc',
            query_len=queries.shape[1],
            pad_mask=question_tokens['attention_mask'],
            device=self.device
        )
        
        # Run cross-modal transformer
        queries_output, _ = self.model.cross_modal_transformer(
            queries,
            image_features_proj,
            text_embeddings=text_embeddings_proj,
            attention_mask=attention_mask
        )
        
        # Normalize queries for visualization
        queries_normalized = F.normalize(queries_output, p=2, dim=-1)
        
        # Compute similarity between queries and image patches
        # queries: (1, num_queries, dim), image_features: (1, num_patches, dim)
        query_image_similarity = torch.einsum('bqd,bpd->bqp', 
                                              queries_normalized, 
                                              F.normalize(image_features_proj, p=2, dim=-1))
        
        # Compute similarity between queries and text
        query_text_similarity = torch.einsum('bqd,bd->bq',
                                             queries_normalized,
                                             cls_text_normalized)
        
        return {
            'image_features': image_features_proj,
            'text_embeddings': text_embeddings_proj,
            'cls_text_embedding': cls_text_embedding,
            'learned_queries': queries,
            'queries_output': queries_output,
            'query_image_similarity': query_image_similarity,
            'query_text_similarity': query_text_similarity,
            'attention_mask': attention_mask
        }
    
    def visualize_query_attention(
        self,
        image: Image.Image,
        features: dict,
        save_path: Optional[str] = None,
        top_k: int = 8
    ) -> plt.Figure:
        """
        Visualize which image regions each query attends to.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Query-Text similarity
        query_text_sim = features['query_text_similarity'].squeeze().cpu().numpy()
        top_indices = np.argsort(query_text_sim)[-top_k:][::-1]
        
        colors = ['steelblue'] * len(query_text_sim)
        for idx in top_indices:
            colors[idx] = 'crimson'
            
        axes[0, 1].bar(range(len(query_text_sim)), query_text_sim, color=colors, alpha=0.8)
        axes[0, 1].set_xlabel('Query Index', fontsize=12)
        axes[0, 1].set_ylabel('Similarity to Text', fontsize=12)
        axes[0, 1].set_title(f'Query-Text Similarity (Top {top_k} highlighted)', fontsize=14, fontweight='bold')
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Query feature similarity matrix
        queries = features['queries_output'].squeeze().cpu().numpy()
        query_similarity = np.corrcoef(queries)
        
        sns.heatmap(query_similarity, ax=axes[0, 2], cmap='RdBu_r', center=0,
                    square=True, cbar_kws={'shrink': 0.8})
        axes[0, 2].set_title('Query-Query Similarity', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Query Index', fontsize=12)
        axes[0, 2].set_ylabel('Query Index', fontsize=12)
        
        # Average attention heatmap
        query_image_sim = features['query_image_similarity'].squeeze().cpu().numpy()
        
        # Get patch grid size
        num_patches = query_image_sim.shape[1]
        patch_size = int(np.sqrt(num_patches))
        
        # Average across all queries
        avg_attention = query_image_sim.mean(axis=0)
        avg_attention = (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min() + 1e-8)
        attention_map = avg_attention.reshape(patch_size, patch_size)
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        attention_resized = np.array(Image.fromarray(attention_map).resize((w, h), Image.BILINEAR))
        
        axes[1, 0].imshow(image)
        heatmap = axes[1, 0].imshow(attention_resized, cmap='jet', alpha=0.5)
        axes[1, 0].set_title('Average Query Attention', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(heatmap, ax=axes[1, 0], shrink=0.8)
        
        # Top query attention
        top_query_idx = top_indices[0]
        top_attention = query_image_sim[top_query_idx]
        top_attention = (top_attention - top_attention.min()) / (top_attention.max() - top_attention.min() + 1e-8)
        top_attention_map = top_attention.reshape(patch_size, patch_size)
        top_attention_resized = np.array(Image.fromarray(top_attention_map).resize((w, h), Image.BILINEAR))
        
        axes[1, 1].imshow(image)
        heatmap2 = axes[1, 1].imshow(top_attention_resized, cmap='jet', alpha=0.5)
        axes[1, 1].set_title(f'Top Query #{top_query_idx} Attention', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(heatmap2, ax=axes[1, 1], shrink=0.8)
        
        # Feature norms
        feature_norms = {
            'Image': np.linalg.norm(features['image_features'].cpu().numpy(), axis=-1).mean(),
            'Text': np.linalg.norm(features['text_embeddings'].cpu().numpy(), axis=-1).mean(),
            'Queries (init)': np.linalg.norm(features['learned_queries'].cpu().numpy(), axis=-1).mean(),
            'Queries (out)': np.linalg.norm(features['queries_output'].cpu().numpy(), axis=-1).mean(),
        }
        
        bars = axes[1, 2].bar(feature_norms.keys(), feature_norms.values(),
                              color=['#9b59b6', '#34495e', '#3498db', '#e74c3c'])
        axes[1, 2].set_ylabel('L2 Norm', fontsize=12)
        axes[1, 2].set_title('Feature Magnitudes', fontsize=14, fontweight='bold')
        axes[1, 2].tick_params(axis='x', rotation=15)
        
        for bar, val in zip(bars, feature_norms.values()):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{val:.2f}', ha='center', fontsize=9)
        
        plt.suptitle('Q-Former Base: Query Attention Visualization', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved to {save_path}")
            
        return fig
    
    def visualize_individual_queries(
        self,
        image: Image.Image,
        features: dict,
        query_indices: List[int] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention maps for individual queries.
        """
        query_text_sim = features['query_text_similarity'].squeeze().cpu().numpy()
        
        if query_indices is None:
            # Select top 4 queries by text similarity
            query_indices = np.argsort(query_text_sim)[-4:][::-1]
        
        num_queries = len(query_indices)
        fig, axes = plt.subplots(2, num_queries, figsize=(4*num_queries, 8))
        
        query_image_sim = features['query_image_similarity'].squeeze().cpu().numpy()
        num_patches = query_image_sim.shape[1]
        patch_size = int(np.sqrt(num_patches))
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        for idx, query_idx in enumerate(query_indices):
            # Top row: Image with text similarity score
            ax_top = axes[0, idx] if num_queries > 1 else axes[0]
            ax_top.imshow(image)
            ax_top.set_title(f'Query {query_idx}\nText Sim: {query_text_sim[query_idx]:.3f}',
                           fontsize=12, fontweight='bold')
            ax_top.axis('off')
            
            # Bottom row: Attention heatmap
            ax_bottom = axes[1, idx] if num_queries > 1 else axes[1]
            
            attention = query_image_sim[query_idx]
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            attention_map = attention.reshape(patch_size, patch_size)
            attention_resized = np.array(Image.fromarray(attention_map).resize((w, h), Image.BILINEAR))
            
            ax_bottom.imshow(image)
            heatmap = ax_bottom.imshow(attention_resized, cmap='jet', alpha=0.5)
            ax_bottom.set_title('Attention Map', fontsize=12)
            ax_bottom.axis('off')
            
        plt.colorbar(heatmap, ax=axes.ravel().tolist(), shrink=0.6, label='Attention')
        plt.suptitle('Individual Query Attention Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved to {save_path}")
            
        return fig


def visualize_base_model(
    model,
    image_path: str,
    question: str,
    output_dir: str = 'visualizations',
    device: torch.device = torch.device('cuda')
):
    """
    Convenience function to visualize Q-Former Base model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Process image
    image_input = model.vision_encoder.processor(images=image, return_tensors="pt")
    image_input = {k: v.to(device) for k, v in image_input.items()}
    
    # Create visualizer
    visualizer = QFormerBaseVisualizer(model, device)
    
    # Extract features
    print(f"\nProcessing: {question}")
    features = visualizer.extract_features(image_input, question)
    
    # Print summary
    query_text_sim = features['query_text_similarity'].squeeze().cpu().numpy()
    print(f"\nQuery-Text Similarity Summary:")
    print(f"   - Num queries: {len(query_text_sim)}")
    print(f"   - Max similarity: {query_text_sim.max():.3f}")
    print(f"   - Mean similarity: {query_text_sim.mean():.3f}")
    print(f"   - Top query index: {query_text_sim.argmax()}")
    
    # Generate visualizations
    sample_name = Path(image_path).stem
    
    print(f"\nGenerating visualizations...")
    
    fig1 = visualizer.visualize_query_attention(
        image, features,
        save_path=os.path.join(output_dir, f'{sample_name}_queries.png')
    )
    
    fig2 = visualizer.visualize_individual_queries(
        image, features,
        save_path=os.path.join(output_dir, f'{sample_name}_individual.png')
    )
    
    plt.close('all')
    
    print(f"\nVisualizations saved to: {output_dir}/")
    
    return features

