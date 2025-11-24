import torch
import torch.nn as nn
from layers.multi_head_cross_attention import MultiHeadCrossAttentionLayer
from layers.multi_head_self_attention import MultiHeadSelfAttentionLayer
from layers.feed_forward import FeedForwardLayer

from torch.nn.init import trunc_normal_


class CrossModalTransformerLayer(nn.Module):
    """
    A transformer layer for cross-modal processing.

    Includes cross-attention between queries and image features,
    self-attention on the concatenated queries and text embeddings,
    and a feed-forward network with layer normalization and residual connections.
    """
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.norm_q_cross = nn.LayerNorm(dim)
        self.norm_kv_cross = nn.LayerNorm(dim)
        self.mhca = MultiHeadCrossAttentionLayer(dim, num_heads)
        self.dropout_cross = nn.Dropout(dropout)

        self.norm_self = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSelfAttentionLayer(dim, num_heads)
        self.dropout_self = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForwardLayer(dim, ffn_expansion_factor, dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, image_features: torch.Tensor, text_embeddings: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Process one layer of the cross-modal transformer.

        Args:
            queries (torch.Tensor): Query tensor of shape batch_size, num_queries, dim.
            image_features (torch.Tensor): Image feature tensor of shape batch_size, num_patches, dim.
            text_embeddings (torch.Tensor): Text embedding tensor of shape batch_size, text_length, dim.
            attention_mask (torch.Tensor, optional): Mask for self-attention of shape batch_size, seq_len + text_len, seq_len + text_len.

        Returns:
            tuple torch.Tensor: Updated queries and text_embeddings.
        """
        query_norm = self.norm_q_cross(queries)
        key_value_norm = self.norm_kv_cross(image_features)

        queries = queries + self.dropout_cross(
            self.mhca(query_norm, key_value_norm, key_value_norm))
        
        combined = torch.cat([queries, text_embeddings], dim=1)

        combined_norm = self.norm_self(combined)

        combined = combined + self.dropout_self(
            self.mhsa(combined_norm, attention_mask))
        
        combined_norm = self.norm_ff(combined)
        combined = combined + self.dropout_ff(
            self.ff(combined_norm))
        
        updated_queries, updated_text_embeddings = torch.split(
            combined, [queries.size(1), text_embeddings.size(1)], dim=1)
        
        return updated_queries, updated_text_embeddings
    

class CrossModalTransformer(nn.Module):
    """
    Cross Modal Transformer that processes information from multiple modalities.

    This transformer performs cross-attention between query embeddings and image features,
    followed by self-attention with text embeddings to integrate information across modalities.
    The model consists of multiple stacked transformer layers, each containing cross-attention,
    self-attention, and feed-forward components with normalization and residual connections.
    """
    def __init__(self, dim: int, num_heads: int, num_layers: int, ffn_expansion_factor: int = 4, dropout: float = 0.1):
        """
        Args:
            dim (int): Dimensionality of the input features (also used for the output dimensions).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers to stack.
            ffn_expansion_factor (int, optional): Expansion factor for the feed-forward network. Default is 4.
            dropout (float, optional): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(dim, num_heads, ffn_expansion_factor, dropout)
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(dim)

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        """
        Initialize weights for the model.
        
        For linear layers, uses truncated normal initialization.
        For layer normalization layers, initializes biases to zero and weights to one.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, queries: torch.Tensor, image_features: torch.Tensor, text_embeddings: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Process multiple modalities through the Cross Modal Transformer.

        Args:
            queries (torch.Tensor): Query tensor of shape (batch_size, num_queries, dim).
            image_features (torch.Tensor): Visual features extracted from an image (batch_size, num_patches, dim).
            text_embeddings (torch.Tensor): Text embedding tensor of shape (batch_size, text_length, dim).
            attention_mask (torch.Tensor, optional): Mask for self-attention of shape (batch_size, seq_len, seq_len).
        """
        for layer in self.layers:
            queries, text_embeddings = layer(
                queries, image_features, text_embeddings, attention_mask)
            
        queries = self.final_layer_norm(queries)
        return queries, text_embeddings