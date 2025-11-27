import torch
import torch.nn as nn
from layers.multi_head_cross_attention import MultiHeadCrossAttentionLayer
from layers.multi_head_self_attention import MultiHeadSelfAttentionLayer
from layers.feed_forward import FeedForwardLayer

from torch.nn.init import trunc_normal_


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False):
    """
    Stochastic Depth (drop path) for regularization.
    Drops entire residual branch with probability drop_prob during training.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Stochastic Depth module wrapper."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class CrossModalTransformerLayer(nn.Module):
    """
    A transformer layer for cross-modal processing with stochastic depth.

    Includes cross-attention between queries and image features,
    self-attention on the concatenated queries and text embeddings,
    and a feed-forward network with layer normalization and residual connections.
    """
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: int = 4, dropout: float = 0.1, drop_path_rate: float = 0.0):
        super().__init__()
        
        self.norm_q_cross = nn.LayerNorm(dim)
        self.norm_kv_cross = nn.LayerNorm(dim)
        self.mhca = MultiHeadCrossAttentionLayer(dim, num_heads, dropout=dropout)
        self.dropout_cross = nn.Dropout(dropout)

        self.norm_self = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSelfAttentionLayer(dim, num_heads)
        self.dropout_self = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForwardLayer(dim, dim * ffn_expansion_factor, dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        # v2 IMPROVEMENT: Stochastic depth for regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

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

        # v2 IMPROVEMENT: Apply stochastic depth to residual connections
        queries = queries + self.drop_path(self.dropout_cross(
            self.mhca(query_norm, key_value_norm, key_value_norm)))
        
        combined = torch.cat([queries, text_embeddings], dim=1)

        combined_norm = self.norm_self(combined)

        combined = combined + self.drop_path(self.dropout_self(
            self.mhsa(combined_norm, attention_mask)))
        
        combined_norm = self.norm_ff(combined)
        combined = combined + self.drop_path(self.dropout_ff(
            self.ff(combined_norm)))
        
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
    
    v2 IMPROVEMENT: Added stochastic depth for better regularization.
    """
    def __init__(self, dim: int, num_heads: int, num_layers: int, ffn_expansion_factor: int = 4, dropout: float = 0.1, stochastic_depth_rate: float = 0.0):
        """
        Args:
            dim (int): Dimensionality of the input features (also used for the output dimensions).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers to stack.
            ffn_expansion_factor (int, optional): Expansion factor for the feed-forward network. Default is 4.
            dropout (float, optional): Dropout rate. Default is 0.1.
            stochastic_depth_rate (float, optional): Maximum drop path rate. Linearly increases from 0 to this value across layers. Default is 0.0.
        """
        super().__init__()
        self.num_layers = num_layers

        # v2 IMPROVEMENT: Linearly increase drop path rate across layers
        # This is a common technique from papers like DeiT
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(dim, num_heads, ffn_expansion_factor, dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)
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