import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        # NOTE: LayerNorm removed - it's applied in CrossModalTransformerLayer

        self.attn_dropout = nn.Dropout(dropout)  # FIX: Use configurable dropout
        self.proj_dropout = nn.Dropout(dropout)  # FIX: Use configurable dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, query_len, _ = query.shape
        _, key_value_len, _ = key.shape

        # NOTE: LayerNorm is already applied in CrossModalTransformerLayer before calling this
        # Removed duplicate layer_norm here to prevent gradient vanishing

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        query = query.reshape(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, key_value_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, key_value_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            # Use large finite negative value instead of -inf to avoid NaN in softmax backward
            scores = scores.masked_fill(attention_mask == 1, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, self.dim)

        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output