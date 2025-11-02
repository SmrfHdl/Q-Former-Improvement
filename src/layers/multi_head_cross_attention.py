import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.layer_norm = nn.LayerNorm(dim)

        self.attn_dropout = nn.Dropout(0.3)
        self.proj_dropout = nn.Dropout(0.3)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor = None):
        batch_size, query_len, _ = query.shape
        _, key_value_len, _ = key.shape

        query = self.layer_norm(query)

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
            scores = scores.masked_fill(attention_mask == 1, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, self.dim)

        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output