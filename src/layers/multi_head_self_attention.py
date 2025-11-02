from functionals.multi_head_attention import multi_head_attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        """
        Multi-Head Self-Attention Layer.

        Args:
            dim (int): The dimensionality of the input features (Dy=Dq=Dk=Dv=Dx).
            num_heads (int): The number of attention heads.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.linear_qkv = nn.Linear(dim, dim * 3)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Compute the multi-head self-attention of the input tensor 'x'.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, sequence_length, sequence_length).

        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, sequence_length, dim).
        """
        batch_size, sequence_length, dim = x.shape

        qkv = self.linear_qkv(x) # (batch_size, sequence_length, dim * 3)

        qkv = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, -1)

        qkv = qkv.permute(0, 3, 2, 1, 4) # (batch_size, num_heads, 3, sequence_length, head_dim=dim//num_heads)

        query, key, value = torch.unbind(qkv, dim=2) # each of shape (batch_size, num_heads, sequence_length, head_dim)

        attn_output = multi_head_attention(query, key, value, attention_mask) # (batch_size, sequence_length, dim)

        y = self.linear(attn_output) # (batch_size, sequence_length, dim)

        return y