import torch
import torch.nn as nn
from torch import transpose, matmul, softmax


def multi_head_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor = None):
    """
    Compute multi-head attention.
    
    Args:
        query (torch.Tensor): The query embeddings of shape (batch_size, num_heads, sequence_length, head_dim).
        key (torch.Tensor): The key embeddings of shape (batch_size, num_heads, sequence_length, head_dim).
        value (torch.Tensor): The value embeddings of shape (batch_size, num_heads, sequence_length, head_dim).
        attention_mask (torch.Tensor, optional): A mask to indicate where attention should not be applied, 
            shape (batch_size, sequence_length + text_length, sequence_length + text_length).

    Returns:
        output (torch.Tensor): The output embeddings of shape (batch_size, sequence_length, num_heads * head_dim).
    """
    batch_size, num_heads, seq_len, head_dim = query.shape

    k_transposed = transpose(key, -2, -1) # (batch_size, num_heads, head_dim, sequence_length)
    attention_mat = matmul(query, k_transposed) / (head_dim ** 0.5) # (batch_size, num_heads, sequence_length, sequence_length)

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, length_q, length_kv)
        attention_mask = attention_mask.expand(-1, num_heads, -1, -1)  # (batch_size, num_heads, length_q, length_kv)
        attention_mat = attention_mat.masked_fill(attention_mask == 1, float('-inf')) # (batch_size, num_heads, length_q, length_kv)

    attention_mat = softmax(attention_mat, dim=-1)  # (batch_size, num_heads, sequence_length, sequence_length)

    mat = matmul(attention_mat, value)  # (batch_size, num_heads, sequence_length, head_dim)
    output = mat.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)  # (batch_size, sequence_length, num_heads * head_dim)
    return output


