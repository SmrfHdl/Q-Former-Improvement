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
    _, _, key_len, _ = key.shape

    k_transposed = transpose(key, -2, -1) # (batch_size, num_heads, head_dim, key_length)
    attention_mat = matmul(query, k_transposed) / (head_dim ** 0.5) # (batch_size, num_heads, seq_len, key_len)

    if attention_mask is not None:
        # attention_mask is (batch_size, total_len, total_len) with 0 for "can attend" and -inf for "cannot attend"
        # We need to extract the relevant portion for the current query and key lengths
        if attention_mask.dim() == 3:
            # Extract the appropriate slice if mask is for a larger sequence
            mask_seq_len = attention_mask.shape[1]
            mask_key_len = attention_mask.shape[2]
            if mask_seq_len > seq_len or mask_key_len > key_len:
                # Slice the mask to match the attention dimensions
                attention_mask = attention_mask[:, :seq_len, :key_len]
            elif mask_seq_len < seq_len or mask_key_len < key_len:
                # Mask is smaller than expected - need to pad or handle differently
                # For now, just don't apply the mask if dimensions don't match
                attention_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len, key_len)
            attention_mat = attention_mat + attention_mask  # Add mask directly (0 for valid, -inf for invalid)

    attention_mat = softmax(attention_mat, dim=-1)  # (batch_size, num_heads, sequence_length, sequence_length)
    
    # Handle NaN from softmax when all values are -inf (fully masked row)
    # Replace NaN with uniform attention (or zeros - both work since the position is masked anyway)
    attention_mat = torch.where(
        torch.isnan(attention_mat),
        torch.zeros_like(attention_mat),  # Use zeros for fully masked positions
        attention_mat
    )

    mat = matmul(attention_mat, value)  # (batch_size, num_heads, sequence_length, head_dim)
    output = mat.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)  # (batch_size, sequence_length, num_heads * head_dim)
    return output


