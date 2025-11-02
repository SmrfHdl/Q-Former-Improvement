import torch.nn as nn
import torch

class FeedForwardLayer(nn.Module):
    """
    Feed Forward Network with 2 linear layers and a GELU activation.
    Used in transformer blocks for point-wise transformations.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Feed Forward Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, dim).
        """
        return self.net(x)