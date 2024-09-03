import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict


class SelfAttention(nn.Module):
    """
    A self-attention mechanism for aggregating information across time steps in a sequence.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Attention mechanism layer to compute attention weights
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention aggregation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, input_dim).

        Returns:
            torch.Tensor: Aggregated tensor of shape (batch_size, hidden_dim).
        """
        # Compute attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)

        # Apply attention and sum
        attended = torch.sum(attention_weights * x, dim=1)
        return attended
