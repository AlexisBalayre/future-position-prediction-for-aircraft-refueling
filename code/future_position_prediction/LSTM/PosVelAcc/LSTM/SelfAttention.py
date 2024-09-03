import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class SelfAttention(nn.Module):
    """
    A self-attention mechanism that computes attention scores for each element in the sequence
    and applies these scores to aggregate information from the sequence.

    Args:
        input_dim (int): The dimensionality of the input feature space.
    """

    def __init__(self, input_dim: int):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, input_dim),
            where each element in the sequence has been weighted according to the attention mechanism.
        """
        # Compute query, key, and value matrices
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compute attention scores
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)

        # Apply softmax to obtain attention weights
        attention = self.softmax(scores)

        # Weight the values using the attention scores
        weighted = torch.bmm(attention, values)

        return weighted
