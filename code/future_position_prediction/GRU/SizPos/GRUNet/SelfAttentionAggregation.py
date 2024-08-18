import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionAggregation(nn.Module):
    """
    A self-attention aggregation module for sequence data.

    This module applies self-attention to the input sequence, computing attention weights
    that are used to aggregate the sequence elements into a single output tensor.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden state in the attention mechanism.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(SelfAttentionAggregation, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SelfAttentionAggregation module.

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
