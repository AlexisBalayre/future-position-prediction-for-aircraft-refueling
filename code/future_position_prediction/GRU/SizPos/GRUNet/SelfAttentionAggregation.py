import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionAggregation(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SelfAttentionAggregation, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention aggregation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, input_dim)

        Returns:
            torch.Tensor: Aggregated tensor of shape (batch_size, hidden_dim)
        """
        # Compute attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)

        # Apply attention and sum
        attended = torch.sum(attention_weights * x, dim=1)
        return attended
