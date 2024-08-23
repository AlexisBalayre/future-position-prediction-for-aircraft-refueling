import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from .SelfAttention import SelfAttention


class DecoderGRU(nn.Module):
    """
    A GRU-based decoder with an attention mechanism for sequence modelling tasks.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden state in the GRU.
        output_dim (int): Dimensionality of the output features.
        n_layers (int): Number of GRU layers.
        dropout (List[float], optional): Dropout rates for the intermediate layers. Default is [0, 0].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: List[float] = [0, 0],
    ):
        super(DecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)

        # Dropout and fully connected layers
        self.dropout = dropout
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # Self-attention layer
        self.attention = SelfAttention(hidden_dim, hidden_dim)

        # Initialise GRU weights
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialise weights for the GRU layers using appropriate methods.
        """
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            h (torch.Tensor): Hidden state tensor of shape (n_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor of shape (batch_size, output_dim) and the updated hidden state tensor.
        """
        # GRU forward pass
        out, h = self.gru(x.unsqueeze(1), h)

        # Apply self-attention mechanism
        x = self.attention(out)

        # Apply dropout, ReLU, and fully connected layers
        x = F.dropout(out[:, -1], self.dropout[0])
        x = self.relu(self.dense1(x))
        x = F.dropout(x, self.dropout[1])
        out = self.dense2(x)

        return out, h
