import torch
import torch.nn as nn
from typing import Tuple


class EncoderGRU(nn.Module):
    """
    A GRU-based encoder for sequence modelling tasks.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden state in the GRU.
        n_layers (int): Number of GRU layers.
        output_frames_nb (int, optional): Number of output frames. Default is 10.
        dropout (float, optional): Dropout rate for the GRU layers. Default is 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_frames_nb: int = 10,
        dropout: float = 0.1,
    ):
        super(EncoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # GRU layer
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
        )

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
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
            h (torch.Tensor): Initial hidden state tensor of shape (n_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The output tensor of shape (batch_size, sequence_length, hidden_dim).
                - The final hidden state tensor of shape (n_layers, batch_size, hidden_dim).
        """
        out, h = self.gru(x, h)
        return out, h
