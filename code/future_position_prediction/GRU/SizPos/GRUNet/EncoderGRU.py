import torch
import torch.nn as nn


class EncoderGRU(nn.Module):
    """
    A GRU-based encoder for sequence-to-sequence tasks.

    This module takes an input sequence and a hidden state, processes them through a GRU layer,
    and returns the GRU outputs and the updated hidden state.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden state in the GRU.
        n_layers (int): The number of GRU layers.
        output_frames_nb (int, optional): Number of output frames. Defaults to 10.
        dropout (float, optional): Dropout probability applied to the GRU. Defaults to 0.1.
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
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
        )
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the GRU layer.

        Biases are initialized to zero, input-hidden weights are initialized using the Kaiming normal method,
        and hidden-hidden weights are initialized using orthogonal initialization.
        """
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        """
        Forward pass through the EncoderGRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).
            h (torch.Tensor): Hidden state tensor of shape (n_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The GRU output tensor of shape (batch_size, sequence_length, hidden_dim)
                                               and the updated hidden state tensor of shape (n_layers, batch_size, hidden_dim).
        """
        out, h = self.gru(x, h)
        return out, h
