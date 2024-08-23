import torch
import torch.nn as nn


class EncoderGRU(nn.Module):
    """
    A GRU-based encoder model for sequence-to-sequence tasks.

    Args:
        input_dim (int): The number of expected features in the input (the last dimension of the input tensor).
        hidden_dim (int): The number of features in the hidden state h.
        n_layers (int): Number of recurrent layers. E.g., setting `n_layers=2` would mean stacking two GRUs together.
        output_frames_nb (int, optional): Number of output frames to predict. Default is 10.
        dropout (float, optional): Dropout probability for the GRU. Default is 0.1.
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

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the GRU with specific methods for biases and weights.
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
        Forward pass through the GRU encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            h (torch.Tensor): Initial hidden state for the GRU of shape (n_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (batch_size, seq_len, hidden_dim) and the updated hidden state.
        """
        out, h = self.gru(x, h)
        return out, h
