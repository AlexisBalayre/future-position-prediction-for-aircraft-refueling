import torch
import torch.nn as nn
import torch.nn.functional as F

from .SelfAttention import SelfAttention


class DecoderGRU(nn.Module):
    """
    A GRU-based decoder model with an attention mechanism for predicting sequences.

    Args:
        input_dim (int): The number of expected features in the input (the last dimension of the input tensor).
        hidden_dim (int): The number of features in the hidden state h.
        output_dim (int): The number of expected features in the output.
        n_layers (int): Number of recurrent layers. E.g., setting `n_layers=2` would mean stacking two GRUs together.
        dropout (list of float, optional): Dropout probabilities for the linear layers. Default is [0, 0].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: list = [0, 0],
    ):
        super(DecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)

        # Dropout rates
        self.dropout = dropout

        # Linear layers for dense connections
        self.dense1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = torch.nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.relu = nn.ReLU()

        # Attention mechanism
        self.attention = SelfAttentionAggregation(hidden_dim, hidden_dim)

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
        Forward pass through the decoder GRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            h (torch.Tensor): Initial hidden state for the GRU of shape (n_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape (batch_size, output_dim) and the updated hidden state.
        """
        # GRU forward pass
        out, h = self.gru(x.unsqueeze(1), h)

        # Apply attention mechanism
        x = self.attention(out)

        # Dropout and linear layers
        x = F.dropout(out[:, -1], self.dropout[0])
        x = self.relu(self.dense1(x))
        x = F.dropout(x, self.dropout[1])
        out = self.dense2(x)

        return out, h
