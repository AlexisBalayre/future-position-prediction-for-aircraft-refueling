import torch
import torch.nn as nn
import torch.nn.functional as F

from future_position_prediction.GRU.SizPos.GRUNet.SelfAttentionAggregation import (
    SelfAttentionAggregation,
)


class DecoderGRU(nn.Module):
    """
    A GRU-based decoder with self-attention for sequence-to-sequence tasks.

    This module takes an input sequence and a hidden state, processes them through a GRU layer,
    applies self-attention on the GRU outputs, and then passes the result through a series of dense layers
    to produce the final output.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden state in the GRU.
        output_dim (int): The dimensionality of the output features.
        n_layers (int): The number of GRU layers.
        dropout (list): A list containing two dropout probabilities. The first is applied after the GRU layer,
                        and the second is applied after the first dense layer.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0, 0]):
        super(DecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.attention = SelfAttentionAggregation(hidden_dim, hidden_dim)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the GRU and dense layers.
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
        Forward pass through the DecoderGRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            h (torch.Tensor): Hidden state tensor of shape (n_layers, batch_size, hidden_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor of shape (batch_size, output_dim)
                                               and the updated hidden state tensor.
        """
        out, h = self.gru(x.unsqueeze(1), h)
        x = self.attention(out)
        x = F.dropout(out[:, -1], self.dropout[0])
        x = self.relu(self.dense1(x))
        x = F.dropout(x, self.dropout[1])
        out = self.dense2(x)
        return out, h
