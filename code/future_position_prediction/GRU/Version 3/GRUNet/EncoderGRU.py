import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from GRUNet.IntermediaryEstimator import IntermediaryEstimator


class EncoderGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_frames_nb: int = 10,
        dropout: float = 0.1,
        activation: Callable = F.relu,
    ):
        super(EncoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
        )
        self.fc_flow = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_input = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.intermediary_estimator = IntermediaryEstimator(
            input_dim=2 * hidden_dim + input_dim,
            output_dim=4 * output_frames_nb,
            activation=activation,
            dropout=[dropout, dropout],
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h, encoded_flow):
        x = self.fc_input(x)
        out, h = self.gru(x, h)
        last_hidden = out[:, -1]
        processed_flow = self.fc_flow(encoded_flow)
        combined = torch.cat([last_hidden, x[:, -1], processed_flow], dim=-1)
        future_velocities = self.intermediary_estimator(combined)
        return out, h, future_velocities
