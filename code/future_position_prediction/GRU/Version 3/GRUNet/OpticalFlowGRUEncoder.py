import torch
import torch.nn as nn

from typing import Tuple


class OpticalFlowGRUEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        num_layers: int = 1,
        image_size: Tuple[int, int] = (640, 480),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        input_dim = input_channels * image_size[0] * image_size[1]
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
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

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size, seq_len, -1)
        x = self.projection(x)
        out, h = self.gru(x, h)
        return out, h
