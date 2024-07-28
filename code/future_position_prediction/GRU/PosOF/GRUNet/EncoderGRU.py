import torch
import torch.nn as nn


class EncoderGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        output_frames_nb: int = 10,
        dropout: float = 0.1
    ):
        super(EncoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout
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

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h
