import torch
import torch.nn as nn
import torch.nn.functional as F

from GRUNet.SelfAttentionAggregation import SelfAttentionAggregation


class DecoderGRU(nn.Module):
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
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x.unsqueeze(1), h)
        x = self.attention(out)
        x = F.dropout(out[:, -1], self.dropout[0])
        x = self.relu(self.dense1(x))
        x = F.dropout(x, self.dropout[1])
        out = self.dense2(x)
        return out, h
