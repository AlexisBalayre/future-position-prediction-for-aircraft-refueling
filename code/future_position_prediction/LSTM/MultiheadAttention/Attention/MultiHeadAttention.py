import torch
from torch import nn

from Attention.AttentionLayer import AttentionLayer

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = AttentionLayer(hidden_dim, num_heads)

    def forward(self, query, key, value):
        return self.attention(query, key, value)