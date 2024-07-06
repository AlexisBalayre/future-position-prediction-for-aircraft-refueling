import torch
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attended, _ = self.attention(query, key, value)
        x = self.norm1(query + attended)
        return self.norm2(x + self.feed_forward(x))

