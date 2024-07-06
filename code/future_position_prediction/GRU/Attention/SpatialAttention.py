import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SpatialAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weight parameter."""
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(
        self, hidden_states: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply spatial attention to hidden states of all objects in a frame.

        Args:
            hidden_states (Dict[str, torch.Tensor]): Dictionary of hidden states for each object

        Returns:
            Dict[str, torch.Tensor]: Dictionary of attended hidden states
        """
        keys = []
        values = []
        for key, value in hidden_states.items():
            keys.append(key)
            values.append(value)

        if values:
            hidden = torch.cat(values, dim=1)
            tanh_hidden = torch.tanh(hidden)
            attention_weights = torch.softmax(torch.matmul(tanh_hidden, self.weight), 1)
            attended_hidden = torch.mul(hidden, attention_weights)

            attended_states = {}
            for i, key in enumerate(keys):
                attended_states[key] = attended_hidden[:, i, :].unsqueeze(1)

            return attended_states
        else:
            return hidden_states
