import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable


class BboxPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        activation: Callable = torch.relu,
        dropout: List[float] = [0, 0],
    ):
        super(BboxPredictor, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.dense1 = nn.Linear(input_dim, 256)
        self.dense2 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict bounding box coordinates.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Predicted bounding box coordinates
        """
        x = F.dropout(x[:, -1], self.dropout[0])
        x = self.activation(self.dense1(x))
        x = F.dropout(x, self.dropout[1])
        return self.dense2(x)
