import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable

class IntermediaryEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 16,
        activation: Callable = torch.relu,
        dropout: List[float] = [0, 0],
    ):
        super(IntermediaryEstimator, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.dense1 = nn.Linear(input_dim, 256)
        self.dense2 = nn.Linear(256, output_dim)
        self.sigmoid = nn.Sigmoid() # Sigmoid activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Intermediate prediction for bounding box coordinates.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Intermediate prediction for bounding box coordinates
        """
        x = F.dropout(x, self.dropout[0])
        x = self.activation(self.dense1(x))
        x = F.dropout(x, self.dropout[1])
        x = self.dense2(x)
        return self.sigmoid(x)