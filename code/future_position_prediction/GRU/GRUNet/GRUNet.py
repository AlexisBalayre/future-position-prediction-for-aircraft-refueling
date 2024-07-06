import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from CustomGRUCell import CustomGRUCell
from IntermediaryEstimator import IntermediaryEstimator


class GRUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        output_cor_dim: int,
        time_steps: int,
        bias: bool = True,
    ):
        super(GRUNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.dropout_rates = [0, 0]
        self.output_cor_dim = output_cor_dim

        self.gru_cell = CustomGRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bounding_box_predictor = IntermediaryEstimator(
            hidden_dim, 4 * time_steps, dropout=self.dropout_rates
        )

        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the GRU cell."""
        for name, param in self.gru_cell.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor,
        bbox: torch.Tensor,
        dist_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GRUNet.

        Args:
            x (torch.Tensor): Input tensor
            hidden_state (torch.Tensor): Previous hidden state
            bbox (torch.Tensor): Bounding box information
            dist_out (torch.Tensor): Distance output

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output, new hidden state, and predicted bounding box
        """
        x = torch.squeeze(x, 0)
        hidden_state = torch.squeeze(hidden_state, 0)
        bbox = torch.squeeze(bbox, 0)

        hidden_state = self.gru_cell(x, hidden_state, bbox)
        out = self.fc(hidden_state)
        hidden_state = hidden_state.unsqueeze(0)
        out = out.unsqueeze(0)
        out = torch.cat([out, dist_out], dim=-1)
        predicted_bbox = self.bounding_box_predictor(out)
        out = F.dropout(out[:, -1], self.dropout_rates[0])

        return out, hidden_state, predicted_bbox
