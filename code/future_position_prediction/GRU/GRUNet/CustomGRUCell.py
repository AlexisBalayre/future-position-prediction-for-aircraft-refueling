import torch
import torch.nn as nn
import numpy as np


class CustomGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Define linear transformations
        self.input_to_hidden = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.hidden_to_hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.bbox_to_hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the parameters with uniform distribution."""
        std = 1.0 / np.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-std, std)

    def forward(
        self, input: torch.Tensor, hidden_state: torch.Tensor, bbox: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the CustomGRUCell.

        Args:
            input (torch.Tensor): Input tensor
            hidden_state (torch.Tensor): Previous hidden state
            bbox (torch.Tensor): Bounding box information

        Returns:
            torch.Tensor: New hidden state
        """
        if hidden_state is None:
            hidden_state = torch.zeros(
                input.size(0), self.hidden_size, device=input.device
            )

        # Compute gate inputs
        gate_input = self.input_to_hidden(input)
        gate_hidden = self.hidden_to_hidden(hidden_state)
        gate_bbox = self.bbox_to_hidden(bbox)

        # Split gates
        reset_input, update_input, new_input = gate_input.chunk(3, 1)
        reset_hidden, update_hidden, new_hidden = gate_hidden.chunk(3, 1)
        reset_bbox, update_bbox, new_bbox = gate_bbox.chunk(3, 1)

        # Compute gate values
        reset_gate = torch.sigmoid(reset_input + reset_hidden + reset_bbox)
        update_gate = torch.sigmoid(update_input + update_hidden + update_bbox)
        new_gate = torch.tanh(new_input + new_bbox + (reset_gate * new_hidden))

        # Compute new hidden state
        new_hidden_state = update_gate * hidden_state + (1 - update_gate) * new_gate

        return new_hidden_state
