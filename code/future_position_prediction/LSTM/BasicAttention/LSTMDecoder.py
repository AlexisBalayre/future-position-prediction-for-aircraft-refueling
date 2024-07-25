import torch
import torch.nn as nn
from typing import Tuple


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        output_activation: callable = None,
        dropout: float = 0.0,
    ):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        # Remove the attention mechanism from here
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = output_activation
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.unsqueeze(1)  # Add a dimension for the sequence length

        # Concatenate input with context
        lstm_input = torch.cat((x, context), dim=2)

        # LSTM forward pass
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = self.dropout(output)

        # Fully connected layers
        output = self.relu(self.fc1(output.squeeze(1)))
        prediction = self.fc2(output)

        if self.output_activation:
            prediction = self.output_activation(prediction)

        return prediction, (hidden, cell)
