import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialise the LSTM Encoder.

        Args:
            input_size (int): Number of features in the input X.
            hidden_size (int): Number of features in the hidden state h.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer. Defaults to 0.0.
        """
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, x_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the LSTM Encoder.

        Args:
            x_input (torch.Tensor): Input of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: LSTM output giving all the hidden states in the sequence.
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the hidden states and cell states.
        """
        output, (hidden_states, cell_states) = self.lstm(x_input)

        return output, (hidden_states, cell_states)

    def init_hidden(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise hidden state.

        Args:
            batch_size (int): Batch size.
            device (torch.device, optional): The device on which to Initialise the hidden state. Defaults to CPU.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized hidden state and cell state.
        """
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )
