import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


from .SelfAttention import SelfAttention


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        output_activation: callable = None,
        dropout: float = 0.0,
    ):
        """
        Initialise the LSTM Decoder.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in LSTM.
            output_size (int): Number of output features.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            output_activation (callable, optional): Activation function for the output. Defaults to None.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
        """
        super(LSTMDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Hidden layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer
        self.output_activation = output_activation  # Output activation function
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.hidden_size = hidden_size  # Hidden size
        self.num_layers = num_layers  # Number of layers

        # Define Attention layer
        self.attention = SelfAttention(hidden_size)

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x_input: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_cell_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the LSTM Decoder.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_size).
            encoder_hidden_states (torch.Tensor): Hidden states of the encoder.
            encoder_cell_states (torch.Tensor): Cell states of the encoder.

        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_size).
            Tuple[torch.Tensor, torch.Tensor]: Hidden state and cell state of the decoder.
        """
        # Propagate input through LSTM
        output, (hidden_states, cell_states) = self.lstm(
            x_input.unsqueeze(1), (encoder_hidden_states, encoder_cell_states)
        )

        # Apply attention
        context_vector = self.attention(output)

        output = context_vector[:, -1, :]  # Using the last output for prediction
        output = self.relu(self.fc1(output))  # First Dense + ReLU
        output = self.dropout(output)  # Dropout
        output = self.fc2(output)  # Final Output

        if self.output_activation is not None:
            output = self.output_activation(
                output
            )  # Apply output activation if provided

        return output, (
            hidden_states,
            cell_states,
        )  # return final output and hidden state
