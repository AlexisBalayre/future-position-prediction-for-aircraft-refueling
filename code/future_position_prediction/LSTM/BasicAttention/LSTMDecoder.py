import torch
import torch.nn as nn

from Attention import Attention


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
        """
        Initialize the LSTM Decoder with attention mechanism.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in LSTM.
            output_dim (int): Number of output features.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            output_activation (callable, optional): Activation function for the output. Defaults to None.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
        """
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.attention = Attention(hidden_dim)
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
        encoder_outputs: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass of the LSTM Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            hidden (torch.Tensor): Hidden state of shape (num_layers, batch_size, hidden_dim).
            cell (torch.Tensor): Cell state of shape (num_layers, batch_size, hidden_dim).
            encoder_outputs (torch.Tensor): Encoder outputs of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_dim).
            torch.Tensor: Updated hidden state of shape (num_layers, batch_size, hidden_dim).
            torch.Tensor: Updated cell state of shape (num_layers, batch_size, hidden_dim).
        """
        x = x.unsqueeze(1)  # Add a dimension for the sequence length
        attn_weights = self.attention(
            hidden, encoder_outputs
        )  # Calculate attention weights
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        )  # Calculate context vector
        lstm_input = torch.cat(
            (x, context), dim=2
        )  # Concatenate context vector with input
        output, (hidden, cell) = self.lstm(
            lstm_input, (hidden, cell)
        )  # LSTM forward pass
        output = self.dropout(output)  # Apply dropout
        output = self.relu(
            self.fc1(output.squeeze(1))
        )  # Fully connected layer with ReLU activation
        prediction = self.fc2(output)  # Final fully connected layer
        if self.output_activation:
            prediction = self.output_activation(
                prediction
            )  # Apply output activation if specified
        return prediction, hidden, cell
