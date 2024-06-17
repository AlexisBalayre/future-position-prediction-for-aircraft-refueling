import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        input_dim,
        num_transformer_layers,
        transformer_heads,
        model_dim,
        forward_expansion,
        output_dim,
    ):
        super(TransformerDecoderModel, self).__init__()

        # Embedding layer to increase dimensionality
        self.embedding = nn.Linear(input_dim, model_dim)

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=model_dim * forward_expansion,
            dropout=0.1,
        )

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_transformer_layers
        )

        # Output layer
        self.output_layer = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # Increase dimensionality and apply positional encoding if needed
        x = self.embedding(x)

        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)

        # Predict output features
        output = self.output_layer(x)
        return output