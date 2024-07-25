import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from typing import Tuple

from utils import (
    compute_ADE,
    compute_FDE,
    compute_AIOU,
    compute_FIOU,
    convert_velocity_to_positions,
)

from LSTMEncoder import LSTMEncoder
from LSTMDecoder import LSTMDecoder


class LSTMLightningModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        input_frames: int = 10,
        output_frames: int = 10,
        batch_size: int = 32,
        bbox_dim: int = 4,
        hidden_dim: int = 256,
        hidden_depth: int = 3,
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (640, 480),
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.9,
    ):
        super(LSTMLightningModel, self).__init__()
        self.save_hyperparameters()

        # Define LSTM Encoders
        self.position_encoder = LSTMEncoder(
            input_dim=bbox_dim, hidden_dim=hidden_dim, num_layers=hidden_depth
        )
        self.velocity_encoder = LSTMEncoder(
            input_dim=bbox_dim, hidden_dim=hidden_dim, num_layers=hidden_depth
        )
        self.acceleration_encoder = LSTMEncoder(
            input_dim=bbox_dim, hidden_dim=hidden_dim, num_layers=hidden_depth
        )

        # Define LSTM Decoder (only for velocity now)
        self.decoder_velocity = LSTMDecoder(
            input_dim=bbox_dim,
            hidden_dim=hidden_dim,
            output_dim=bbox_dim,
            num_layers=hidden_depth,
            dropout=dropout,
        )

        # Learned combination layers
        self.combine_outputs = nn.Linear(hidden_dim * 3, hidden_dim)
        self.combine_hidden = nn.Linear(hidden_dim * 3, hidden_dim)
        self.combine_cell = nn.Linear(hidden_dim * 3, hidden_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # Gating mechanism
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def combine_encoder_outputs(
        self, outputs_bbox, outputs_velocity, outputs_acceleration
    ):
        combined = torch.cat(
            (outputs_bbox, outputs_velocity, outputs_acceleration), dim=-1
        )
        return self.combine_outputs(combined)

    def combine_hidden_states(self, hidden_bbox, hidden_velocity, hidden_acceleration):
        combined = torch.cat(
            (hidden_bbox, hidden_velocity, hidden_acceleration), dim=-1
        )
        return self.combine_hidden(combined)

    def combine_cell_states(self, cell_bbox, cell_velocity, cell_acceleration):
        combined = torch.cat((cell_bbox, cell_velocity, cell_acceleration), dim=-1)
        return self.combine_cell(combined)

    def forward(self, bbox_seq, velocity_seq, acceleration_seq):

        # Encode the Position, Velocity, and Acceleration
        encoder_outputs_bbox, hidden_bbox, cell_bbox = self.position_encoder(bbox_seq)
        encoder_outputs_velocity, hidden_velocity, cell_velocity = (
            self.velocity_encoder(velocity_seq)
        )
        encoder_outputs_acceleration, hidden_acceleration, cell_acceleration = (
            self.acceleration_encoder(acceleration_seq)
        )

        # Combine encoder outputs and states
        encoder_output_combined = self.combine_encoder_outputs(
            encoder_outputs_bbox, encoder_outputs_velocity, encoder_outputs_acceleration
        )
        hidden_combined = self.combine_hidden_states(
            hidden_bbox, hidden_velocity, hidden_acceleration
        )
        cell_combined = self.combine_cell_states(
            cell_bbox, cell_velocity, cell_acceleration
        )

        decoder_input_velocity = velocity_seq[:, -1, :]
        predictions_velocity = []

        # Prepare encoder_output_combined for attention
        # Shape should be (seq_len, batch_size, hidden_dim)
        encoder_output_combined = encoder_output_combined.transpose(0, 1)

        for _ in range(self.hparams.output_frames):
            # Reshape hidden_combined for attention
            # Shape should be (1, batch_size, hidden_dim)
            query = hidden_combined[-1].unsqueeze(0)

            # Ensure key and value have the correct shape
            key = encoder_output_combined
            value = encoder_output_combined

            # Compute attention context
            attn_output, _ = self.attention(query, key, value)

            # Reshape attention output
            # Shape should be (batch_size, 1, hidden_dim)
            context = attn_output.transpose(0, 1)

            # Apply gating
            gate = torch.sigmoid(
                self.gate(
                    torch.cat((encoder_output_combined[-1], context.squeeze(1)), dim=-1)
                )
            )
            context = gate * encoder_output_combined[-1] + (1 - gate) * context.squeeze(
                1
            )

            decoder_output_velocity, (hidden_combined, cell_combined) = (
                self.decoder_velocity(
                    decoder_input_velocity,
                    hidden_combined,
                    cell_combined,
                    context.unsqueeze(1),  # Add sequence dimension for decoder
                )
            )

            predictions_velocity.append(decoder_output_velocity)
            decoder_input_velocity = (
                decoder_output_velocity  # Use prediction as next input
            )

        predictions_velocity = torch.stack(predictions_velocity, dim=1)

        return predictions_velocity

    def _shared_step(self, batch, batch_idx, stage):
        (
            video_id,
            input_bboxes,
            input_velocities,
            input_accelerations,
            output_bboxes,
            output_velocities,
        ) = batch

        # Predict future velocities
        predicted_velocity = self(input_bboxes, input_velocities, input_accelerations)

        # Convert predicted future velocities to future positions
        velocities_to_positions = convert_velocity_to_positions(
            predicted_velocity.detach(),
            input_bboxes.detach(),
            self.hparams.image_size,
        )

        # Normalize the output bounding boxes
        norm_factor = torch.tensor(
            [
                self.hparams.image_size[0],
                self.hparams.image_size[1],
                self.hparams.image_size[0],
                self.hparams.image_size[1],
            ]
        ).to(output_bboxes.device)
        velocities_to_positions_norm = velocities_to_positions / norm_factor
        output_bboxes_denorm = output_bboxes * norm_factor

        # Compute losses
        velocity_loss = F.smooth_l1_loss(predicted_velocity, output_velocities)
        velocities_to_positions_loss = F.smooth_l1_loss(
            velocities_to_positions_norm, output_bboxes
        )
        total_loss = velocity_loss + velocities_to_positions_loss * 0.1

        # Log losses
        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            f"{stage}_velocity_loss",
            velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # Compute metrics
        ade = compute_ADE(velocities_to_positions, output_bboxes_denorm)
        fde = compute_FDE(velocities_to_positions, output_bboxes_denorm)
        aiou = compute_AIOU(velocities_to_positions_norm, output_bboxes)
        fiou = compute_FIOU(velocities_to_positions_norm, output_bboxes)

        # Log metrics
        self.log(f"{stage}_ADE", ade, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_FDE", fde, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_AIoU", aiou, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_FIoU", fiou, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "total_loss": total_loss,
            "velocity_loss": velocity_loss,
            "velocities_to_positions_loss": velocities_to_positions_loss,
            "ADE": ade,
            "FDE": fde,
            "AIoU": aiou,
            "FIoU": fiou,
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")["total_loss"]

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.scheduler_patience,
            factor=self.hparams.scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
