import torch

torch.autograd.set_detect_anomaly(True)

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
        hardtanh_limit: float = 1.0,
        image_size: Tuple[int, int] = (640, 480),  # Add image_size parameter
    ):
        super(LSTMLightningModel, self).__init__()
        self.save_hyperparameters()

        # Define LSTM Encoders
        self.bboxes_position_encoder = nn.LSTM(
            input_size=bbox_dim,
            hidden_size=hidden_dim,
            num_layers=hidden_depth,
            batch_first=True,
        )
        self.bboxes_velocity_encoder = nn.LSTM(
            input_size=bbox_dim,
            hidden_size=hidden_dim,
            num_layers=hidden_depth,
            batch_first=True,
        )

        # Define Fully Connected Layers
        self.fc_bboxes_velocity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bbox_dim),
            nn.Hardtanh(min_val=-hardtanh_limit, max_val=hardtanh_limit),
        )

        # Define LSTM Decoders
        self.bboxes_velocity_decoder = nn.LSTMCell(
            input_size=bbox_dim, hidden_size=hidden_dim
        )

    def forward(self, bboxes_position, bboxes_velocity):
        # Encode the Position and Velocity Bounding Boxes
        _, (h_vel, c_vel) = self.bboxes_velocity_encoder(bboxes_velocity)
        _, (h_pos, c_pos) = self.bboxes_position_encoder(bboxes_position)

        h_vel = h_vel[-1]
        c_vel = c_vel[-1]
        h_pos = h_pos[-1]
        c_pos = c_pos[-1]

        # Initialize hidden and cell states for the decoders
        h_vel_dec = torch.add(h_vel, h_pos)  # Utilisation de torch.add au lieu de +
        c_vel_dec = torch.add(c_vel, c_pos)  # Utilisation de torch.add au lieu de +

        # Decode Velocities
        velocity_outputs = []
        last_velocity = bboxes_velocity[:, -1, :].detach().clone()
        for _ in range(self.hparams.output_frames):
            h_vel_dec, c_vel_dec = self.bboxes_velocity_decoder(
                last_velocity, (h_vel_dec, c_vel_dec)
            )
            velocity_output = self.fc_bboxes_velocity(h_vel_dec)
            velocity_outputs.append(velocity_output)
            last_velocity = velocity_output.clone()
        predicted_velocity = torch.stack(velocity_outputs, dim=1)

        # Convert Velocities to Positions
        positions_from_velocity = convert_velocity_to_positions(
            predicted_velocity.detach(),
            bboxes_position.detach(),
            self.hparams.image_size,
        )

        return positions_from_velocity, predicted_velocity

    def _shared_step(self, batch, batch_idx, stage):
        (
            input_bboxes_position,
            input_bboxes_velocity,
            output_bboxes_position,
            output_bboxes_velocity,
        ) = batch

        predicted_bboxes, predicted_velocity = self(
            input_bboxes_position, input_bboxes_velocity
        )

        bbox_loss = F.smooth_l1_loss(predicted_bboxes, output_bboxes_position)
        velocity_loss = F.smooth_l1_loss(predicted_velocity, output_bboxes_velocity)
        total_loss = bbox_loss + velocity_loss

        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_bbox_loss",
            bbox_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_velocity_loss",
            velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # Compute and log metrics
        with torch.no_grad():
            # Detach the predicted bboxes to avoid backpropagating through the metrics
            output_bboxes_position = output_bboxes_position.detach()
            predicted_bboxes = predicted_bboxes.detach()

            # Denormalize the Ground Truth Bounding Boxes
            denorm_factor = torch.tensor(
                [
                    self.hparams.image_size[0],
                    self.hparams.image_size[1],
                    self.hparams.image_size[0],
                    self.hparams.image_size[1],
                ]
            ).to(output_bboxes_position.device)
            output_bboxes_denorm = output_bboxes_position * denorm_factor
            predicted_bboxes_norm = predicted_bboxes / denorm_factor

            # Compute metrics
            ade = compute_ADE(
                predicted_bboxes, output_bboxes_denorm
            )  # Average Displacement Error
            fde = compute_FDE(
                predicted_bboxes, output_bboxes_denorm
            )  # Final Displacement Error
            aiou = compute_AIOU(
                predicted_bboxes_norm, output_bboxes_position
            )  # Average Intersection Over Union
            fiou = compute_FIOU(
                predicted_bboxes_norm, output_bboxes_position
            )  # Final Intersection Over Union
            self.log(
                f"{stage}_ADE",
                ade,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{stage}_FDE",
                fde,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{stage}_AIoU",
                aiou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"{stage}_FIoU",
                fiou,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=15, factor=0.9
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
