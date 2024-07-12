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
        image_size: Tuple[int, int] = (640, 480),
        num_heads: int = 4,
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
        self.fc_bboxes_position = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bbox_dim),
            nn.Sigmoid()
        )

        # Define LSTM Decoders
        self.bboxes_velocity_decoder = nn.LSTMCell(
            input_size=bbox_dim, hidden_size=hidden_dim
        )
        self.bboxes_position_decoder = nn.LSTMCell(
            input_size=bbox_dim + hidden_dim, hidden_size=hidden_dim
        )

    def forward(self, bboxes_position, bboxes_velocity):
        # Encode the Position and Velocity Bounding Boxes
        _, (h_vel, c_vel) = self.bboxes_velocity_encoder(bboxes_velocity)
        _, (h_pos, c_pos) = self.bboxes_position_encoder(bboxes_position)

        # Attend to the Encoded Position and Velocity Bounding Boxes
        h_vel_dec = h_pos_dec = torch.add(h_pos.squeeze(0), h_vel.squeeze(0))
        c_vel_dec = c_pos_dec = torch.add(c_pos.squeeze(0), c_vel.squeeze(0))

        # Decode Velocities
        velocity_outputs = []
        last_velocity = bboxes_velocity[:, -1, :].detach()
        for _ in range(self.hparams.output_frames):
            h_vel_dec, c_vel_dec  = self.bboxes_velocity_decoder(last_velocity, (h_vel_dec, c_vel_dec))
            velocity_output = self.fc_bboxes_velocity(h_vel_dec)
            velocity_outputs.append(velocity_output)
            last_velocity = velocity_output.detach()
        predicted_velocity = torch.stack(velocity_outputs, dim=1)

        # Decode Positions
        position_outputs = []
        last_position = bboxes_position[:, -1, :].detach()
        for _ in range(self.hparams.output_frames):
            h_pos_dec, c_pos_dec = self.bboxes_position_decoder(
                torch.cat([last_position, h_vel_dec], dim=1), (h_pos_dec, c_pos_dec)
            )
            position_output = self.fc_bboxes_position(h_pos_dec)
            position_outputs.append(position_output)
            last_position = position_output.detach()
        predicted_position = torch.stack(position_outputs, dim=1)

        return predicted_position, predicted_velocity

    def _shared_step(self, batch, batch_idx, stage):
        (
            input_bboxes_position,
            input_bboxes_velocity,
            target_bboxes_position,
            target_bboxes_velocity,
        ) = batch
        
        # Infer the future positions and velocities
        predicted_position, predicted_velocity = self(
            input_bboxes_position, input_bboxes_velocity
        )

        # Convert predicted future velocities to future positions
        velocities_to_positions = convert_velocity_to_positions(
            predicted_velocity.detach(),
            input_bboxes_position.detach(),
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
        ).to(target_bboxes_position.device)
        velocities_to_positions_norm = velocities_to_positions / norm_factor
        target_bboxes_position_denorm = target_bboxes_position * norm_factor

        # Compute losses
        bbox_loss = F.smooth_l1_loss(predicted_position, target_bboxes_position)
        velocities_to_positions_loss = F.smooth_l1_loss(
            velocities_to_positions_norm, target_bboxes_position
        )
        velocity_loss = F.smooth_l1_loss(predicted_velocity, target_bboxes_velocity)
        total_loss = velocity_loss + bbox_loss + velocities_to_positions_loss * 0.1

        # Log losses
        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False
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

        # Compute metrics
        ade = compute_ADE(velocities_to_positions, target_bboxes_position_denorm)
        fde = compute_FDE(velocities_to_positions, target_bboxes_position_denorm)
        aiou = compute_AIOU(velocities_to_positions_norm, target_bboxes_position)
        fiou = compute_FIOU(predicted_position, target_bboxes_position)

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
            prog_bar=False,
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
            optimizer, mode="min", patience=15, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
