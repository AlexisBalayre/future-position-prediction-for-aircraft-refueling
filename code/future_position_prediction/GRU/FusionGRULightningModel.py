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

from GRUNet.DecoderGRU import DecoderGRU
from GRUNet.GRUCell import GRUCell
from GRUNet.SelfAttentionAggregation import SelfAttentionAggregation
from GRUNet.IntermediaryEstimator import IntermediaryEstimator


class FusionGRULightningModel(L.LightningModule):
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
    ):
        super(FusionGRULightningModel, self).__init__()
        self.save_hyperparameters()

        self.gru_cell = GRUCell(bbox_dim * 2, hidden_dim, bbox_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        self.intermediary_estimator = IntermediaryEstimator(
            input_dim=hidden_dim,
            output_dim=bbox_dim * output_frames,
            activation=F.relu,
            dropout=[dropout, dropout],
        )

        self.self_attention = SelfAttentionAggregation(bbox_dim, hidden_dim)

        self.decoder = DecoderGRU(hidden_dim, hidden_dim, bbox_dim, n_layers=1)
        self.fc_out = nn.Linear(hidden_dim, bbox_dim)


    def forward(self, bboxes_position, bboxes_velocity):
        batch_size = bboxes_position.size(0)
        bboxes_position = bboxes_position.clone()
        bboxes_velocity = bboxes_velocity.clone()

        h = torch.zeros(
            batch_size, self.hparams.hidden_dim, device=bboxes_position.device
        )

        # Fusion-GRU encoding
        for t in range(self.hparams.input_frames):
            x = torch.cat([bboxes_position[:, t], bboxes_velocity[:, t]], dim=-1)
            h = self.gru_cell(x, h, bboxes_velocity[:, t])

        out = self.fc(h)

        # Intermediary estimation
        f_vel = self.intermediary_estimator(out.unsqueeze(1))  # Add time dimension
        f_vel_reshaped = f_vel.view(
            batch_size, self.hparams.output_frames, self.hparams.bbox_dim
        )

        # Decoding with self-attention
        predicted_velocities = []
        h_dec = h.unsqueeze(0)  # Add an extra dimension for n_layers
        for t in range(self.hparams.output_frames):
            x_agg = self.self_attention(f_vel_reshaped[:, t:])
            x_agg = x_agg.unsqueeze(1)  # Add time dimension for GRU input
            vel_t, h_dec = self.decoder(x_agg, h_dec)
            predicted_velocities.append(vel_t)

        predicted_velocities = torch.stack(predicted_velocities, dim=1)

        # Convert velocities to positions
        predicted_bboxes = convert_velocity_to_positions(
            predicted_velocities.detach(),
            bboxes_position.detach(),
            self.hparams.image_size,
        )

        return predicted_bboxes, predicted_velocities

    def _shared_step(self, batch, batch_idx, stage):
        (
            input_bboxes_position,
            input_bboxes_velocity,
            target_bboxes_position,
            target_bboxes_velocity,
        ) = batch

        predicted_bboxes, predicted_velocity = self(
            input_bboxes_position, input_bboxes_velocity
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
        predicted_bboxes_norm = predicted_bboxes / norm_factor
        target_bboxes_position_denorm = target_bboxes_position * norm_factor

        # Compute losses
        bbox_loss = F.smooth_l1_loss(predicted_bboxes_norm, target_bboxes_position)
        velocity_loss = F.smooth_l1_loss(predicted_velocity, target_bboxes_velocity)
        total_loss = bbox_loss + velocity_loss

        # Log losses
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

        # Compute metrics
        ade = compute_ADE(predicted_bboxes, target_bboxes_position_denorm)
        fde = compute_FDE(predicted_bboxes, target_bboxes_position_denorm)
        aiou = compute_AIOU(predicted_bboxes_norm, target_bboxes_position)
        fiou = compute_FIOU(predicted_bboxes_norm, target_bboxes_position)

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
            optimizer, mode="min", patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
