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
from GRUNet.EncoderGRU import EncoderGRU
from GRUNet.SelfAttentionAggregation import SelfAttentionAggregation


class FusionGRULightningModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        input_frames: int = 10,
        output_frames: int = 10,
        batch_size: int = 32,
        bbox_dim: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (640, 480),
        encoder_layer_nb: int = 1,
        decoder_layer_nb: int = 1,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        super(FusionGRULightningModel, self).__init__()
        self.save_hyperparameters()

        self.encoder = EncoderGRU(
            input_dim=bbox_dim * 2,
            hidden_dim=hidden_dim,
            n_layers=encoder_layer_nb,
            output_frames_nb=output_frames,
            dropout=dropout,
        )

        self.self_attention = SelfAttentionAggregation(bbox_dim, hidden_dim)

        self.decoder = DecoderGRU(
            hidden_dim, hidden_dim, bbox_dim, n_layers=decoder_layer_nb
        )

    def forward(self, bboxes_position, bboxes_velocity):
        batch_size = bboxes_position.size(0)
        bboxes_position = bboxes_position.clone()
        bboxes_velocity = bboxes_velocity.clone()

        # Initialize hidden state
        h = torch.zeros(
            self.encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=bboxes_position.device,
        )

        # Fusion-GRU encoding
        x = torch.cat([bboxes_position, bboxes_velocity], dim=-1)
        out, h, future_velocities = self.encoder(x, h)

        future_velocities = future_velocities.view(
            batch_size, self.hparams.output_frames, self.hparams.bbox_dim
        )

        # Decoding with self-attention
        predicted_velocities = []
        h_dec = h
        for t in range(self.hparams.output_frames):
            x_agg = self.self_attention(future_velocities[:, t:])
            x_agg = x_agg.unsqueeze(1)  # Add time dimension for GRU input
            vel_t, h_dec = self.decoder(x_agg, h_dec)
            predicted_velocities.append(vel_t)

        predicted_velocities = torch.stack(predicted_velocities, dim=1)
        return predicted_velocities

    def _shared_step(self, batch, batch_idx, stage):
        (
            input_bboxes_position,
            input_bboxes_velocity,
            target_bboxes_position,
            target_bboxes_velocity,
        ) = batch

        predicted_velocity = self(input_bboxes_position, input_bboxes_velocity)

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
        velocities_to_positions_loss = F.smooth_l1_loss(
            velocities_to_positions_norm, target_bboxes_position
        )
        velocity_loss = F.smooth_l1_loss(predicted_velocity, target_bboxes_velocity)
        total_loss = velocity_loss + velocities_to_positions_loss * 0.1

        # Log losses
        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False
        )

        # Compute metrics
        ade_from_vel = compute_ADE(
            velocities_to_positions, target_bboxes_position_denorm
        )
        fde_from_vel = compute_FDE(
            velocities_to_positions, target_bboxes_position_denorm
        )
        aiou_from_vel = compute_AIOU(
            velocities_to_positions_norm, target_bboxes_position
        )
        fiou_from_vel = compute_FIOU(
            velocities_to_positions_norm, target_bboxes_position
        )

        # Log metrics
        self.log(
            f"{stage}_ADE",
            ade_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_FDE",
            fde_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_AIoU",
            aiou_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}_FIoU",
            fiou_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {
            "total_loss": total_loss,
            "ADE": ade_from_vel,
            "FDE": fde_from_vel,
            "AIoU": aiou_from_vel,
            "FIoU": fiou_from_vel,
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
