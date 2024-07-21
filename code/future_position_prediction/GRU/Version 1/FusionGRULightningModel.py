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

        self.self_attention = SelfAttentionAggregation(bbox_dim * 2, hidden_dim)

        self.decoder = DecoderGRU(
            hidden_dim, hidden_dim, bbox_dim * 2, n_layers=decoder_layer_nb
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
        out, h, future_predictions = self.encoder(x, h)

        future_predictions = future_predictions.view(
            batch_size, self.hparams.output_frames, self.hparams.bbox_dim * 2
        )

        # Decoding with self-attention
        predicted_outputs = []
        h_dec = h
        for t in range(self.hparams.output_frames):
            x_agg = self.self_attention(future_predictions[:, t:])
            x_agg = x_agg.unsqueeze(1)  # Add time dimension for GRU input
            out_t, h_dec = self.decoder(x_agg, h_dec)
            predicted_outputs.append(out_t)

        predicted_outputs = torch.stack(predicted_outputs, dim=1)

        # Split predictions into velocities and positions
        predicted_velocities = predicted_outputs[:, :, : self.hparams.bbox_dim]
        predicted_positions = predicted_outputs[:, :, self.hparams.bbox_dim :]

        # Refine position predictions using velocity
        refined_positions = self.refine_positions(
            bboxes_position[:, -1], predicted_velocities, predicted_positions
        )

        return refined_positions, predicted_velocities

    def refine_positions(
        self, last_input_position, predicted_velocities, predicted_positions
    ):
        # Initialize refined positions with the predicted positions
        refined_positions = predicted_positions.clone()

        # Compute cumulative displacements
        cumulative_displacements = torch.cumsum(predicted_velocities, dim=1)

        # Add the last input position and cumulative displacements
        refined_positions = last_input_position.unsqueeze(1) + cumulative_displacements

        # Blend between direct position predictions and velocity-based predictions
        alpha = 0.5  # Blending factor, can be tuned
        refined_positions = (
            alpha * refined_positions + (1 - alpha) * predicted_positions
        )

        # Clip to ensure positions are within image bounds
        refined_positions = torch.clamp(
            refined_positions, min=0, max=1
        )  # Assuming normalized coordinates

        return refined_positions

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

        # Compute losses
        position_loss = F.smooth_l1_loss(predicted_bboxes, target_bboxes_position)
        velocity_loss = F.smooth_l1_loss(predicted_velocity, target_bboxes_velocity)
        total_loss = position_loss + velocity_loss

        # Log losses
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_position_loss", position_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_velocity_loss", velocity_loss, on_step=False, on_epoch=True, prog_bar=False)

        # Compute metrics
        ade = compute_ADE(predicted_bboxes, target_bboxes_position)
        fde = compute_FDE(predicted_bboxes, target_bboxes_position)
        aiou = compute_AIOU(predicted_bboxes, target_bboxes_position)
        fiou = compute_FIOU(predicted_bboxes, target_bboxes_position)

        self.log(f"{stage}_ADE", ade, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_FDE", fde, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_AIoU", aiou, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_FIoU", fiou, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "total_loss": total_loss,
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
