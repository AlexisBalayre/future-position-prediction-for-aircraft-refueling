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
            input_dim=bbox_dim,
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
        x = bboxes_position
        out, h, future_positions = self.encoder(x, h)

        future_positions = future_positions.view(
            batch_size, self.hparams.output_frames, self.hparams.bbox_dim
        )

        # Decoding with self-attention
        predicted_positions = []
        h_dec = h
        for t in range(self.hparams.output_frames):
            x_agg = self.self_attention(future_positions[:, t:])
            x_agg = x_agg.unsqueeze(1)  # Add time dimension for GRU input
            vel_t, h_dec = self.decoder(x_agg, h_dec)
            predicted_positions.append(vel_t)

        predicted_positions = torch.stack(predicted_positions, dim=1)
        return predicted_positions

    def _shared_step(self, batch, batch_idx, stage):
        (
            input_bboxes_position,
            input_bboxes_velocity,
            target_bboxes_position,
            target_bboxes_velocity,
        ) = batch

        predicted_positions = self(input_bboxes_position, input_bboxes_velocity)

        # Normalize the output bounding boxes
        norm_factor = torch.tensor(
            [
                self.hparams.image_size[0],
                self.hparams.image_size[1],
                self.hparams.image_size[0],
                self.hparams.image_size[1],
            ]
        ).to(target_bboxes_position.device)
        target_bboxes_position_denorm = target_bboxes_position * norm_factor
        predicted_positions_denorm = predicted_positions * norm_factor

        # Compute losses
        predicted_bbox_loss = F.smooth_l1_loss(
            predicted_positions, target_bboxes_position
        )
        total_loss = predicted_bbox_loss

        # Log losses
        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False
        )

        # Compute metrics
        ade = compute_ADE(predicted_positions_denorm, target_bboxes_position_denorm)
        fde = compute_FDE(predicted_positions_denorm, target_bboxes_position_denorm)
        aiou = compute_AIOU(predicted_positions, target_bboxes_position)
        fiou = compute_FIOU(predicted_positions, target_bboxes_position)

        # Log metrics
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
