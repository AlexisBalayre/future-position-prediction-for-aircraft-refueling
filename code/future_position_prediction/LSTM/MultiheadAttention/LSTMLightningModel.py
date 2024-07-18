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
        num_heads: int = 4,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.9
    ):
        super(LSTMLightningModel, self).__init__()
        self.save_hyperparameters()

        # Define LSTM Encoders
        self.bboxes_position_encoder = LSTMEncoder(
            input_dim=bbox_dim, hidden_dim=hidden_dim, num_layers=hidden_depth
        )
        self.bboxes_velocity_encoder = LSTMEncoder(
            input_dim=bbox_dim, hidden_dim=hidden_dim, num_layers=hidden_depth
        )

        # Define LSTM Decoders
        self.decoder_bbox = LSTMDecoder(
            input_dim=bbox_dim,
            hidden_dim=hidden_dim,
            output_dim=bbox_dim,
            num_layers=hidden_depth,
            output_activation=nn.Sigmoid(),
            dropout=dropout,
            num_heads=num_heads
        )
        self.decoder_velocity = LSTMDecoder(
            input_dim=bbox_dim,
            hidden_dim=hidden_dim,
            output_dim=bbox_dim,
            num_layers=hidden_depth,
            dropout=dropout,
            num_heads=num_heads
        ) 

    def forward(self, bbox_seq, velocity_seq):
        # print(bboxes_position.shape)
        # 16, 15, 4 -> batch_size, input_frames, bbox_dim

        # Encode the Position and Velocity Bounding Boxes
        encoder_outputs_bbox, hidden_bbox, cell_bbox = self.bboxes_position_encoder(
            bbox_seq
        )
        encoder_outputs_velocity, hidden_velocity, cell_velocity = (
            self.bboxes_velocity_encoder(velocity_seq)
        )

        decoder_input_bbox = bbox_seq[:, -1, :]
        decoder_input_velocity = velocity_seq[:, -1, :]

        predictions_bbox = []
        predictions_velocity = []

        for _ in range(self.hparams.output_frames):
            decoder_output_bbox, hidden_bbox, cell_bbox = self.decoder_bbox(
                decoder_input_bbox, hidden_bbox, cell_bbox, encoder_outputs_bbox
            )
            decoder_output_velocity, hidden_velocity, cell_velocity = (
                self.decoder_velocity(
                    decoder_input_velocity,
                    hidden_velocity,
                    cell_velocity,
                    encoder_outputs_velocity,
                )
            )

            predictions_bbox.append(decoder_output_bbox)
            predictions_velocity.append(decoder_output_velocity)

            decoder_input_bbox = decoder_output_bbox
            decoder_input_velocity = decoder_output_velocity

        predictions_bbox = torch.stack(predictions_bbox, dim=1)
        predictions_velocity = torch.stack(predictions_velocity, dim=1)

        return predictions_bbox, predictions_velocity

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
        predicted_position_denorm = predicted_position * norm_factor

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
        ade_from_vel = compute_ADE(velocities_to_positions, target_bboxes_position_denorm)
        fde_from_vel = compute_FDE(velocities_to_positions, target_bboxes_position_denorm)
        aiou_from_vel = compute_AIOU(velocities_to_positions_norm, target_bboxes_position)
        fiou_from_vel = compute_FIOU(velocities_to_positions_norm, target_bboxes_position)
        ade = compute_ADE(predicted_position_denorm, target_bboxes_position_denorm)
        fde = compute_FDE(predicted_position_denorm, target_bboxes_position_denorm)
        aiou = compute_AIOU(predicted_position, target_bboxes_position)
        fiou = compute_FIOU(predicted_position, target_bboxes_position)

        # Log metrics
        self.log(
            f"{stage}_ADE_from_vel",
            ade_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_FDE_from_vel",
            fde_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_AIoU_from_vel",
            aiou_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_FIoU_from_vel",
            fiou_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
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

        return {
            "total_loss": total_loss,
            "bbox_loss": bbox_loss,
            "velocity_loss": velocity_loss,
            "velocities_to_positions_loss": velocities_to_positions_loss,
            "ADE_from_vel": ade_from_vel,
            "FDE_from_vel": fde_from_vel,
            "AIoU_from_vel": aiou_from_vel,
            "FIoU_from_vel": fiou_from_vel,
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
