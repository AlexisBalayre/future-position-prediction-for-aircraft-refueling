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


class GRULightningModelAverage(L.LightningModule):
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
        super(GRULightningModelAverage, self).__init__()
        self.save_hyperparameters()

        self.bbox_encoder = EncoderGRU(
            input_dim=bbox_dim,
            hidden_dim=hidden_dim,
            n_layers=encoder_layer_nb,
            output_frames_nb=output_frames,
            dropout=dropout,
        )
        self.vel_encoder = EncoderGRU(
            input_dim=bbox_dim,
            hidden_dim=hidden_dim,
            n_layers=encoder_layer_nb,
            output_frames_nb=output_frames,
            dropout=dropout,
        )
        self.acc_encoder = EncoderGRU(
            input_dim=bbox_dim,
            hidden_dim=hidden_dim,
            n_layers=encoder_layer_nb,
            output_frames_nb=output_frames,
            dropout=dropout,
        )

        self.pos_decoder = DecoderGRU(
            bbox_dim,
            hidden_dim,
            bbox_dim,
            n_layers=decoder_layer_nb,
            dropout=[dropout, dropout],
        )
        self.vel_decoder = DecoderGRU(
            bbox_dim,
            hidden_dim,
            bbox_dim,
            n_layers=decoder_layer_nb,
            dropout=[dropout, dropout],
        )

    def forward(
        self,
        bbox_seq,
        velocity_seq,
        acceleration_seq,
    ):
        batch_size = bbox_seq.size(0)

        # Initialize hidden state
        h_pos = torch.zeros(
            self.bbox_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=bbox_seq.device,
        )
        h_vel = torch.zeros(
            self.vel_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=velocity_seq.device,
        )
        h_acc = torch.zeros(
            self.acc_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=acceleration_seq.device,
        )

        # Encode the Position, Velocity, and Acceleration
        encoder_out_bbox, encoder_hidden_states_bbox = self.bbox_encoder(
            bbox_seq, h_pos
        )
        encoder_out_vel, encoder_hidden_states_vel = self.vel_encoder(
            velocity_seq, h_vel
        )
        _, encoder_hidden_states_acc = self.acc_encoder(acceleration_seq, h_acc)

        # Combine the hidden states and cell states (average)
        h_pos = h_vel = (
            encoder_hidden_states_bbox
            + encoder_hidden_states_vel
            + encoder_hidden_states_acc
        ) / 3

        # decoder with teacher forcing
        decoder_input_pos = bbox_seq[:, -1, :]
        decoder_input_vel = velocity_seq[:, -1, :]

        predictions_position = torch.tensor([], device=bbox_seq.device)
        predictions_velocity = torch.tensor([], device=velocity_seq.device)

        for t in range(self.hparams.output_frames):
            out_pos, h_pos = self.pos_decoder(decoder_input_pos, h_pos)
            out_vel, h_vel = self.vel_decoder(decoder_input_vel, h_vel)

            predictions_position = torch.cat(
                (predictions_position, out_pos.unsqueeze(1)), dim=1
            )
            predictions_velocity = torch.cat(
                (predictions_velocity, out_vel.unsqueeze(1)), dim=1
            )

            decoder_input_pos = out_pos.detach()
            decoder_input_vel = out_vel.detach()

        return predictions_position, predictions_velocity

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
        predicted_positions, predicted_velocity = self(
            input_bboxes,
            input_velocities,
            input_accelerations,
        )

        # Convert predicted future velocities to future positions
        velocities_to_positions = convert_velocity_to_positions(
            predicted_velocity=predicted_velocity,
            past_positions=input_bboxes,
        )

        # Compute losses
        velocity_loss = F.smooth_l1_loss(predicted_velocity, output_velocities)
        pos_loss = F.smooth_l1_loss(predicted_positions, output_bboxes)
        velocities_to_positions_loss = F.smooth_l1_loss(
            velocities_to_positions, output_bboxes
        )
        total_loss = velocity_loss + velocities_to_positions_loss * 0.1 + pos_loss

        self.log(
            f"{stage}_vel_loss",
            velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_pos_loss",
            pos_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_vel_to_pos_loss",
            velocities_to_positions_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        # Log metrics
        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False
        )

        # Compute metrics
        ade_from_vel = compute_ADE(
            velocities_to_positions, output_bboxes, self.hparams.image_size
        )
        fde_from_vel = compute_FDE(
            velocities_to_positions, output_bboxes, self.hparams.image_size
        )
        aiou_from_vel = compute_AIOU(velocities_to_positions, output_bboxes)
        fiou_from_vel = compute_FIOU(velocities_to_positions, output_bboxes)
        ade = compute_ADE(predicted_positions, output_bboxes, self.hparams.image_size)
        fde = compute_FDE(predicted_positions, output_bboxes, self.hparams.image_size)
        aiou = compute_AIOU(predicted_positions, output_bboxes)
        fiou = compute_FIOU(predicted_positions, output_bboxes)

        # Log Best Value between fiou and fiou_from_vel
        if fiou < fiou_from_vel:
            self.log(
                f"{stage}_best_fiou",
                fiou,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        else:
            self.log(
                f"{stage}_best_fiou",
                fiou_from_vel,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        
        self.log(
            f"{stage}_ade_from_vel",
            ade_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_fde_from_vel",
            fde_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_aiou_from_vel",
            aiou_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_fiou_from_vel",
            fiou_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_ade",
            ade,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_fde",
            fde,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_aiou",
            aiou,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_fiou",
            fiou,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
            optimizer,
            mode="min",
            patience=self.hparams.scheduler_patience,
            factor=self.hparams.scheduler_factor,
            threshold=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
