import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from typing import Tuple
import random

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
        bbox_size: int = 4,
        hidden_size: int = 256,
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
            input_size=bbox_size,
            hidden_size=hidden_size,
            num_layers=hidden_depth,
        )
        self.velocity_encoder = LSTMEncoder(
            input_size=bbox_size, hidden_size=hidden_size, num_layers=hidden_depth
        )
        self.acceleration_encoder = LSTMEncoder(
            input_size=bbox_size, hidden_size=hidden_size, num_layers=hidden_depth
        )

        # Define LSTM Decoder (only for velocity now)
        self.decoder_velocity = LSTMDecoder(
            input_size=bbox_size,
            hidden_size=hidden_size,
            output_size=bbox_size,
            num_layers=hidden_depth,
            dropout=dropout,
        )

    def forward(
        self,
        bbox_seq,
        velocity_seq,
        acceleration_seq,
        teacher_forcing_ratio=0.5,
        training_prediction="recursive",
        velocity_seq_target=None,
    ):
        # Encode the Position, Velocity, and Acceleration
        _, (encoder_hidden_states_bbox, encoder_cell_states_bbox) = (
            self.position_encoder(bbox_seq)
        )
        _, (
            encoder_hidden_states_vel,
            encoder_cell_states_vel,
        ) = self.velocity_encoder(velocity_seq)
        _, (
            encoder_hidden_states_acc,
            encoder_cell_states_acc,
        ) = self.acceleration_encoder(acceleration_seq)

        # Combine the outputs and states
        hidden_states_combined = (
            encoder_hidden_states_vel
            + encoder_hidden_states_acc
            + encoder_hidden_states_bbox
        )
        cell_states_combined = (
            encoder_cell_states_vel + encoder_cell_states_acc + encoder_cell_states_bbox
        )

        # decoder with teacher forcing
        decoder_input = velocity_seq[:, -1, :]
        predictions_velocity = torch.tensor([], device=velocity_seq.device)

        if training_prediction == "recursive":
            # predict recursively
            for _ in range(self.hparams.output_frames):
                decoder_output_velocity, (
                    hidden_states_combined,
                    cell_states_combined,
                ) = self.decoder_velocity(
                    decoder_input, hidden_states_combined, cell_states_combined
                )
                predictions_velocity = torch.cat(
                    (predictions_velocity, decoder_output_velocity.unsqueeze(1)), dim=1
                )
                decoder_input = decoder_output_velocity.detach()

        if training_prediction == "teacher_forcing":
            # use teacher forcing
            if random.random() < teacher_forcing_ratio:
                for t in range(self.hparams.output_frames):
                    decoder_output_velocity, (
                        hidden_states_combined,
                        cell_states_combined,
                    ) = self.decoder_velocity(
                        decoder_input, hidden_states_combined, cell_states_combined
                    )
                    predictions_velocity = torch.cat(
                        (predictions_velocity, decoder_output_velocity.unsqueeze(1)),
                        dim=1,
                    )
                    decoder_input = velocity_seq_target[:, t, :]

            # predict recursively
            else:
                for t in range(self.hparams.output_frames):
                    decoder_output_velocity, (
                        hidden_states_combined,
                        cell_states_combined,
                    ) = self.decoder_velocity(
                        decoder_input, hidden_states_combined, cell_states_combined
                    )
                    predictions_velocity = torch.cat(
                        (predictions_velocity, decoder_output_velocity.unsqueeze(1)),
                        dim=1,
                    )
                    decoder_input = decoder_output_velocity.detach()

        if training_prediction == "mixed_teacher_forcing":
            # predict using mixed teacher forcing
            for t in range(self.hparams.output_frames):
                decoder_output_velocity, (
                    hidden_states_combined,
                    cell_states_combined,
                ) = self.decoder_velocity(
                    decoder_input, hidden_states_combined, cell_states_combined
                )
                predictions_velocity = torch.cat(
                    (predictions_velocity, decoder_output_velocity.unsqueeze(1)), dim=1
                )

                # predict with teacher forcing
                if random.random() < teacher_forcing_ratio:
                    decoder_input = velocity_seq_target[:, t, :]

                # predict recursively
                else:
                    decoder_input = decoder_output_velocity.detach()


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

        if stage == "train":
            predicted_velocity = self(
                input_bboxes,
                input_velocities,
                input_accelerations,
                training_prediction="mixed_teacher_forcing",
                velocity_seq_target=output_velocities,
                teacher_forcing_ratio=0.6,
            )
        else:
            predicted_velocity = self(
                input_bboxes, input_velocities, input_accelerations
            )

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
        total_loss = velocity_loss  # + velocities_to_positions_loss * 0.1

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
        self.log(
            f"{stage}_ADE",
            ade,
            on_step=False,
            on_epoch=True,
            prog_bar=False if stage == "train" else True,
        )
        self.log(
            f"{stage}_FDE",
            fde,
            on_step=False,
            on_epoch=True,
            prog_bar=False if stage == "train" else True,
        )
        self.log(
            f"{stage}_AIoU",
            aiou,
            on_step=False,
            on_epoch=True,
            prog_bar=False if stage == "train" else True,
        )
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
