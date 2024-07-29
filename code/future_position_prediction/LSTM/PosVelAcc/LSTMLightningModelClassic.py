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

from MetricsMonitoring import MetricsMonitoring
from LSTMEncoder import LSTMEncoder
from LSTMDecoder import LSTMDecoder


class LSTMLightningModelClassic(L.LightningModule):
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
        super(LSTMLightningModelClassic, self).__init__()
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

        # Define LSTM Decoders
        self.decoder_velocity = LSTMDecoder(
            input_size=bbox_size,
            hidden_size=hidden_size,
            output_size=bbox_size,
            num_layers=hidden_depth,
            dropout=dropout,
        )
        self.decoder_position = LSTMDecoder(
            input_size=bbox_size,
            hidden_size=hidden_size,
            output_size=bbox_size,
            num_layers=hidden_depth,
            dropout=dropout,
        )

        # Initialize metrics monitoring
        self.train_metrics = MetricsMonitoring(image_size)
        self.val_metrics = MetricsMonitoring(image_size)
        self.test_metrics = MetricsMonitoring(image_size)


    def forward(
        self,
        bbox_seq,
        velocity_seq,
        teacher_forcing_ratio=0.6,
        training_prediction="recursive",
        position_seq_target=None,
        velocity_seq_target=None,
    ):
        # Encode the Position, Velocity, and Acceleration
        _, (hidden_states_pos, cell_states_pos) = self.position_encoder(bbox_seq)
        _, (
            hidden_states_vel,
            cell_states_vel,
        ) = self.velocity_encoder(velocity_seq)

        # decoder with teacher forcing
        decoder_input_pos = bbox_seq[:, -1, :]
        decoder_input_vel = velocity_seq[:, -1, :]

        predictions_position = torch.tensor([], device=bbox_seq.device)
        predictions_velocity = torch.tensor([], device=velocity_seq.device)

        if training_prediction == "recursive":
            # predict recursively
            for _ in range(self.hparams.output_frames):
                decoder_output_position, (
                    hidden_states_pos,
                    cell_states_pos,
                ) = self.decoder_position(
                    decoder_input_pos, hidden_states_pos, cell_states_pos
                )
                decoder_output_velocity, (
                    hidden_states_vel,
                    cell_states_vel,
                ) = self.decoder_velocity(
                    decoder_input_vel, hidden_states_vel, cell_states_vel
                )

                predictions_position = torch.cat(
                    (predictions_position, decoder_output_position.unsqueeze(1)), dim=1
                )
                predictions_velocity = torch.cat(
                    (predictions_velocity, decoder_output_velocity.unsqueeze(1)), dim=1
                )

                decoder_input_pos = decoder_output_position.detach()
                decoder_input_vel = decoder_output_velocity.detach()

        if training_prediction == "teacher_forcing":
            # use teacher forcing
            if random.random() < teacher_forcing_ratio:
                for t in range(self.hparams.output_frames):
                    decoder_output_position, (
                        hidden_states_pos,
                        cell_states_pos,
                    ) = self.decoder_position(
                        decoder_input_pos, hidden_states_pos, cell_states_pos
                    )
                    decoder_output_velocity, (
                        hidden_states_vel,
                        cell_states_vel,
                    ) = self.decoder_velocity(
                        decoder_input_vel, hidden_states_vel, cell_states_vel
                    )

                    predictions_position = torch.cat(
                        (predictions_position, decoder_output_position.unsqueeze(1)),
                        dim=1,
                    )
                    predictions_velocity = torch.cat(
                        (predictions_velocity, decoder_output_velocity.unsqueeze(1)),
                        dim=1,
                    )

                    decoder_input_pos = position_seq_target[:, t, :]
                    decoder_input_vel = velocity_seq_target[:, t, :]

            # predict recursively
            else:
                for t in range(self.hparams.output_frames):
                    decoder_output_position, (
                        hidden_states_pos,
                        cell_states_pos,
                    ) = self.decoder_position(
                        decoder_input_pos, hidden_states_pos, cell_states_pos
                    )
                    decoder_output_velocity, (
                        hidden_states_vel,
                        cell_states_vel,
                    ) = self.decoder_velocity(
                        decoder_input_vel, hidden_states_vel, cell_states_vel
                    )

                    predictions_position = torch.cat(
                        (predictions_position, decoder_output_position.unsqueeze(1)),
                        dim=1,
                    )
                    predictions_velocity = torch.cat(
                        (predictions_velocity, decoder_output_velocity.unsqueeze(1)),
                        dim=1,
                    )

                    decoder_input_pos = decoder_output_position.detach()
                    decoder_input_vel = decoder_output_velocity.detach()

        if training_prediction == "mixed_teacher_forcing":
            # predict using mixed teacher forcing
            for t in range(self.hparams.output_frames):
                decoder_output_position, (
                    hidden_states_pos,
                    cell_states_pos,
                ) = self.decoder_position(
                    decoder_input_pos, hidden_states_pos, cell_states_pos
                )
                decoder_output_velocity, (
                    hidden_states_vel,
                    cell_states_vel,
                ) = self.decoder_velocity(
                    decoder_input_vel, hidden_states_vel, cell_states_vel
                )

                predictions_position = torch.cat(
                    (predictions_position, decoder_output_position.unsqueeze(1)), dim=1
                )
                predictions_velocity = torch.cat(
                    (predictions_velocity, decoder_output_velocity.unsqueeze(1)), dim=1
                )

                decoder_input_pos = decoder_output_position.detach()
                decoder_input_vel = decoder_output_velocity.detach()

                # predict with teacher forcing
                if random.random() < teacher_forcing_ratio:
                    decoder_input_pos = position_seq_target[:, t, :]
                    decoder_input_vel = velocity_seq_target[:, t, :]

                # predict recursively
                else:
                    decoder_input_pos = decoder_output_position
                    decoder_input_vel = decoder_output_velocity

        return predictions_position, predictions_velocity

    def _shared_step(self, batch, batch_idx, stage):
        (
            _,
            input_bboxes,
            input_velocities,
            _,
            output_bboxes,
            output_velocities,
        ) = batch

        # Predict future velocities
        if stage == "train":
            predicted_positions, predicted_velocity = self(
                input_bboxes,
                input_velocities,
                position_seq_target=output_bboxes,
                velocity_seq_target=output_velocities,
            )
        else:
            predicted_positions, predicted_velocity = self(
                input_bboxes, input_velocities
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

        # Log losses
        self.log_dict(
            {
                f"{stage}_vel_loss": velocity_loss,
                f"{stage}_pos_loss": pos_loss,
                f"{stage}_vel_to_pos_loss": velocities_to_positions_loss,
                f"{stage}_loss": total_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # Update metrics
        metrics_monitor = getattr(self, f"{stage}_metrics")
        metrics_monitor.update(
            predicted_bbox=predicted_positions,
            predicted_bbox_from_vel=velocities_to_positions,
            ground_truth_bbox=output_bboxes,
        )

        return total_loss

    def on_train_epoch_end(self):
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        self._log_metrics("val")

    def on_test_epoch_end(self):
        self._log_metrics("test")

    def _log_metrics(self, stage: str):
        metrics_monitor = getattr(self, f"{stage}_metrics")
        metrics = metrics_monitor.compute()
        self.log_dict(
            {f"{stage}_{k}": v for k, v in metrics.items()},
            on_epoch=True,
            prog_bar=True,
        )
        metrics_monitor.reset()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
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
