import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from typing import Tuple, Dict
import random

from .utils import (
    compute_ADE,
    compute_FDE,
    compute_AIOU,
    compute_FIOU,
    convert_velocity_to_positions,
)

from .MetricsMonitoring import MetricsMonitoring
from .LSTM.LSTMEncoder import LSTMEncoder
from .LSTM.LSTMDecoder import LSTMDecoder


class LSTMPosVelAccLightningModelSum(L.LightningModule):
    """
    Main PyTorch Lightning module of the PosVelAcc-LSTM model that predicts future bounding box positions
    and velocities by summing the hidden states from separate encoders for bounding boxes,
    velocities, and accelerations.

    Args:
        lr (float, optional): Learning rate. Default is 1e-4.
        input_frames (int, optional): Number of input frames. Default is 10.
        output_frames (int, optional): Number of output frames to predict. Default is 10.
        batch_size (int, optional): Batch size. Default is 32.
        bbox_size (int, optional): Dimension of the bounding box input (typically 4: x, y, width, height). Default is 4.
        hidden_size (int, optional): Dimension of the hidden layers in LSTM. Default is 256.
        hidden_depth (int, optional): Number of layers in the LSTM encoder and decoder. Default is 3.
        dropout (float, optional): Dropout rate. Default is 0.1.
        image_size (Tuple[int, int], optional): Size of the input images. Default is (640, 480).
        scheduler_patience (int, optional): Patience for learning rate scheduler. Default is 10.
        scheduler_factor (float, optional): Factor for learning rate reduction. Default is 0.9.
    """

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
        super(LSTMPosVelAccLightningModelSum, self).__init__()
        self.save_hyperparameters()

        # Define LSTM Encoders for position, velocity, and acceleration
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

        # Define LSTM Decoders for position and velocity
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

        # Initialise metrics monitoring
        self.train_metrics = MetricsMonitoring(image_size)
        self.val_metrics = MetricsMonitoring(image_size)
        self.test_metrics = MetricsMonitoring(image_size)

    def forward(
        self,
        bbox_seq: torch.Tensor,
        velocity_seq: torch.Tensor,
        acceleration_seq: torch.Tensor,
        teacher_forcing_ratio: float = 0.6,
        training_prediction: str = "recursive",
        position_seq_target: torch.Tensor = None,
        velocity_seq_target: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            bbox_seq (torch.Tensor): Input sequence of bounding boxes (batch_size, input_frames, bbox_size).
            velocity_seq (torch.Tensor): Input sequence of velocities (batch_size, input_frames, bbox_size).
            acceleration_seq (torch.Tensor): Input sequence of accelerations (batch_size, input_frames, bbox_size).
            teacher_forcing_ratio (float, optional): Ratio of teacher forcing to use during training. Default is 0.6.
            training_prediction (str, optional): Prediction mode, either 'recursive', 'teacher_forcing', or 'mixed_teacher_forcing'. Default is 'recursive'.
            position_seq_target (torch.Tensor, optional): Target sequence of positions for teacher forcing.
            velocity_seq_target (torch.Tensor, optional): Target sequence of velocities for teacher forcing.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted positions and velocities.
        """
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

        # Combine the outputs and states (Average)
        hidden_states_pos = hidden_states_vel = (
            encoder_hidden_states_vel
            + encoder_hidden_states_acc
            + encoder_hidden_states_bbox
        )
        cell_states_pos = cell_states_vel = (
            encoder_cell_states_vel + encoder_cell_states_acc + encoder_cell_states_bbox
        )

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

    def _shared_step(self, batch: Tuple, batch_idx: int, stage: str) -> torch.Tensor:
        """
        Shared step function for training, validation, and testing.

        Args:
            batch (Tuple): A batch of data containing input sequences and target sequences.
            batch_idx (int): Index of the current batch.
            stage (str): Stage of the model ('train', 'val', 'test').

        Returns:
            torch.Tensor: Total loss computed for the batch.
        """
        (
            _,
            input_bboxes,
            input_velocities,
            input_accelerations,
            output_bboxes,
            output_velocities,
        ) = batch

        # Predict future velocities
        if stage == "train":
            predicted_positions, predicted_velocity = self(
                input_bboxes,
                input_velocities,
                input_accelerations,
                position_seq_target=output_bboxes,
                velocity_seq_target=output_velocities,
            )
        else:
            predicted_positions, predicted_velocity = self(
                input_bboxes, input_velocities, input_accelerations
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
        total_loss = velocity_loss + velocities_to_positions_loss + pos_loss

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
        """
        Function to be called at the end of each training epoch to log metrics.
        """
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        """
        Function to be called at the end of each validation epoch to log metrics.
        """
        self._log_metrics("val")

    def on_test_epoch_end(self):
        """
        Function to be called at the end of each test epoch to log metrics.
        """
        self._log_metrics("test")

    def _log_metrics(self, stage: str):
        """
        Logs metrics for the given stage (train, val, test).

        Args:
            stage (str): Stage of the model ('train', 'val', 'test').
        """
        metrics_monitor = getattr(self, f"{stage}_metrics")
        metrics = metrics_monitor.compute()
        self.log_dict(
            {f"{stage}_{k}": v for k, v in metrics.items()},
            on_epoch=True,
            prog_bar=True,
        )
        metrics_monitor.reset()

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step to be called during training.

        Args:
            batch (Tuple): A batch of data containing input sequences and target sequences.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Total loss computed for the batch.
        """
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation step to be called during validation.

        Args:
            batch (Tuple): A batch of data containing input sequences and target sequences.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Total loss computed for the batch.
        """
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Test step to be called during testing.

        Args:
            batch (Tuple): A batch of data containing input sequences and target sequences.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Total loss computed for the batch.
        """
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> Dict[str, Dict]:
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Dict]: Dictionary containing the optimizer and the learning rate scheduler.
        """
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
                "monitor": "train_loss",
                "interval": "epoch",
            },
        }
