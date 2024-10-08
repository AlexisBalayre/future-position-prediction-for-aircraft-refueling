import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from typing import Tuple, Dict

from .utils import (
    compute_ADE,
    compute_FDE,
    compute_AIOU,
    compute_FIOU,
    convert_velocity_to_positions,
)

from .MetricsMonitoring import MetricsMonitoring
from .GRUNet.DecoderGRU import DecoderGRU
from .GRUNet.EncoderGRU import EncoderGRU


class PosVelAccGRULightningModelConcat(L.LightningModule):
    """
    Main PyTorch Lightning module of the PosVelAcc-GRU model that predicts future bounding box positions
    and velocities by concatenating hidden states from separate encoders for bounding boxes,
    velocities, and accelerations.

    Args:
        lr (float, optional): Learning rate. Default is 1e-4.
        input_frames (int, optional): Number of input frames. Default is 10.
        output_frames (int, optional): Number of output frames to predict. Default is 10.
        batch_size (int, optional): Batch size. Default is 32.
        bbox_dim (int, optional): Dimension of the bounding box (typically 4: x, y, width, height). Default is 4.
        hidden_dim (int, optional): Dimension of the hidden layers in GRU. Default is 256.
        dropout (float, optional): Dropout rate. Default is 0.1.
        image_size (Tuple[int, int], optional): Size of the input images. Default is (640, 480).
        encoder_layer_nb (int, optional): Number of layers in the GRU encoder. Default is 1.
        decoder_layer_nb (int, optional): Number of layers in the GRU decoder. Default is 1.
        scheduler_patience (int, optional): Patience for learning rate scheduler. Default is 5.
        scheduler_factor (float, optional): Factor for learning rate reduction. Default is 0.5.
    """

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
        super().__init__()
        self.save_hyperparameters()

        # Initialise GRU-based encoders
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

        # Initialise GRU-based decoders
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

        # Combine hidden states from different encoders
        self.combine_hidden = nn.Linear(hidden_dim * 3, hidden_dim)

        # Initialise metrics monitoring
        self.train_metrics = MetricsMonitoring(image_size)
        self.val_metrics = MetricsMonitoring(image_size)
        self.test_metrics = MetricsMonitoring(image_size)

    def combine_hidden_states(
        self,
        hidden_bbox: torch.Tensor,
        hidden_velocity: torch.Tensor,
        hidden_acceleration: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combines hidden states from bounding box, velocity, and acceleration encoders.

        Args:
            hidden_bbox (torch.Tensor): Hidden state from the bounding box encoder.
            hidden_velocity (torch.Tensor): Hidden state from the velocity encoder.
            hidden_acceleration (torch.Tensor): Hidden state from the acceleration encoder.

        Returns:
            torch.Tensor: Combined hidden state.
        """
        combined = torch.cat(
            (hidden_bbox, hidden_velocity, hidden_acceleration), dim=-1
        )
        return self.combine_hidden(combined)

    def forward(
        self,
        bbox_seq: torch.Tensor,
        velocity_seq: torch.Tensor,
        acceleration_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            bbox_seq (torch.Tensor): Input sequence of bounding boxes (batch_size, input_frames, bbox_dim).
            velocity_seq (torch.Tensor): Input sequence of velocities (batch_size, input_frames, bbox_dim).
            acceleration_seq (torch.Tensor): Input sequence of accelerations (batch_size, input_frames, bbox_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted positions and velocities.
        """
        batch_size = bbox_seq.size(0)
        device = bbox_seq.device

        # Initialise hidden states
        h_pos = torch.zeros(
            self.bbox_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=device,
        )
        h_vel = torch.zeros(
            self.vel_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=device,
        )
        h_acc = torch.zeros(
            self.acc_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=device,
        )

        # Encode the Position, Velocity, and Acceleration
        _, encoder_hidden_states_bbox = self.bbox_encoder(bbox_seq, h_pos)
        _, encoder_hidden_states_vel = self.vel_encoder(velocity_seq, h_vel)
        _, encoder_hidden_states_acc = self.acc_encoder(acceleration_seq, h_acc)

        # Combine hidden states
        h_pos = h_vel = self.combine_hidden_states(
            encoder_hidden_states_bbox,
            encoder_hidden_states_vel,
            encoder_hidden_states_acc,
        )

        # Decoder with teacher forcing
        decoder_input_pos = bbox_seq[:, -1, :]
        decoder_input_vel = velocity_seq[:, -1, :]

        predictions_position = []
        predictions_velocity = []

        for _ in range(self.hparams.output_frames):
            out_pos, h_pos = self.pos_decoder(decoder_input_pos, h_pos)
            out_vel, h_vel = self.vel_decoder(decoder_input_vel, h_vel)

            predictions_position.append(out_pos.unsqueeze(1))
            predictions_velocity.append(out_vel.unsqueeze(1))

            decoder_input_pos = out_pos.detach()
            decoder_input_vel = out_vel.detach()

        predictions_position = torch.cat(predictions_position, dim=1)
        predictions_velocity = torch.cat(predictions_velocity, dim=1)

        return predictions_position, predictions_velocity

    def _shared_step(self, batch, batch_idx: int, stage: str) -> torch.Tensor:
        """
        A shared step function for training, validation, and test steps.

        Args:
            batch (Tuple): A batch of data containing video_id, input sequences, and output sequences.
            batch_idx (int): Index of the current batch.
            stage (str): Stage name, either "train", "val", or "test".

        Returns:
            torch.Tensor: The total loss for the batch.
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
        Called at the end of the training epoch to log training metrics.
        """
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to log validation metrics.
        """
        self._log_metrics("val")

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to log test metrics.
        """
        self._log_metrics("test")

    def _log_metrics(self, stage: str):
        """
        Logs the computed metrics at the end of an epoch.

        Args:
            stage (str): The stage for which to log metrics, either "train", "val", or "test".
        """
        metrics_monitor = getattr(self, f"{stage}_metrics")
        metrics = metrics_monitor.compute()
        self.log_dict(
            {f"{stage}_{k}": v for k, v in metrics.items()},
            on_epoch=True,
            prog_bar=True,
        )
        metrics_monitor.reset()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step function.

        Args:
            batch (Tuple): A batch of training data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The total loss for the batch.
        """
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Validation step function.

        Args:
            batch (Tuple): A batch of validation data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The total loss for the batch.
        """
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Test step function.

        Args:
            batch (Tuple): A batch of test data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The total loss for the batch.
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
