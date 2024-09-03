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
    convert_PosSize_to_PosVel,
)

from .MetricsMonitoring import MetricsMonitoring
from .GRUNet.DecoderGRU import DecoderGRU
from .GRUNet.EncoderGRU import EncoderGRU


class SizPosGRULightningModelAverage(L.LightningModule):
    """
    Main PyTorch Lightning module of the SizPos-GRU model that predicts future bounding box 
    positions and sizes by averaging the hidden states from the position and size encoders.

    Args:
        lr (float, optional): Learning rate. Default is 1e-4.
        input_frames (int, optional): Number of input frames. Default is 10.
        output_frames (int, optional): Number of output frames to predict. Default is 10.
        batch_size (int, optional): Batch size. Default is 32.
        position_dim (int, optional): Dimension of the position input. Default is 6.
        size_dim (int, optional): Dimension of the size input. Default is 4.
        hidden_dim (int, optional): Dimension of the hidden layers in GRU. Default is 256.
        hidden_depth (int, optional): Number of layers in the GRU encoders and decoders. Default is 3.
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
        position_dim: int = 6, 
        size_dim: int = 4, 
        hidden_dim: int = 256,
        hidden_depth: int = 3,
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (640, 480),
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.9,
    ):
        super(SizPosGRULightningModelAverage, self).__init__()
        self.save_hyperparameters()

        # Initialise the GRU-based encoders
        self.position_encoder = EncoderGRU(
            input_dim=position_dim,
            hidden_dim=hidden_dim,
            n_layers=hidden_depth,
            output_frames_nb=output_frames,
            dropout=dropout,
        )
        self.size_encoder = EncoderGRU(
            input_dim=size_dim,
            hidden_dim=hidden_dim,
            n_layers=hidden_depth,
            output_frames_nb=output_frames,
            dropout=dropout,
        )

        # Initialise the GRU-based decoders
        self.pos_decoder = DecoderGRU(
            position_dim,
            hidden_dim,
            position_dim,
            n_layers=hidden_depth,
            dropout=[dropout, dropout],
        )
        self.size_decoder = DecoderGRU(
            size_dim,
            hidden_dim,
            size_dim,
            n_layers=hidden_depth,
            dropout=[dropout, dropout],
        )

        # Initialise metrics monitoring
        self.train_metrics = MetricsMonitoring(image_size)
        self.val_metrics = MetricsMonitoring(image_size)
        self.test_metrics = MetricsMonitoring(image_size)

    def forward(
        self,
        position_seq: torch.Tensor,
        size_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            position_seq (torch.Tensor): Input sequence of positions (batch_size, input_frames, position_dim).
            size_seq (torch.Tensor): Input sequence of sizes (batch_size, input_frames, size_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted positions and sizes.
        """
        batch_size = position_seq.size(0)

        # Initialise hidden states for the GRU encoders
        h_pos = torch.zeros(
            self.position_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=position_seq.device,
        )
        h_size = torch.zeros(
            self.size_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=size_seq.device,
        )

        # Encode the Position and Size sequences
        encoder_out_pos, encoder_hidden_states_pos = self.position_encoder(
            position_seq, h_pos
        )
        encoder_out_size, encoder_hidden_states_size = self.size_encoder(
            size_seq, h_size
        )

        # Combine the hidden states (average)
        h_pos = h_size = (encoder_hidden_states_pos + encoder_hidden_states_size) / 2

        # Initialise decoder inputs with the last frame of the input sequence
        decoder_input_pos = position_seq[:, -1, :]
        decoder_input_size = size_seq[:, -1, :]

        predictions_position = torch.tensor([], device=position_seq.device)
        predictions_size = torch.tensor([], device=size_seq.device)

        # Decoder loop for generating predictions
        for t in range(self.hparams.output_frames):
            out_pos, h_pos = self.pos_decoder(decoder_input_pos, h_pos)
            out_size, h_size = self.size_decoder(decoder_input_size, h_size)

            predictions_position = torch.cat(
                (predictions_position, out_pos.unsqueeze(1)), dim=1
            )
            predictions_size = torch.cat(
                (predictions_size, out_size.unsqueeze(1)), dim=1
            )

            # Teacher forcing: use the predicted output as the next input
            decoder_input_pos = out_pos.detach()
            decoder_input_size = out_size.detach()

        return predictions_position, predictions_size

    def _shared_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int, stage: str
    ) -> torch.Tensor:
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
            input_positions,
            input_sizes,
            ground_truth_positions,
            ground_truth_sizes,
        ) = batch

        # Predict future sizes
        predicted_positions, predicted_sizes = self(input_positions, input_sizes)

        # Convert predictions to positions and velocities
        predicted_bboxes, predicted_velocities = convert_PosSize_to_PosVel(
            predicted_positions, predicted_sizes
        )

        # Convert ground truth to positions and velocities
        ground_truth_bboxes, ground_truth_velocities = convert_PosSize_to_PosVel(
            ground_truth_positions, ground_truth_sizes
        )

        # Convert inputs to positions and velocities
        input_bboxes, _ = convert_PosSize_to_PosVel(input_positions, input_sizes)

        # Convert predicted future velocities to future positions
        velocities_to_positions = convert_velocity_to_positions(
            predicted_velocity=predicted_velocities, past_positions=input_bboxes
        )

        # Compute losses
        velocity_loss = F.smooth_l1_loss(predicted_velocities, ground_truth_velocities)
        pos_loss = F.smooth_l1_loss(predicted_bboxes, ground_truth_bboxes)
        velocities_to_positions_loss = F.smooth_l1_loss(
            velocities_to_positions, ground_truth_bboxes
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
            predicted_bbox=predicted_bboxes,
            predicted_bbox_from_vel=velocities_to_positions,
            ground_truth_bbox=ground_truth_bboxes,
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

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step function.

        Args:
            batch (Tuple): A batch of training data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The total loss for the batch.
        """
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step function.

        Args:
            batch (Tuple): A batch of validation data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The total loss for the batch.
        """
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
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
