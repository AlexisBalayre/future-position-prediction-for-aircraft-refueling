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
    convert_PosSize_to_PosVel,
)

from MetricsMonitoring import MetricsMonitoring
from GRUNet.DecoderGRU import DecoderGRU
from GRUNet.EncoderGRU import EncoderGRU


class GRULightningModelWithOpticalFlow(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        input_frames: int = 10,
        output_frames: int = 10,
        batch_size: int = 32,
        position_dim: int = 6,  # [xcenter, ycenter, velxcenter, velycenter, accxcenter, accycenter]
        size_dim: int = 4,  # [w, h, deltaw, deltah]
        hidden_dim: int = 256,
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (480, 640),
        hidden_depth: int = 3,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
    ):
        super(GRULightningModelWithOpticalFlow, self).__init__()
        self.save_hyperparameters()

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

        # Optical flow encoder
        self.optical_flow_encoder = nn.Sequential(
            nn.Linear(2 * image_size[0] * image_size[1], 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GRU for processing sequence of optical flow features
        self.optical_flow_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=hidden_depth,
            batch_first=True,
            dropout=dropout,
        )

        self.pos_decoder = DecoderGRU(
            position_dim,
            hidden_dim,
            position_dim,
            n_layers=hidden_depth,
            dropout=[dropout, dropout],
        )
        self.size = DecoderGRU(
            size_dim,
            hidden_dim,
            size_dim,
            n_layers=hidden_depth,
            dropout=[dropout, dropout],
        )

        self.combine_hidden = nn.Linear(hidden_dim * 3, hidden_dim)

        # Initialize metrics monitoring
        self.train_metrics = MetricsMonitoring(image_size)
        self.val_metrics = MetricsMonitoring(image_size)
        self.test_metrics = MetricsMonitoring(image_size)

    def combine_hidden_states(self, hidden_pos, hidden_size, hidden_optical_flow):
        # Ensure all hidden states have the same batch dimension
        hidden_pos = hidden_pos[-1]  # Take the last layer's hidden state
        hidden_size = hidden_size[-1]
        hidden_optical_flow = hidden_optical_flow[-1]

        combined = torch.cat((hidden_pos, hidden_size, hidden_optical_flow), dim=-1)
        return self.combine_hidden(combined)

    def forward(
        self,
        pos_seq,
        size_seq,
        optical_flow_seq,
    ):
        batch_size, time_steps, height, width, _ = optical_flow_seq.shape

        # Initialize hidden states
        h_pos = torch.zeros(
            self.position_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=pos_seq.device,
        )
        h_size = torch.zeros(
            self.size_encoder.n_layers,
            batch_size,
            self.hparams.hidden_dim,
            device=size_seq.device,
        )

        # Encode the Position and Velocity
        _, encoder_hidden_states_pos = self.position_encoder(pos_seq, h_pos)
        _, encoder_hidden_states_size = self.size_encoder(size_seq, h_size)

        # Encode Optical Flow (process all frames)
        optical_flow_features = []
        for t in range(time_steps):
            optical_flow_t = optical_flow_seq[:, t].reshape(batch_size, -1)
            optical_flow_features.append(self.optical_flow_encoder(optical_flow_t))
        optical_flow_features = torch.stack(optical_flow_features, dim=1)
        _, encoder_hidden_states_flow = self.optical_flow_gru(optical_flow_features)

        # Combine hidden states
        h_combined = self.combine_hidden_states(
            encoder_hidden_states_pos,
            encoder_hidden_states_size,
            encoder_hidden_states_flow,
        )

        h_pos = h_size = h_combined.unsqueeze(0).repeat(self.pos_decoder.n_layers, 1, 1)

        # decoder with teacher forcing
        decoder_input_pos = pos_seq[:, -1, :]
        decoder_input_size = size_seq[:, -1, :]

        predictions_position = torch.tensor([], device=pos_seq.device)
        predictions_size = torch.tensor([], device=size_seq.device)

        for t in range(self.hparams.output_frames):
            out_pos, h_pos = self.pos_decoder(decoder_input_pos, h_pos)
            out_size, h_size = self.size(decoder_input_size, h_size)

            predictions_position = torch.cat(
                (predictions_position, out_pos.unsqueeze(1)), dim=1
            )
            predictions_size = torch.cat(
                (predictions_size, out_size.unsqueeze(1)), dim=1
            )

            decoder_input_pos = out_pos.detach()
            decoder_input_size = out_size.detach()

        return predictions_position, predictions_size

    def _shared_step(self, batch, batch_idx, stage):
        (
            _,
            input_positions,
            input_sizes,
            input_optical_flows,
            ground_truth_positions,
            ground_truth_sizes,
        ) = batch

        # Predict future positions and velocities
        predicted_positions, predicted_sizes = self(
            input_positions,
            input_sizes,
            input_optical_flows,
        )
        
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
        sizes_loss = F.smooth_l1_loss(predicted_sizes, ground_truth_sizes)
        positions_loss = F.smooth_l1_loss(predicted_positions, ground_truth_positions)
        total_loss = sizes_loss + positions_loss + velocity_loss + velocities_to_positions_loss * 0.1 + pos_loss

        # Log losses
        self.log_dict(
            {
                f"{stage}_sizes_loss": sizes_loss,
                f"{stage}_positions_loss": positions_loss,
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
