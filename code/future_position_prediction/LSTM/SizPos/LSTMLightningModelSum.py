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


class LSTMLightningModelSum(L.LightningModule):
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
        super(LSTMLightningModelSum, self).__init__()
        self.save_hyperparameters()

        # Define LSTM Encoders
        self.position_encoder = LSTMEncoder(
            input_size=position_size,
            hidden_size=hidden_size,
            num_layers=hidden_depth,
        )
        self.size_encoder = LSTMEncoder(
            input_size=size_size, hidden_size=hidden_size, num_layers=hidden_depth
        )

        # Define LSTM Decoders
        self.decoder_position = LSTMDecoder(
            input_size=position_size,
            hidden_size=hidden_size,
            output_size=position_size,
            num_layers=hidden_depth,
            dropout=dropout,
        )
        self.decoder_size = LSTMDecoder(
            input_size=size_size,
            hidden_size=hidden_size,
            output_size=size_size,
            num_layers=hidden_depth,
            dropout=dropout,
        )

    def forward(
        self,
        position_seq,
        size_seq,
        teacher_forcing_ratio=0.5,
        training_prediction="recursive",
        position_seq_target=None,
        size_seq_target=None,
    ):
       # Encode the Position and Size
        _, (encoder_hidden_states_position, encoder_cell_states_position) = (
            self.position_encoder(position_seq)
        )
        _, (encoder_hidden_states_size, encoder_cell_states_size) = self.size_encoder(
            size_seq
        )

        # Combine the outputs and states (Average)
        hidden_states_size = hidden_states_pos = (
            encoder_hidden_states_position + encoder_hidden_states_size
        ) / 2
        cell_states_size = cell_states_pos = (
            encoder_cell_states_position + encoder_cell_states_size
        ) / 2
        
        # decoder with teacher forcing
        decoder_input_position = position_seq[:, -1, :]
        decoder_input_size = size_seq[:, -1, :]

        predictions_position = torch.tensor([], device=position_seq.device)
        predictions_size = torch.tensor([], device=size_seq.device)

        if training_prediction == "recursive":
            # predict recursively
            for _ in range(self.hparams.output_frames):
                decoder_output_position, (
                    hidden_states_pos,
                    cell_states_pos,
                ) = self.decoder_position(
                    decoder_input_position, hidden_states_pos, cell_states_pos
                )
                decoder_output_size, (
                    hidden_states_size,
                    cell_states_size,
                ) = self.decoder_size(
                    decoder_input_size, hidden_states_size, cell_states_size
                )

                predictions_position = torch.cat(
                    (predictions_position, decoder_output_position.unsqueeze(1)), dim=1
                )
                predictions_size = torch.cat(
                    (predictions_size, decoder_output_size.unsqueeze(1)), dim=1
                )

                decoder_input_position = decoder_output_position.detach()
                decoder_input_size = decoder_output_size.detach()

        if training_prediction == "teacher_forcing":
            # use teacher forcing
            if random.random() < teacher_forcing_ratio:
                for t in range(self.hparams.output_frames):
                    decoder_output_position, (
                        hidden_states_pos,
                        cell_states_pos,
                    ) = self.decoder_position(
                        decoder_input_position, hidden_states_pos, cell_states_pos
                    )
                    decoder_output_size, (
                        hidden_states_size,
                        cell_states_size,
                    ) = self.decoder_size(
                        decoder_input_size, hidden_states_size, cell_states_size
                    )

                    predictions_position = torch.cat(
                        (predictions_position, decoder_output_position.unsqueeze(1)),
                        dim=1,
                    )
                    predictions_size = torch.cat(
                        (predictions_size, decoder_output_size.unsqueeze(1)),
                        dim=1,
                    )

                    decoder_input_position = position_seq_target[:, t, :]
                    decoder_input_size = size_seq_target[:, t, :]

            # predict recursively
            else:
                for t in range(self.hparams.output_frames):
                    decoder_output_position, (
                        hidden_states_pos,
                        cell_states_pos,
                    ) = self.decoder_position(
                        decoder_input_position, hidden_states_pos, cell_states_pos
                    )
                    decoder_output_size, (
                        hidden_states_size,
                        cell_states_size,
                    ) = self.decoder_size(
                        decoder_input_size, hidden_states_size, cell_states_size
                    )

                    predictions_position = torch.cat(
                        (predictions_position, decoder_output_position.unsqueeze(1)),
                        dim=1,
                    )
                    predictions_size = torch.cat(
                        (predictions_size, decoder_output_size.unsqueeze(1)),
                        dim=1,
                    )

                    decoder_input_position = decoder_output_position.detach()
                    decoder_input_size = decoder_output_size.detach()

        if training_prediction == "mixed_teacher_forcing":
            # predict using mixed teacher forcing
            for t in range(self.hparams.output_frames):
                decoder_output_position, (
                    hidden_states_pos,
                    cell_states_pos,
                ) = self.decoder_position(
                    decoder_input_position, hidden_states_pos, cell_states_pos
                )
                decoder_output_size, (
                    hidden_states_size,
                    cell_states_size,
                ) = self.decoder_size(
                    decoder_input_size, hidden_states_size, cell_states_size
                )

                predictions_position = torch.cat(
                    (predictions_position, decoder_output_position.unsqueeze(1)), dim=1
                )
                predictions_size = torch.cat(
                    (predictions_size, decoder_output_size.unsqueeze(1)), dim=1
                )

                # predict with teacher forcing
                if random.random() < teacher_forcing_ratio:
                    decoder_input_position = position_seq_target[:, t, :]
                    decoder_input_size = size_seq_target[:, t, :]

                # predict recursively
                else:
                    decoder_input_position = decoder_output_position.detach()
                    decoder_input_size = decoder_output_size.detach()

        return predictions_position, predictions_size

    def _shared_step(self, batch, batch_idx, stage):
        (
            video_id,
            input_positions,
            input_sizes,
            output_positions,
            output_sizes,
        ) = batch

        # Predict future positions and sizes
        if stage == "train":
            predicted_positions, predicted_sizes = self(
                input_positions,
                input_sizes,
                training_prediction="mixed_teacher_forcing",
                position_seq_target=output_positions,
                size_seq_target=output_sizes,
                teacher_forcing_ratio=0.6,
            )
        else:
            predicted_positions, predicted_sizes = self(input_positions, input_sizes)

        # Convert predicted_positions and predicted_sizes to YOLO format
        predicted_bboxes = torch.zeros_like(predicted_positions[:, :, :4])
        predicted_velocities = torch.zeros_like(predicted_positions[:, :, :4])

        # YOLO format: xcenter, ycenter, width, height
        predicted_bboxes[:, :, 0] = predicted_positions[:, :, 0]  # xcenter
        predicted_bboxes[:, :, 1] = predicted_positions[:, :, 1]  # ycenter
        predicted_bboxes[:, :, 2] = predicted_sizes[:, :, 0]  # width
        predicted_bboxes[:, :, 3] = predicted_sizes[:, :, 1]  # height

        # Velocities: velx, vely, deltaw, deltah
        predicted_velocities[:, :, 0] = predicted_positions[:, :, 2]  # velx
        predicted_velocities[:, :, 1] = predicted_positions[:, :, 3]  # vely
        predicted_velocities[:, :, 2] = predicted_sizes[:, :, 2]  # deltaw
        predicted_velocities[:, :, 3] = predicted_sizes[:, :, 3]  # deltah

        # Input positions to YOLO format
        input_bboxes = torch.zeros_like(input_positions[:, :, :4])
        input_bboxes[:, :, 0] = input_positions[:, :, 0]  # xcenter
        input_bboxes[:, :, 1] = input_positions[:, :, 1]  # ycenter
        input_bboxes[:, :, 2] = input_sizes[:, :, 0]  # width
        input_bboxes[:, :, 3] = input_sizes[:, :, 1]  # height

        # Output positions to YOLO format
        output_bboxes = torch.zeros_like(output_positions[:, :, :4])
        output_bboxes[:, :, 0] = output_positions[:, :, 0]  # xcenter
        output_bboxes[:, :, 1] = output_positions[:, :, 1]  # ycenter
        output_bboxes[:, :, 2] = output_sizes[:, :, 0]  # width
        output_bboxes[:, :, 3] = output_sizes[:, :, 1]  # height

        ground_truth_velocities = torch.zeros_like(output_positions[:, :, :4])
        ground_truth_velocities[:, :, 0] = output_positions[:, :, 2]  # velx
        ground_truth_velocities[:, :, 1] = output_positions[:, :, 3]  # vely
        ground_truth_velocities[:, :, 2] = output_sizes[:, :, 2]  # deltaw
        ground_truth_velocities[:, :, 3] = output_sizes[:, :, 3]  # deltah

        # Convert predicted future velocities to future positions
        velocities_to_positions = convert_velocity_to_positions(
            predicted_velocity=predicted_velocities,
            past_positions=input_bboxes
        )

        # Compute losses
        x_loss = F.mse_loss(
            predicted_positions[:, :, 0], output_positions[:, :, 0], reduction="mean"
        )
        y_loss = F.mse_loss(
            predicted_positions[:, :, 1], output_positions[:, :, 1], reduction="mean"
        )
        velx_loss = F.mse_loss(
            predicted_positions[:, :, 2], output_positions[:, :, 2], reduction="mean"
        )
        vely_loss = F.mse_loss(
            predicted_positions[:, :, 3], output_positions[:, :, 3], reduction="mean"
        )
        accx_loss = F.mse_loss(
            predicted_positions[:, :, 4], output_positions[:, :, 4], reduction="mean"
        )
        accy_loss = F.mse_loss(
            predicted_positions[:, :, 5], output_positions[:, :, 5], reduction="mean"
        )
        w_loss = F.mse_loss(
            predicted_sizes[:, :, 0], output_sizes[:, :, 0], reduction="mean"
        )
        h_loss = F.mse_loss(
            predicted_sizes[:, :, 1], output_sizes[:, :, 1], reduction="mean"
        )
        deltaw_loss = F.mse_loss(
            predicted_sizes[:, :, 2], output_sizes[:, :, 2], reduction="mean"
        )
        deltah_loss = F.mse_loss(
            predicted_sizes[:, :, 3], output_sizes[:, :, 3], reduction="mean"
        )

        position_loss = x_loss + y_loss + velx_loss + vely_loss + accx_loss + accy_loss
        size_loss = w_loss + h_loss + deltaw_loss + deltah_loss
        total_loss = position_loss + size_loss

        self.log(f"{stage}_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_position_loss", position_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_size_loss", size_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_x_loss", x_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_y_loss", y_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_velx_loss", velx_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_vely_loss", vely_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_accx_loss", accx_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_accy_loss", accy_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_w_loss", w_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_h_loss", h_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_deltaw_loss", deltaw_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_deltah_loss", deltah_loss, on_step=False, on_epoch=True)

        # Compute metrics
        ade = compute_ADE(
            velocities_to_positions, output_bboxes, self.hparams.image_size
        )
        fde = compute_FDE(
            velocities_to_positions, output_bboxes, self.hparams.image_size
        )
        aiou_from_vel = compute_AIOU(velocities_to_positions, output_bboxes)
        fiou_from_vel = compute_FIOU(velocities_to_positions, output_bboxes)
        aiou = compute_AIOU(predicted_bboxes, output_bboxes)
        fiou = compute_FIOU(predicted_bboxes, output_bboxes)

        # Log metrics
        self.log(f"{stage}_ADE", ade, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_FDE", fde, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_AIoU", aiou, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_FIoU", fiou, on_step=False, on_epoch=True, prog_bar=True)
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
            prog_bar=True,
        )

        return {
            "total_loss": total_loss,
            "position_loss": position_loss,
            "size_loss": size_loss,
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
                "monitor": "val_total_loss",
                "interval": "epoch",
            },
        }
