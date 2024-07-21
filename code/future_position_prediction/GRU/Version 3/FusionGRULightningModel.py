import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from typing import Tuple
import math

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
from GRUNet.OpticalFlowGRUEncoder import OpticalFlowGRUEncoder


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

        self.position_velocity_gru = nn.GRU(
            input_size=8,  # 4 for position, 4 for velocity
            hidden_size=hidden_dim,
            num_layers=encoder_layer_nb,
            batch_first=True,
            dropout=dropout if encoder_layer_nb > 1 else 0,
        )

        self.conv3d = nn.Conv3d(
            in_channels=2,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        conv_output_size = self._get_conv_output_size(image_size)
        self.fc_optical_flow = nn.Linear(conv_output_size, hidden_dim)

        self.fusion_gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=encoder_layer_nb,
            batch_first=True,
            dropout=dropout if encoder_layer_nb > 1 else 0,
        )

        self.decoder_gru = nn.GRU(
            input_size=bbox_dim,
            hidden_size=hidden_dim,
            num_layers=decoder_layer_nb,
            batch_first=True,
            dropout=dropout if decoder_layer_nb > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_dim, bbox_dim)

        self.dropout = nn.Dropout(dropout)

        self.hidden_states: Dict[
            str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _get_conv_output_size(self, image_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, self.hparams.input_frames, *image_size)
            dummy_output = self.pool3d(self.conv3d(dummy_input))
            return (
                dummy_output.numel() // dummy_output.shape[0] // dummy_output.shape[2]
            )

    def forward(self, bboxes_position, bboxes_velocity, optical_flow, video_id):
        batch_size = bboxes_position.size(0)
        seq_len = bboxes_position.size(1)

        # Initialize or retrieve hidden states
        if video_id not in self.hidden_states:
            h_pv = torch.zeros(
                self.position_velocity_gru.num_layers,
                batch_size,
                self.hparams.hidden_dim,
                device=self.device,
            )
            h_fusion = torch.zeros(
                self.fusion_gru.num_layers,
                batch_size,
                self.hparams.hidden_dim,
                device=self.device,
            )
            h_decoder = torch.zeros(
                self.decoder_gru.num_layers,
                batch_size,
                self.hparams.hidden_dim,
                device=self.device,
            )
        else:
            h_pv, h_fusion, h_decoder = self.hidden_states[video_id]
            h_pv = h_pv.detach()
            h_fusion = h_fusion.detach()
            h_decoder = h_decoder.detach()

        # Process position and velocity
        position_velocity = torch.cat((bboxes_position, bboxes_velocity), dim=-1)
        position_velocity_out, h_pv = self.position_velocity_gru(
            position_velocity, h_pv
        )

        # Process optical flow
        optical_flow = optical_flow.permute(
            0, 4, 1, 2, 3
        )  # (batch, channels, seq, height, width)
        optical_flow = self.conv3d(optical_flow)
        optical_flow = self.pool3d(optical_flow)
        optical_flow = optical_flow.view(batch_size, seq_len, -1)
        optical_flow = self.fc_optical_flow(optical_flow)

        # Fuse features
        fused_features = torch.cat((position_velocity_out, optical_flow), dim=-1)
        fused_out, h_fusion = self.fusion_gru(fused_features, h_fusion)

        # Decoder loop
        outputs = []
        decoder_input = bboxes_velocity[:, -1, :]  # Start with the last known velocity
        for _ in range(self.hparams.output_frames):
            decoder_input = decoder_input.unsqueeze(1)  # Add sequence dimension
            decoder_out, h_decoder = self.decoder_gru(decoder_input, h_decoder)
            decoder_out = self.fc_out(self.dropout(decoder_out.squeeze(1)))
            outputs.append(decoder_out)
            decoder_input = decoder_out  # Use current prediction as next input

        output = torch.stack(outputs, dim=1)  # Stack all predictions

        # Update hidden states
        self.hidden_states[video_id] = (h_pv, h_fusion, h_decoder)

        return output

    def _shared_step(self, batch, batch_idx, stage):
        (
            video_id,
            input_bboxes_position,
            input_bboxes_velocity,
            input_optical_flow,
            target_bboxes_position,
            target_bboxes_velocity,
        ) = batch

        predicted_velocity = self(
            input_bboxes_position, input_bboxes_velocity, input_optical_flow, video_id
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

        # Compute losses
        velocities_to_positions_loss = F.smooth_l1_loss(
            velocities_to_positions_norm, target_bboxes_position
        )
        velocity_loss = F.smooth_l1_loss(predicted_velocity, target_bboxes_velocity)
        total_loss = velocity_loss + velocities_to_positions_loss * 0.1

        # Log losses
        """ self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False
        ) """

        # Compute metrics
        ade_from_vel = compute_ADE(
            velocities_to_positions, target_bboxes_position_denorm
        )
        fde_from_vel = compute_FDE(
            velocities_to_positions, target_bboxes_position_denorm
        )
        aiou_from_vel = compute_AIOU(
            velocities_to_positions_norm, target_bboxes_position
        )
        fiou_from_vel = compute_FIOU(
            velocities_to_positions_norm, target_bboxes_position
        )

        """ # Log metrics
        self.log(
            f"{stage}_ADE",
            ade_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_FDE",
            fde_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            f"{stage}_AIoU",
            aiou_from_vel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        ) """
        self.log(
            f"{stage}_FIoU",
            fiou_from_vel,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return {
            "total_loss": total_loss,
            "ADE": ade_from_vel,
            "FDE": fde_from_vel,
            "AIoU": aiou_from_vel,
            "FIoU": fiou_from_vel,
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
