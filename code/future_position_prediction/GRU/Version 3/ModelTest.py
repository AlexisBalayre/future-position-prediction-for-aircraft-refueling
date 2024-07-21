import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Tuple
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class OpticalFlowEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        return x


class AdvancedFusionGRUModel(L.LightningModule):
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

        # Encoders
        self.bbox_encoder = nn.GRU(
            bbox_dim, hidden_dim, encoder_layer_nb, batch_first=True, dropout=dropout
        )
        self.of_encoder = OpticalFlowEncoder(input_channels=2, hidden_dim=hidden_dim)

        # Fusion attention
        self.fusion_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_layer_nb
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, bbox_dim)

        # Scheduled sampling
        self.scheduled_sampling_ratio = 1.0

    def forward(self, bboxes_position, bboxes_velocity, optical_flow):
        batch_size, seq_len, _, _, _ = optical_flow.size()

        # Encode bounding box information
        bbox_features, _ = self.bbox_encoder(bboxes_velocity)

        # Encode optical flow
        of_features = self.of_encoder(optical_flow.permute(0, 4, 1, 2, 3))
        of_features = of_features.unsqueeze(1).repeat(1, seq_len, 1)

        # Fuse bbox and optical flow features using attention
        fused_features, _ = self.fusion_attention(bbox_features, of_features, of_features)

        # Add positional encoding
        fused_features = self.pos_encoder(fused_features.permute(1, 0, 2))

        # Transformer decoding
        tgt = torch.zeros_like(fused_features)
        output = self.transformer_decoder(tgt, fused_features)

        # Predict velocities
        predicted_velocities = self.output_layer(output.permute(1, 0, 2))

        return predicted_velocities

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")

    def _shared_step(self, batch, batch_idx, stage):
        (
            input_bboxes_position,
            input_bboxes_velocity,
            input_optical_flow,
            target_bboxes_position,
            target_bboxes_velocity,
        ) = batch

        predicted_velocity = self(
            input_bboxes_position, input_bboxes_velocity, input_optical_flow
        )

        # Calculate losses
        mse_loss = F.mse_loss(predicted_velocity, target_bboxes_velocity)
        l1_loss = F.smooth_l1_loss(predicted_velocity, target_bboxes_velocity)

        # IoU loss (assuming predicted_velocity and target_bboxes_velocity are in [x, y, w, h] format)
        predicted_bbox = self._velocity_to_bbox(
            input_bboxes_position[:, -1, :], predicted_velocity
        )
        target_bbox = self._velocity_to_bbox(
            input_bboxes_position[:, -1, :], target_bboxes_velocity
        )
        iou_loss = 1 - self._bbox_iou(predicted_bbox, target_bbox).mean()

        # Combine losses
        loss = mse_loss + l1_loss + iou_loss

        # Log metrics
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mse_loss", mse_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_l1_loss", l1_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_iou_loss", iou_loss, on_step=False, on_epoch=True)

        return loss

    def _velocity_to_bbox(self, initial_bbox, velocity):
        return initial_bbox.unsqueeze(1) + torch.cumsum(velocity, dim=1)

    def _bbox_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1.unbind(-1)
        x2, y2, w2, h2 = bbox2.unbind(-1)

        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        intersection = torch.clamp(
            torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min), min=0
        ) * torch.clamp(torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min), min=0)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / (union + 1e-6)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
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

    def on_train_epoch_start(self):
        # Update scheduled sampling ratio
        self.scheduled_sampling_ratio = max(
            0.0, self.scheduled_sampling_ratio - 0.05
        )  # Decrease by 0.05 each epoch
