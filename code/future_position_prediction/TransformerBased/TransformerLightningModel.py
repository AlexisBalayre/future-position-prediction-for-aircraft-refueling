import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchvision.ops as ops
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerLightningModel(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        input_frames=10,
        output_frames=10,
        batch_size=32,
        bbox_dim=4,
        hidden_dim=256,
        output_dim=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=8,
        dropout=0.1,
    ):
        super(TransformerLightningModel, self).__init__()
        self.save_hyperparameters()

        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.input_proj = nn.Linear(bbox_dim * 2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, bbox_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bbox_dim),
            nn.Sigmoid(),
        )

        self.delta_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bbox_dim),
        )

        self.conf_fc = nn.Linear(hidden_dim, 1)

        self.train_iou = IntersectionOverUnion(box_format="cxcywh")
        self.val_iou = IntersectionOverUnion(box_format="cxcywh")
        self.test_iou = IntersectionOverUnion(box_format="cxcywh")

    def forward(self, bboxes, delta_bboxes):
        batch_size, seq_len, _ = bboxes.shape

        # Combine bboxes and delta_bboxes
        inputs = torch.cat([bboxes, delta_bboxes], dim=-1)
        inputs = self.input_proj(inputs)

        # Add positional encoding
        inputs = inputs.transpose(0, 1)  # Change to (seq_len, batch_size, hidden_dim)
        inputs = self.pos_encoder(inputs)

        # Transformer encoder
        memory = self.transformer_encoder(inputs)

        # Prepare decoder input (use the last frame of encoder output as initial input)
        decoder_input = memory[:, -1:, :]

        outputs = []
        delta_outputs = []
        conf_outputs = []

        for _ in range(self.hparams.output_frames):
            # Transformer decoder
            decoder_output = self.transformer_decoder(decoder_input, memory)

            # Project decoder output
            output = self.fc(decoder_output)
            delta_output = self.delta_fc(decoder_output)
            conf_output = self.conf_fc(decoder_output)

            output = torch.clamp(output, 0, 1)
            output = torch.nan_to_num(output, nan=0.0)

            outputs.append(output)
            delta_outputs.append(delta_output)
            conf_outputs.append(conf_output)

            # Use the current output as the next decoder input
            decoder_input = decoder_output

        # Stack outputs and transpose back to (batch_size, seq_len, dim)
        outputs = torch.cat(outputs, dim=0).transpose(0, 1)
        delta_outputs = torch.cat(delta_outputs, dim=0).transpose(0, 1)
        conf_outputs = torch.cat(conf_outputs, dim=0).transpose(0, 1)

        return outputs, delta_outputs, conf_outputs

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_iou",
                "interval": "epoch",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def _shared_step(self, batch, batch_idx, stage):
        input_bboxes, input_delta_bboxes, output_bboxes, output_delta_bboxes = batch

        predicted_bboxes, predicted_deltas, predicted_conf = self(
            input_bboxes, input_delta_bboxes
        )

        pred_xyxy = self._xywhn_to_xyxy(predicted_bboxes.reshape(-1, 4))
        target_xyxy = self._xywhn_to_xyxy(output_bboxes.reshape(-1, 4))

        giou_loss = ops.generalized_box_iou_loss(
            pred_xyxy, target_xyxy, reduction="none"
        )
        giou_loss = 1 - torch.exp(-giou_loss)
        bbox_loss = giou_loss.mean()

        delta_loss = F.smooth_l1_loss(predicted_deltas, output_delta_bboxes)

        ious = ops.box_iou(pred_xyxy, target_xyxy)
        diag_ious = torch.diagonal(ious, dim1=-2, dim2=-1)
        mean_iou = diag_ious.mean()

        pred_conf = predicted_conf.reshape(-1)
        target_conf = (diag_ious > 0.5).float().reshape(-1)
        conf_loss = self.focal_loss(pred_conf, target_conf)

        total_loss = bbox_loss + delta_loss + conf_loss

        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_bbox_loss",
            bbox_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_delta_loss",
            delta_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_conf_loss",
            conf_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        with torch.no_grad():
            batch_size = predicted_bboxes.shape[0]
            dummy_labels = torch.zeros(
                batch_size, dtype=torch.int64, device=predicted_bboxes.device
            )

            preds = [
                {
                    "boxes": predicted_bboxes[:, -1].cpu(),
                    "scores": torch.sigmoid(predicted_conf[:, -1]).cpu(),
                    "labels": dummy_labels.cpu(),
                }
            ]
            targets = [
                {"boxes": output_bboxes[:, -1].cpu(), "labels": dummy_labels.cpu()}
            ]
            getattr(self, f"{stage}_iou").update(preds, targets)

        return total_loss

    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def _xywhn_to_xyxy(self, x):
        x = torch.clamp(x, 0, 1)
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return torch.clamp(y, 0, 1)

    def _on_epoch_end(self, stage):
        iou = getattr(self, f"{stage}_iou").compute()
        iou_value = iou["iou"]
        self.log(
            f"{stage}_iou",
            iou_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        getattr(self, f"{stage}_iou").reset()
