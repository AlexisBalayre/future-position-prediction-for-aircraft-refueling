import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchvision.ops as ops

from ConvLSTM import ConvLSTM


class ConvLSTMLightningModel(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        input_frames=2,
        output_frames=2,
        batch_size=32,
        input_dim=3,
        hidden_dim=64,
        output_dim=4,
        hidden_depth=3,
        dropout=0.2,
    ):
        super(ConvLSTMLightningModel, self).__init__()
        self.save_hyperparameters()

        self.conv_lstm = ConvLSTM(
             input_channels=input_dim + 4,
            hidden_channels=[hidden_dim] * hidden_depth,
            kernel_size=[(3, 3)] * hidden_depth,
            num_layers=hidden_depth,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        self.fc = None
        self.dropout = nn.Dropout(dropout)

        self.train_iou = IntersectionOverUnion(box_format="xyxy")
        self.val_iou = IntersectionOverUnion(box_format="xyxy")
        self.test_iou = IntersectionOverUnion(box_format="xyxy")

    def forward(self, x, bboxes):
        # Reshape bboxes to match the dimensions of x
        b, seq_len, _, h, w = x.size()
        bboxes = bboxes.view(b, seq_len, 4, 1, 1).expand(b, seq_len, 4, h, w)
        x = torch.cat((x, bboxes), dim=2)  # Concatenate along the channel dimension

        lstm_out, _ = self.conv_lstm(x)
        lstm_out = self.dropout(lstm_out[0])
        lstm_out = lstm_out.view(b, seq_len, -1)  # Flatten the spatial dimensions

        # Initialize the fully connected layer if not already done
        if self.fc is None:
            self.fc = nn.Linear(lstm_out.size(-1), self.hparams.output_dim)
            self.fc = self.fc.to(lstm_out.device)

        out = self.fc(lstm_out)  # Apply fully connected layer to each time step
        return out

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def on_training_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def _shared_step(self, batch, batch_idx, stage):
        x, input_bboxes, y_target = batch
        y_pred = self(x, input_bboxes)

        # Compute loss across the sequence
        loss = 0
        for t in range(y_target.size(1)):
            loss += ops.generalized_box_iou_loss(
                y_pred[:, t], y_target[:, t], reduction="mean"
            )
        loss /= y_target.size(1)

        preds = y_pred.detach().cpu()
        targets = y_target.detach().cpu()

        class_labels = torch.zeros(preds.size(0), dtype=torch.int)
        preds = [
            {"boxes": preds[:, t], "labels": class_labels} for t in range(preds.size(1))
        ]
        targets = [
            {"boxes": targets[:, t], "labels": class_labels}
            for t in range(targets.size(1))
        ]
        getattr(self, f"{stage}_iou").update(preds, targets)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

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
