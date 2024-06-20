import torch
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchvision.ops as ops

class GRULightningModel(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        input_frames=10,
        output_frames=10,
        batch_size=32,
        input_dim=8,  # Updated to 8 dimensions
        hidden_dim=128,
        output_dim=4,
        hidden_depth=3,
        dropout=0.2,
    ):
        super(GRULightningModel, self).__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            hidden_depth,
            batch_first=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_dim, 256)  # Fully connected layer to produce 256-dimensional vector
        self.fc2 = nn.Linear(256, output_dim)  # Fully connected layer to produce final output
        self.dropout = nn.Dropout(dropout)

        self.train_iou = IntersectionOverUnion(box_format="xywh")
        self.val_iou = IntersectionOverUnion(box_format="xywh")
        self.test_iou = IntersectionOverUnion(box_format="xywh")

    def forward(self, x):
        gru_out, hidden = self.gru(x)  # hidden state is of shape (num_layers, batch_size, hidden_dim)
        hidden = hidden[-1]  # Use the hidden state of the last layer
        hidden = self.dropout(hidden)
        feature_vector = self.fc1(hidden)
        out = self.fc2(feature_vector)
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
            optimizer, mode="min", factor=0.9, patience=100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def _shared_step(self, batch, batch_idx, stage):
        x, y_target = batch
        y_pred = self(x)

        # Ensure y_pred and y_target have the correct shape
        y_pred = y_pred.view(-1, 4)
        y_target = y_target.view(-1, 4)

        # Compute loss
        loss = ops.generalized_box_iou_loss(y_pred, y_target, reduction="mean")

        preds = y_pred.detach().cpu()
        targets = y_target.detach().cpu()

        class_labels = torch.zeros(preds.size(0), dtype=torch.int)
        preds = [{"boxes": preds, "labels": class_labels}]
        targets = [{"boxes": targets, "labels": class_labels}]
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