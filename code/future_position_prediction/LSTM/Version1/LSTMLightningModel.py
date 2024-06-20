import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchvision.ops as ops


class LSTMLightningModel(L.LightningModule):
    """
    A PyTorch Lightning module for an LSTM-based model.

    Args:
        lr (float, optional): Learning rate. Defaults to 1e-4.
        batch_size (int, optional): Batch size. Defaults to 32.
        input_dim (int, optional): Input dimensionality. Defaults to 4.
        hidden_dim (int, optional): Hidden dimensionality. Defaults to 64.
        output_dim (int, optional): Output dimensionality. Defaults to 4.
        hidden_depth (int, optional): Number of LSTM layers. Defaults to 2.
    """

    def __init__(
        self,
        lr=1e-4,
        input_frames=10,
        output_frames=10,
        batch_size=32,
        input_dim=8,
        hidden_dim=128,
        output_dim=8,
        hidden_depth=3,
        dropout=0.2,
    ):
        super(LSTMLightningModel, self).__init__()
        self.save_hyperparameters()  # Automatically logs and saves hyperparameters for reproducibility

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
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        gru_out, hidden = self.gru(x)  # hidden state is of shape (num_layers, batch_size, hidden_dim)
        hidden = hidden[-1]  # Use the hidden state of the last layer
        hidden = self.dropout(hidden)
        feature_vector = self.fc1(hidden)
        out = self.fc2(feature_vector)
        #out[:, :4] = torch.sigmoid(out[:, :4])  # Constrain bbox predictions to [0, 1]
        return out

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        return self._shared_step(batch, batch_idx, "test")

    def on_training_epoch_end(self):
        """
        Called at the end of a training epoch.
        """
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        """
        Called at the end of a validation epoch.
        """
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        """
        Called at the end of a test epoch.
        """
        self._on_epoch_end("test")

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training the model.

        Returns:
            dict: The dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def _shared_step(self, batch, batch_idx, stage):
        """
        Shared step for training, validation, and test.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.
            stage (str): Stage (train, val, or test).

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y_target = batch
        y_pred = self(x)

        # print("y_pred", y_pred)
        #print("y_target", y_target)

        #print("y_pred", y_pred)
        #print("y_target", y_target)

        # Extract [x, y, w, h] from y_pred and y_target
        y_pred_boxes = y_pred[:, :4]
        y_target_boxes = y_target[:, :4]

        # Extract [vx, vy, dw, dh] from y_pred and y_target
        y_pred_values = y_pred[:, 4:]
        y_target_values = y_target[:, 4:]

        # Compute IoU loss
        iou_loss = ops.generalized_box_iou_loss(
            y_pred_boxes, y_target_boxes, reduction="mean"
        )

        # Compute MSE loss for other values
        mse_loss = F.mse_loss(y_pred_values, y_target_values, reduction="mean")

        # Combine losses
        total_loss = iou_loss + mse_loss

        preds = y_pred_boxes.detach().cpu()
        targets = y_target_boxes.detach().cpu()

        class_labels = torch.zeros(preds.size(0), dtype=torch.int)
        preds = [{"boxes": preds, "labels": class_labels}]
        targets = [{"boxes": targets, "labels": class_labels}]
        getattr(self, f"{stage}_iou").update(preds, targets)

        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return total_loss

    def _on_epoch_end(self, stage):
        """
        Called at the end of an epoch for a specific stage.

        Args:
            stage (str): Stage (train, val, or test).
        """
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
