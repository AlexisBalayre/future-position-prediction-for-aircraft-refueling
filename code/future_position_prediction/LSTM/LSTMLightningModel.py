import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchmetrics


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
        batch_size=32,
        input_dim=4,
        hidden_dim=64,
        output_dim=4,
        hidden_depth=2,
    ):
        super(LSTMLightningModel, self).__init__()
        self.save_hyperparameters()  # Automatically logs and saves hyperparameters for reproducibility

        self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_depth, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

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
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
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
        loss = F.mse_loss(y_pred, y_target)

        # Prepare predictions and targets for IoU computation
        preds = y_pred.detach().cpu() 
        targets = y_target.detach().cpu()
        class_labels = torch.zeros(preds.size(0), dtype=torch.int) # Only one class here (fuel port)
        preds = [{"boxes": preds, "labels": class_labels}] # Predicted bounding boxe
        targets = [{"boxes": targets, "labels": class_labels}] # Ground truth bounding box 
        getattr(self, f"{stage}_iou").update(preds, targets) # Update IoU metric

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
