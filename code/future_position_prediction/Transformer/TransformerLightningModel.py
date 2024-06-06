import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion


class TransformerLightningModel(L.LightningModule):
    """
    A PyTorch Lightning module for a Transformer-based model.

    Args:
        lr (float, optional): Learning rate. Defaults to 1e-4.
        batch_size (int, optional): Batch size. Defaults to 32.
        input_dim (int, optional): Input dimensionality. Defaults to 4.
        hidden_dim (int, optional): Hidden dimensionality. Defaults to 128.
        output_dim (int, optional): Output dimensionality. Defaults to 4.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        num_layers (int, optional): Number of transformer layers. Defaults to 2.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        lr=1e-4,
        batch_size=32,
        input_dim=4,
        hidden_dim=128,
        output_dim=4,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        super(TransformerLightningModel, self).__init__()
        self.save_hyperparameters()  # Automatically logs and saves hyperparameters for reproducibility

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
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
        x = self.embedding(x)
        tgt = torch.zeros_like(
            x
        )  # Initialize target sequence (for example, zero tensor)
        out = self.transformer(x, tgt)  # Pass through the transformer
        out = self.fc(out[:, -1, :])  # Apply the final linear layer
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
        class_labels = torch.zeros(
            preds.size(0), dtype=torch.int
        )  # Only one class here (fuel port)
        preds = [{"boxes": preds, "labels": class_labels}]  # Predicted bounding boxes
        targets = [
            {"boxes": targets, "labels": class_labels}
        ]  # Ground truth bounding boxes
        getattr(self, f"{stage}_iou").update(preds, targets)  # Update IoU metric

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
