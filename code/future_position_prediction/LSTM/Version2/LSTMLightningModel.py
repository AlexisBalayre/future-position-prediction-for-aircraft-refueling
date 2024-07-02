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
        feature_vector_shape=(3, 20, 20),
        bbox_dim=4,
        class_id_dim=1,
        hidden_dim=128,
        output_dim=4,
        hidden_depth=3,
        dropout=0.2,
        fc_multiplier=2,
    ):
        super(LSTMLightningModel, self).__init__()
        self.save_hyperparameters()  # Automatically logs and saves hyperparameters for reproducibility

        # Calculate input size for LSTM
        self.input_size = (feature_vector_shape[0] * feature_vector_shape[1] * feature_vector_shape[2]) + bbox_dim + class_id_dim

        # LSTM to process sequence of features and bboxes
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=hidden_depth,
            batch_first=True,
            dropout=dropout,
        )
        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),  # Output 4 values for bbox (x, y, w, h)
        )

        self.train_iou = IntersectionOverUnion(box_format="xywh")
        self.val_iou = IntersectionOverUnion(box_format="xywh")
        self.test_iou = IntersectionOverUnion(box_format="xywh")

    def forward(self, feature_vectors, bboxes, input_class_ids):
        batch_size, seq_len = feature_vectors.shape[:2]
        
        # Flatten the feature vectors
        flattened_features = feature_vectors.view(batch_size, seq_len, -1)

        # Ensure input_class_ids has the same dimensions
        input_class_ids = input_class_ids.unsqueeze(-1).repeat(1, 1, 1)
        
        # Concatenate flattened features with bboxes and class IDs
        lstm_input = torch.cat([flattened_features, bboxes, input_class_ids], dim=-1)

        # Process through LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Use only the last output from LSTM
        last_output = lstm_out[:, -1, :]
        
        # Final bbox prediction
        bbox_prediction = self.fc(last_output)
        
        return bbox_prediction

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
        (
            _,
            input_bboxes,
            input_class_ids,
            input_feature_vectors,
            output_bboxes,
        ) = batch

        # Predict bboxes for the last frame
        predicted_bboxes = self(input_feature_vectors, input_bboxes, input_class_ids)

        # Convert xywh to xyxy format for GIoU loss calculation
        predicted_bboxes_xyxy = ops.box_convert(
            predicted_bboxes, in_fmt="xywh", out_fmt="xyxy"
        )
        target_bboxes_xyxy = ops.box_convert(
            output_bboxes, in_fmt="xywh", out_fmt="xyxy"
        )

        # Compute GIoU loss
        giou_loss = ops.generalized_box_iou_loss(
            predicted_bboxes_xyxy, target_bboxes_xyxy, reduction="none"
        )

        # Transform the loss to be non-negative
        positive_loss = 1 - torch.exp(-giou_loss)

        # Take the mean of the transformed loss
        final_loss = positive_loss.mean()

        # Update IoU metric
        preds = predicted_bboxes.detach().cpu()
        targets = output_bboxes.detach().cpu()
        class_labels = torch.zeros(preds.size(0), dtype=torch.int)
        preds = [{"boxes": preds, "labels": class_labels}]
        targets = [{"boxes": targets, "labels": class_labels}]
        getattr(self, f"{stage}_iou").update(preds, targets)

        self.log(
            f"{stage}_loss",
            final_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return final_loss

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
