import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from ultralytics import YOLOv10
from torchmetrics.detection import IntersectionOverUnion


class YOLOv10LightningModel(L.LightningModule):
    """
    A PyTorch Lightning module for a Transformer-based model.

    Args:
        lr (float, optional): Learning rate. Defaults to 1e-4.
        batch_size (int, optional): Batch size. Defaults to 32.
    """

    def __init__(self, lr=1e-4, batch_size=32):
        super(YOLOv10LightningModel, self).__init__()
        self.save_hyperparameters()  # Automatically logs and saves hyperparameters for reproducibility

        self.model = YOLOv10.from_pretrained("jameslahm/yolov10m")

        self.train_map = MeanAveragePrecision(box_format="xyxy")
        self.val_map = MeanAveragePrecision(box_format="xyxy")
        self.test_map = MeanAveragePrecision(box_format="xyxy")

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # YOLO models generally return a dictionary with predictions
        return self.model(x)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, bboxes, class_labels = batch
        loss = self.model.loss(images, bboxes, class_labels)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, bboxes, class_labels = batch
        preds = self.model.predict(images)

        for pred, bbox, label in zip(preds, bboxes, class_labels):
            pred_boxes = pred[:, :4]
            pred_labels = pred[:, 5].long()
            self.val_map.update([pred_boxes], [bbox], [pred_labels], [label])

        loss = self.model.loss(images, bboxes, class_labels)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_validation_epoch_end(self):
        map_score = self.val_map.compute()
        self.log("val_map", map_score, on_epoch=True, prog_bar=True, logger=True)
        self.val_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
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
