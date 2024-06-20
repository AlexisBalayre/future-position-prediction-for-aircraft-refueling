import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchvision.ops as ops

class CubicLSTMLightningModel(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        input_frames=10,
        output_frames=10,
        batch_size=32,
        input_dim=512 + 4,  # 512 for ResNet features + 4 for bbox coordinates
        hidden_dim=128,
        output_dim=4,
        hidden_depth=3,
        dropout=0.2,
    ):
        super(CubicLSTMLightningModel, self).__init__()
        self.save_hyperparameters()  # Automatically logs and saves hyperparameters for reproducibility

        self.hidden_dim = hidden_dim
        self.cubic_lstm = CubicLSTMCell(input_dim, hidden_dim, kernel_size=3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.feature_extractor = FeatureExtractor()

        self.train_iou = IntersectionOverUnion(box_format="xyxy")
        self.val_iou = IntersectionOverUnion(box_format="xyxy")
        self.test_iou = IntersectionOverUnion(box_format="xyxy")

    def forward(self, images, bboxes):
        batch_size, seq_len, _, _, _ = images.size()
        
        features = []
        for t in range(seq_len):
            feature_batch = [self.feature_extractor.extract(image) for image in images[:, t, :, :, :]]
            feature_batch = torch.stack(feature_batch)
            features.append(feature_batch)
        
        features = torch.stack(features, dim=1)  # Shape: (batch_size, seq_len, feature_dim)
        
        combined_input = torch.cat((features, bboxes), dim=-1)  # Concatenate along the last dimension
        
        h_t = torch.zeros(batch_size, self.hidden_dim, images.size(3), images.size(4), device=images.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, images.size(3), images.size(4), device=images.device)
        h_s_t = torch.zeros(batch_size, self.hidden_dim, images.size(3), images.size(4), device=images.device)
        c_s_t = torch.zeros(batch_size, self.hidden_dim, images.size(3), images.size(4), device=images.device)
        
        for t in range(seq_len):
            x_t = combined_input[:, t, :, :, :]
            h_t, c_t, h_s_t, c_s_t = self.cubic_lstm(x_t, h_t, c_t, h_s_t, c_s_t)
        
        h_t = self.dropout(h_t)
        out = self.fc(h_t.view(batch_size, -1))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
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
        images, bboxes, y_target = batch
        y_pred = self(images, bboxes)

        loss = ops.generalized_box_iou_loss(y_pred, y_target, reduction='mean')

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