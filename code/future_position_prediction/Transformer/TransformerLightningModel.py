import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchvision.ops as ops


class TransformerLightningModel(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        input_dim=4,
        hidden_dim=128,
        output_dim=4,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        batch_size=32,
        input_frames=1,  # Number of input frames
        output_frames=1,  # Number of output frames
    ):
        super(TransformerLightningModel, self).__init__()
        self.save_hyperparameters()

        # Embeddings for the encoder and decoder
        self.encoder_embedding = nn.Linear(input_dim, hidden_dim)
        self.decoder_embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Metrics
        self.train_iou = IntersectionOverUnion(box_format="xyxy")
        self.val_iou = IntersectionOverUnion(box_format="xyxy")
        self.test_iou = IntersectionOverUnion(box_format="xyxy")

    def forward(self, x, tgt):
        print("printtgt ", tgt[:, -1, :])

        # Embed the input and target tensors
        x = self.encoder_embedding(x)
        tgt = self.decoder_embedding(tgt)

        # Permute the tensors to the correct shape
        x = x.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        # Pass the tensors through the transformer
        out = self.transformer(x, tgt)
        # Permute the output tensor to the correct shape
        out = out.permute(1, 0, 2)


        out = self.fc_out(out[:, -1, :])

        print("printout ", out)

        return out

    def _shared_step(self, batch, batch_idx, stage):
        x, y_target = batch
        y_pred = self(x, y_target)

        loss = ops.generalized_box_iou_loss(
            y_pred, y_target[:, -1, :], reduction="mean"
        )

        preds = y_pred.detach().cpu()
        targets = y_target[:, -1, :].detach().cpu()

        class_labels = torch.zeros(preds.size(0), dtype=torch.int)
        preds = [{"boxes": p.unsqueeze(0), "labels": class_labels[i:i+1]} for i, p in enumerate(preds)]
        targets = [{"boxes": t.unsqueeze(0), "labels": class_labels[i:i+1]} for i, t in enumerate(targets)]

        getattr(self, f"{stage}_iou").update(preds, targets)

        self.log(
            f"{stage}_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

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

    def _on_epoch_end(self, stage):
        iou = getattr(self, f"{stage}_iou").compute()

        # Since IoU is returned as a dict with keys for each metric value type, ensure we log the actual value
        iou_value = iou["iou"].item() if isinstance(iou, dict) else iou.item()

        self.log(
            f"{stage}_iou",
            iou_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        getattr(self, f"{stage}_iou").reset()
