import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion
import torchvision.ops as ops

class FusionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.new_state = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        n = torch.tanh(self.new_state(torch.cat((x, r * h), dim=1)))
        h = (1 - z) * n + z * h
        return h

class SelfAttentionAggregation(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=2)
        return torch.bmm(attn_weights, v)

class FusionGRULightningModel(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        input_frames=10,
        output_frames=10,
        batch_size=32,
        bbox_dim=4,
        hidden_dim=256,
        output_dim=4,
        hidden_depth=3,
        num_heads=4,
        dropout=0.3,
    ):
        super(FusionGRULightningModel, self).__init__()
        self.save_hyperparameters()

        # Define GRU Encoders using ModuleList
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=bbox_dim, hidden_size=hidden_dim, num_layers=hidden_depth, batch_first=True, bidirectional=True, dropout=dropout),
            nn.GRU(input_size=bbox_dim, hidden_size=hidden_dim, num_layers=hidden_depth, batch_first=True, bidirectional=True, dropout=dropout),
        ])

        self.feature_fusion = nn.Linear(hidden_dim * 4, hidden_dim)

        self.fusion_gru = FusionGRU(hidden_dim, hidden_dim)

        self.attention = SelfAttentionAggregation(hidden_dim)

        self.intermediary_estimator = IntermediaryEstimator(hidden_dim, bbox_dim)

        self.decoder = nn.GRUCell(hidden_dim, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bbox_dim),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.train_iou = IntersectionOverUnion(box_format="cxcywh")
        self.val_iou = IntersectionOverUnion(box_format="cxcywh")
        self.test_iou = IntersectionOverUnion(box_format="cxcywh")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def forward(self, bboxes, delta_bboxes):
        batch_size, seq_len = bboxes.shape[:2]

        inputs = [
            delta_bboxes.float().view(batch_size, seq_len, -1),
            bboxes.float().view(batch_size, seq_len, -1),
        ]

        encoded_features = []
        for encoder, input_data in zip(self.encoders, inputs):
            output, _ = encoder(input_data)
            encoded_features.append(output)

        # Concatenate and fuse features
        fused_features = torch.cat(encoded_features, dim=-1)
        fused_features = self.feature_fusion(fused_features)

        # Apply Fusion-GRU
        h = torch.zeros(batch_size, self.hparams.hidden_dim, device=self.device)
        fusion_gru_outputs = []
        for t in range(seq_len):
            h = self.fusion_gru(fused_features[:, t], h)
            fusion_gru_outputs.append(h)
        fusion_gru_outputs = torch.stack(fusion_gru_outputs, dim=1)

        # Apply attention
        attended_features = self.attention(fusion_gru_outputs)

        # Intermediary estimation
        intermediary_bboxes = self.intermediary_estimator(attended_features)

        # Decode
        h = attended_features[:, -1]
        outputs = []
        for _ in range(self.hparams.output_frames):
            h = self.decoder(h)
            output = self.fc(h)
            output = torch.clamp(output, 0, 1)
            output = torch.nan_to_num(output, nan=0.0)
            outputs.append(output)

        return torch.stack(outputs, dim=1), intermediary_bboxes

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
            optimizer, mode="max", factor=0.5, patience=5
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
        (
            input_bboxes,
            input_delta_bboxes,
            output_bboxes,
            output_delta_bboxes
        ) = batch

        predicted_bboxes, intermediary_bboxes = self(
            input_bboxes, input_delta_bboxes
        )

        if output_bboxes.shape != predicted_bboxes.shape:
            output_bboxes = output_bboxes.unsqueeze(1).expand_as(predicted_bboxes).contiguous()

        final_loss = self.smooth_l1_loss(predicted_bboxes, output_bboxes)
        intermediary_loss = self.smooth_l1_loss(intermediary_bboxes, output_bboxes[:, :intermediary_bboxes.size(1)])
        
        loss = final_loss + 0.3 * intermediary_loss

        if torch.isnan(loss):
            loss = torch.tensor(1e6, device=self.device)

        preds = predicted_bboxes[:, -1].detach().cpu()
        targets = output_bboxes[:, -1].detach().cpu()
        class_labels = torch.zeros(preds.size(0), dtype=torch.int)
        preds = [{"boxes": preds, "labels": class_labels}]
        targets = [{"boxes": targets, "labels": class_labels}]
        getattr(self, f"{stage}_iou").update(preds, targets)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def smooth_l1_loss(self, pred, target, beta=1.0):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.mean()

    def _on_epoch_end(self, stage):
        iou = getattr(self, f"{stage}_iou").compute()
        iou_value = iou["iou"]
        self.log(f"{stage}_iou", iou_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        getattr(self, f"{stage}_iou").reset()