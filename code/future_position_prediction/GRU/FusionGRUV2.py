import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torchmetrics.detection import IntersectionOverUnion



class IntermediateBBoxPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=16, activation=torch.relu, dropout=[0, 0]):
        super(IntermediateBBoxPredictor, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.dense1 = nn.Linear(input_dim, 256)
        self.dense2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout[0])
        x = self.activation(self.dense1(x))
        x = F.dropout(x, self.dropout[1])
        x = self.dense2(x)
        return x


class TemporalAttentionAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalAttentionAggregator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence):
        # Projection des caractéristiques
        projected_features = self.feature_projection(sequence)
        # Calcul des scores d'attention
        attention_scores = F.softmax(self.attention_score(projected_features), dim=1)
        # Application de l'attention et agrégation
        aggregated_features = torch.sum(attention_scores * projected_features, dim=1)
        return aggregated_features


class FusionGRUModel(L.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        input_seq_length=10,
        output_seq_length=10,
        batch_size=32,
        bbox_dim=4,
        hidden_dim=256,
        output_dim=4,
        future_steps=9,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Projections des caractéristiques d'entrée
        self.appearance_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU()
        )
        self.bbox_projector = nn.Sequential(nn.Linear(bbox_dim, hidden_dim), nn.ReLU())
        self.velocity_projector = nn.Sequential(
            nn.Linear(bbox_dim, hidden_dim), nn.ReLU()
        )

        # Réseaux GRU principaux
        self.fusion_gru = GRUNet(
            hidden_dim * 2, hidden_dim, output_dim, 1, 32, future_steps
        )
        self.velocity_gru = nn.GRU(hidden_dim, 32, 1, batch_first=True)

        # Module d'agrégation par attention temporelle
        self.temporal_aggregator = TemporalAttentionAggregator(4, hidden_dim)

        # Décodeur GRU final
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Métriques d'évaluation
        self.train_iou = IntersectionOverUnion()
        self.val_iou = IntersectionOverUnion()
        self.test_iou = IntersectionOverUnion()

    def forward(self, appearance_features, bboxes, velocities):
        batch_size, seq_length = appearance_features.shape[:2]

        # Projection des caractéristiques d'entrée
        appearance_features = self.appearance_projector(appearance_features)
        bbox_features = self.bbox_projector(bboxes)
        velocity_features = self.velocity_projector(velocities)

        # Initialisation des états cachés
        fusion_hidden_state = torch.zeros(
            batch_size, self.hparams.hidden_dim, device=self.device
        )
        velocity_hidden_state = torch.zeros(1, batch_size, 32, device=self.device)

        final_outputs = []
        intermediate_outputs = []

        for t in range(seq_length):
            # Traitement des caractéristiques de vitesse
            _, velocity_hidden_state = self.velocity_gru(
                velocity_features[:, t].unsqueeze(1), velocity_hidden_state
            )

            # Fusion des caractéristiques d'apparence et de boîte englobante
            fused_features = torch.cat(
                [appearance_features[:, t], bbox_features[:, t]], dim=-1
            )

            # Passage à travers le réseau FusionGRU
            fusion_output, fusion_hidden_state, intermediate_bboxes = self.fusion_gru(
                fused_features,
                fusion_hidden_state,
                bbox_features[:, t],
                velocity_hidden_state.squeeze(0),
            )

            intermediate_outputs.append(intermediate_bboxes)

            # Reshape et agrégation temporelle
            intermediate_bboxes_reshaped = intermediate_bboxes.view(
                batch_size, self.hparams.future_steps, 4
            )
            aggregated_features = self.temporal_aggregator(intermediate_bboxes_reshaped)

            # Décodage final
            decoded_features, fusion_hidden_state = self.decoder_gru(
                aggregated_features.unsqueeze(1), fusion_hidden_state.unsqueeze(0)
            )
            final_output = self.output_layer(decoded_features.squeeze(1))
            final_outputs.append(final_output)

        final_outputs = torch.stack(final_outputs, dim=1)
        intermediate_outputs = torch.stack(intermediate_outputs, dim=1)

        return final_outputs, intermediate_outputs

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def _shared_step(self, batch, batch_idx, stage):
        bboxes, velocities, appearance_features, target_bboxes = batch

        predicted_bboxes, intermediate_bboxes = self(
            appearance_features, bboxes, velocities
        )

        # Calcul des pertes
        final_loss = F.smooth_l1_loss(predicted_bboxes, target_bboxes)
        intermediate_loss = F.smooth_l1_loss(
            intermediate_bboxes, target_bboxes.repeat(1, 1, self.hparams.future_steps)
        )

        total_loss = final_loss + 0.3 * intermediate_loss

        # Enregistrement des métriques
        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        getattr(self, f"{stage}_iou")(predicted_bboxes[:, -1], target_bboxes[:, -1])

        return total_loss

    def on_validation_epoch_end(self):
        val_iou = self.val_iou.compute()
        self.log("val_iou", val_iou, prog_bar=True)
        self.val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
