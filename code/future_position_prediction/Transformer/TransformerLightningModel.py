import torch
import torch.nn.functional as F
from torch import nn, Tensor
import lightning as L
from typing import Tuple

from utils import (
    compute_ADE,
    compute_FDE,
    compute_AIOU,
    compute_FIOU,
    convert_velocity_to_positions,
    get_src_trg,
    generate_square_subsequent_mask,
)

from PositionalEncoder import PositionalEncoder


class TransformerLightningModel(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        input_frames: int = 10,
        output_frames: int = 10,
        batch_size: int = 32,
        bbox_dim: int = 4,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (640, 480),
    ):
        super(TransformerLightningModel, self).__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoder_input_layer = nn.Linear(
            in_features=bbox_dim, out_features=hidden_dim
        )
        self.pos_encoder_input = PositionalEncoder(
            dropout=dropout,
            max_seq_len=input_frames,
            d_model=hidden_dim,
            batch_first=True,
        )
        self.pos_encoder_output = PositionalEncoder(
            dropout=dropout,
            max_seq_len=output_frames,
            d_model=hidden_dim,
            batch_first=True,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_encoder_layers, norm=None
        )

        # Decoder
        self.decoder_input_layer = nn.Linear(
            in_features=bbox_dim, out_features=hidden_dim
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_decoder_layers, norm=None
        )
        self.linear_mapping = nn.Linear(in_features=hidden_dim, out_features=bbox_dim)
        self.Sigmoid = nn.Sigmoid()

        # Layer Normalization for output
        self.output_norm = nn.LayerNorm(bbox_dim)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor = None,
        src_mask: Tensor = None,
        tgt_mask: Tensor = None,
    ) -> Tensor:
        return self._forward_inference(src)

    def _net(
        self,
        src: Tensor,
        tgt: Tensor = None,
        src_mask: Tensor = None,
        tgt_mask: Tensor = None,
    ) -> Tensor:
        src = self.encoder_input_layer(src)
        src = self.pos_encoder_input(src)
        memory = self.transformer_encoder(src)

        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.transformer_decoder(
            tgt=decoder_output, memory=memory, tgt_mask=tgt_mask, memory_mask=src_mask
        )
        decoder_output = self.linear_mapping(decoder_output)
        decoder_output = self.Sigmoid(decoder_output)
        return decoder_output

    def _forward_inference(self, src: Tensor) -> Tensor:
        device = src.device

        # Encode the source sequence
        src_encoded = self.encoder_input_layer(src)
        src_encoded = self.pos_encoder_input(src_encoded)
        memory = self.transformer_encoder(src_encoded)

        # Initialize the target sequence with the last input frame
        tgt = src[:, -1:, :]

        # Generate output frames one by one
        for _ in range(self.hparams.output_frames - 1):
            dim_a = tgt.shape[1]
            dim_b = src.shape[1]

            # Generate masks
            tgt_mask = generate_square_subsequent_mask(dim_a, dim_a).to(device)
            src_mask = generate_square_subsequent_mask(dim_a, dim_b).to(device)

            # Generate the next frame prediction
            decoder_output = self.decoder_input_layer(tgt)
            decoder_output = self.transformer_decoder(
                tgt=decoder_output,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=src_mask,
            )
            decoder_output = self.linear_mapping(decoder_output)
            decoder_output = self.Sigmoid(decoder_output)

            # Concatenate the prediction to the target sequence
            tgt = torch.cat([tgt, decoder_output[:, -1:, :]], dim=1)

        dim_a = tgt.shape[1]
        dim_b = src.shape[1]

        tgt_mask = generate_square_subsequent_mask(dim_a, dim_a).to(device)
        src_mask = generate_square_subsequent_mask(dim_a, dim_b).to(device)

        # Generate the last frame prediction
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.transformer_decoder(
            tgt=decoder_output,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )
        decoder_output = self.linear_mapping(decoder_output)
        decoder_output = self.Sigmoid(decoder_output)
        return decoder_output

    def _shared_step(self, batch, batch_idx, stage):
        (
            _,
            input_bboxes_position,
            input_bboxes_vel,
            _,
            target_bboxes_position,
            target_bboxes_vel,
        ) = batch

        device = input_bboxes_position.device

        # Prepare the source and target sequences for bounding boxes and velocities
        src_bboxes, trg_bboxes = get_src_trg(
            sequence=torch.cat(
                (input_bboxes_position, target_bboxes_position), dim=1
            ).to(device),
            enc_seq_len=self.hparams.input_frames,
            target_seq_len=self.hparams.output_frames,
        )

        src = src_bboxes.to(device)
        tgt = trg_bboxes.to(device)

        tgt_mask = generate_square_subsequent_mask(
            dim1=self.hparams.output_frames, dim2=self.hparams.output_frames
        ).to(device)

        src_mask = generate_square_subsequent_mask(
            dim1=self.hparams.output_frames, dim2=self.hparams.input_frames
        ).to(device)

        if stage == "train":
            output = self(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        else:
            output = self._forward_inference(src)

        # Compute losses
        prediction_loss = F.smooth_l1_loss(output, target_bboxes_position)

        # Log losses
        self.log(
            f"{stage}_loss",
            prediction_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # Compute metrics
        aiou = compute_AIOU(output, target_bboxes_position)
        fiou = compute_FIOU(output, target_bboxes_position)

        self.log(
            f"{stage}_AIoU",
            aiou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            f"{stage}_FIoU",
            fiou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return prediction_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=15, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
