import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

from .LSTMPosVelAccLightningDataModule import LSTMPosVelAccLightningDataModule
from .LSTMPosVelAccLightningModelSum import LSTMPosVelAccLightningModelSum

# Script to test the PosVelAcc-LSTM model using trained weights.
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    L.seed_everything(42)

    # Model initialisation with specified architecture parameters
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/AARP/frames/full_dataset_annotated_fpp/test_filter_savgol.json"
    hparams_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/LSTM/PosVelAcc/checkpoints/15input_30output/hparams.yaml"
    checkpoint_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/LSTM/PosVelAcc/checkpoints/15input_30output/checkpoints/epoch=90-step=8554.ckpt"
    input_frames = 15  # Number of input frames
    output_frames = 30  # Number of output frames

    # Data Module
    data_module = LSTMPosVelAccLightningDataModule(
        train_dataset_path=test_dataset_path,
        val_dataset_path=test_dataset_path,
        test_dataset_path=test_dataset_path,
        input_frames=input_frames,
        output_frames=output_frames,
    )

    # Setup the data module
    data_module.setup("test")

    # Trainer initialisation with configurations for training process
    trainer = L.Trainer(
        accelerator="cpu",  # Specifies the training will be on CPU
        devices="auto",  # Automatically selects the available devices
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
        logger=CSVLogger("logs", name="concat"),
    )

    # Model (Hidden State Concatenation)
    model_concat = LSTMPosVelAccLightningModelSum.load_from_checkpoint(
        hparams_file=hparams_path, checkpoint_path=checkpoint_path
    )

    # Test phase
    trainer.test(model_concat, datamodule=data_module, ckpt_path=checkpoint_path)
