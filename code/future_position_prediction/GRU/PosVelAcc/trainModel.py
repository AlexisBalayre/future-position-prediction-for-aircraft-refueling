import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

from .PosVelAccGRULightningDataModule import PosVelAccGRULightningDataModule
from .PosVelAccGRULightningModelConcat import PosVelAccGRULightningModelConcat

# Script to train the PosVelAcc-GRU model.
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    L.seed_everything(42)

    # Model initialisation with specified architecture parameters
    train_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/AARP/frames/full_dataset_annotated_fpp/train_filter_savgol.json"
    val_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/AARP/frames/full_dataset_annotated_fpp/val_filter_savgol.json"
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/AARP/frames/full_dataset_annotated_fpp/test_filter_savgol.json"
    num_workers = 8  # Number of workers for data loading
    batch_size = 16  # Number of samples per batch
    input_frames = 30  # Number of input frames
    output_frames = 60  # Number of output frames
    hidden_size = 128  # Size of the model's hidden layers
    hidden_depth = 8  # Number of hidden layers
    learning_rate = 5e-4  # Initial learning rate
    scheduler_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
    scheduler_factor = 0.9  # Factor by which the learning rate will be reduced
    max_epochs = 100  # Maximum number of training epochs
    dropout = 0.2  # Dropout rate

    # Data Module initialisation
    data_module = PosVelAccGRULightningDataModule(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        test_dataset_path=test_dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        input_frames=input_frames,
        output_frames=output_frames,
    )

    # Setup the data module
    data_module.setup("train")

    # Trainer initialisation with configurations for training process
    trainer = L.Trainer(
        max_epochs=max_epochs,  # Maximum number of epochs for training
        accelerator="cpu",  # Specifies the training will be on CPU
        devices="auto",  # Automatically selects the available devices
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
        callbacks=[
            ModelCheckpoint(
                save_top_k=1,
                mode="min",
                monitor="val_Best_FDE",
            ),
        ],
        logger=CSVLogger("logs", name="concat"),
    )

    # Model (Hidden State Concatenation) initialisation
    model_concat = PosVelAccGRULightningModelConcat(
        lr=learning_rate,
        input_frames=input_frames,
        output_frames=output_frames,
        batch_size=batch_size,
        hidden_dim=hidden_size,
        encoder_layer_nb=hidden_depth,
        decoder_layer_nb=hidden_depth,
        dropout=dropout,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
    )

    # Training phase
    trainer.fit(model_concat, datamodule=data_module)

    # Setup data module for testing
    data_module.setup("test")

    # Testing phase
    trainer.test(model_concat, datamodule=data_module, ckpt_path="best")
