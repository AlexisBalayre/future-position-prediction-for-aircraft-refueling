import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

# Importing necessary modules from the custom TurbulenceModel package and utilities
from LSTMLightningDataModule import LSTMLightningDataModule
from LSTMLightningModel import LSTMLightningModel

# Main execution block
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    L.seed_everything(42)

    # Model initialization with specified architecture parameters
    train_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/train.json"
    val_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/val.json"
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/test.json"
    images_folder = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/images"
    num_workers = 8  # Number of workers for data loading
    batch_size = 16  # Number of samples per batch
    input_frames = 60 # Number of input frames
    output_frames = 30  # Number of output frames
    hidden_dims = [32, 64, 128, 256, 512]  # Size of the model's hidden layers
    hidden_depths = [1, 2, 4, 6]  # Number of hidden layers
    learning_rate = 1e-3  # Initial learning rate
    scheduler_patiences = [10, 20]
    scheduler_factors = [0.9, 0.5, 0.1]
    max_epochs = 100  # Maximum number of training epochs
    dropout = 0.1  # Dropout rate
    nb_heads = [2, 4, 8]  # Number of attention heads

    # Create a dataframe to store the results
    results = pd.DataFrame(
        columns=[
            "batch_size",
            "hidden_dim",
            "hidden_depth",
            "learning_rate",
            "scheduler_patience",
            "scheduler_factor",
            "max_epochs",
            "dropout",
            "input_frames",
            "output_frames",
            "nb_heads",
            "total_loss",
            "bbox_loss",
            "velocity_loss", 
            "velocities_to_positions_loss",
            "ADE_from_vel",
            "FDE_from_vel",
            "AIoU_from_vel",
            "FIoU_from_vel",
            "ADE",
            "FDE",
            "AIoU",
            "FIoU"
        ]
    )

    # Data Module
    data_module = LSTMLightningDataModule(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        test_dataset_path=test_dataset_path,
        images_folder=images_folder,
        batch_size=batch_size,
        num_workers=num_workers,
        input_frames=input_frames, 
        output_frames=output_frames
    )

    # Hyperparameters search space
    for hidden_dim in hidden_dims:
        for hidden_depth in hidden_depths:
            for nb_head in nb_heads:
                for scheduler_patience in scheduler_patiences:
                    for scheduler_factor in scheduler_factors:
                            # Setup the data module
                            data_module.setup("train")

                            # Model initialization with specified architecture parameters
                            model = LSTMLightningModel(
                                lr=learning_rate,
                                input_frames=input_frames,
                                output_frames=output_frames,
                                batch_size=batch_size,
                                hidden_dim=hidden_dim,
                                hidden_depth=hidden_depth,
                                dropout=dropout,
                                scheduler_factor=scheduler_factor,
                                scheduler_patience=scheduler_patience,
                                num_heads=nb_head
                            )

                            # Early stopping and checkpointing callbacks
                            callbacks = [
                                ModelCheckpoint(save_top_k=1, mode="max", monitor="val_FIoU"),
                            ]

                            # Trainer initialization with configurations for training process
                            trainer = L.Trainer(
                                max_epochs=max_epochs,  # Maximum number of epochs for training
                                accelerator="auto",  # Specifies the training will be on CPU
                                devices="auto",  # Automatically selects the available devices
                                deterministic=True,  # Ensures reproducibility of results
                                precision=32,  # Use 32-bit floating point precision
                                callbacks=callbacks,
                            )

                            # Training phase
                            trainer.fit(model, datamodule=data_module)

                            # Compute the MSE over the test dataset
                            metrics = trainer.test(
                                model, datamodule=data_module
                            )  # Test the model

                            # Convert the metrics to a pandas DataFrame
                            metrics = pd.DataFrame(metrics)

                            # Concatenate the version with the metrics
                            results = pd.concat(
                                [
                                    results,
                                    pd.concat(
                                        [
                                            pd.DataFrame(
                                                {
                                                    "batch_size": batch_size,
                                                    "hidden_dim": hidden_dim,
                                                    "hidden_depth": hidden_depth,
                                                    "learning_rate": learning_rate,
                                                    "scheduler_patience": scheduler_patience,
                                                    "scheduler_factor": scheduler_factor,
                                                    "max_epochs": max_epochs,
                                                    "dropout": dropout,
                                                    "input_frames": input_frames,
                                                    "output_frames": output_frames,
                                                    "nb_heads": nb_head
                                                }
                                            ),
                                            metrics,
                                        ],
                                        axis=1,
                                    ),
                                ]
                            )

                            # Save the results to a CSV file
                            results.to_csv("lstm_multihead_attention_hp_tuning.csv", index=False)


