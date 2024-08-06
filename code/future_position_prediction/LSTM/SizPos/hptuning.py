import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

# Importing necessary modules from the custom TurbulenceModel package and utilities
from LSTMLightningDataModule import LSTMLightningDataModule
from LSTMLightningModelSum import LSTMLightningModelSum
from LSTMLightningModelAverage import LSTMLightningModelAverage
from LSTMLightningModelConcat import LSTMLightningModelConcat
from LSTMLightningModelClassic import LSTMLightningModelClassic

# Main execution block
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    L.seed_everything(42)

    # Model initialization with specified architecture parameters
    train_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/train_filter_savgol.json"
    val_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/val_filter_savgol.json"
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/test_filter_savgol.json"
    images_folder = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/images"
    num_workers = 8  # Number of workers for data loading
    batch_size = 16  # Number of samples per batch
    input_frames = [30]  # Number of input frames
    output_frames = [60]  # Number of output frames
    hidden_sizes = [128]  # Size of the model's hidden layers
    hidden_depths = [8]  # Number of hidden layers
    learning_rate = 1e-4  # Initial learning rate
    scheduler_patiences = [15]
    scheduler_factors = [0.5]
    max_epochs = 100  # Maximum number of training epochs
    dropout = 0.1  # Dropout rate
    
    # Create a dataframe to store the results
    results = pd.DataFrame(
        columns=[
            "model",
            "batch_size",
            "hidden_size",
            "hidden_depth",
            "learning_rate",
            "scheduler_patience",
            "scheduler_factor",
            "max_epochs",
            "dropout",
            "input_frames",
            "output_frames",
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
        input_frames=input_frames[0],
        output_frames=output_frames[0],
    )

    # Hyperparameters search space
    for in_frames in input_frames:
        for ou_frames in output_frames:
            for hidden_size in hidden_sizes:
                for hidden_depth in hidden_depths:
                    for scheduler_patience in scheduler_patiences:
                        for scheduler_factor in scheduler_factors:
                            # Setup the data module
                            data_module.setup("train")

                            # Trainer initialization with configurations for training process
                            trainer_classic_model = L.Trainer(
                                max_epochs=max_epochs,  # Maximum number of epochs for training
                                accelerator="cpu",  # Specifies the training will be on CPU
                                devices="auto",  # Automatically selects the available devices
                                deterministic=True,  # Ensures reproducibility of results
                                precision=32,  # Use 32-bit floating point precision
                                callbacks=[
                                    ModelCheckpoint(
                                        save_top_k=1,
                                        mode="max",
                                        monitor="val_Best_FIOU",
                                    ),
                                ],
                                logger=CSVLogger("logs", name="classic"),
                            )
                            """ trainer_sum_model = L.Trainer(
                                max_epochs=max_epochs,  # Maximum number of epochs for training
                                accelerator="cpu",  # Specifies the training will be on CPU
                                devices="auto",  # Automatically selects the available devices
                                deterministic=True,  # Ensures reproducibility of results
                                precision=32,  # Use 32-bit floating point precision
                                callbacks=[
                                    ModelCheckpoint(
                                        save_top_k=1,
                                        mode="max",
                                        monitor="val_Best_FIOU",
                                    ),
                                ],
                                logger=CSVLogger("logs", name="sum"),
                            )
                            trainer_average_model = L.Trainer(
                                max_epochs=max_epochs,  # Maximum number of epochs for training
                                accelerator="cpu",  # Specifies the training will be on CPU
                                devices="auto",  # Automatically selects the available devices
                                deterministic=True,  # Ensures reproducibility of results
                                precision=32,  # Use 32-bit floating point precision
                                callbacks=[
                                    ModelCheckpoint(
                                        save_top_k=1,
                                        mode="max",
                                        monitor="val_Best_FIOU",
                                    ),
                                ],
                                logger=CSVLogger("logs", name="average"),
                            ) """
                            trainer_concat_model = L.Trainer(
                                max_epochs=max_epochs,  # Maximum number of epochs for training
                                accelerator="cpu",  # Specifies the training will be on CPU
                                devices="auto",  # Automatically selects the available devices
                                deterministic=True,  # Ensures reproducibility of results
                                precision=32,  # Use 32-bit floating point precision
                                callbacks=[
                                    ModelCheckpoint(
                                        save_top_k=1,
                                        mode="max",
                                        monitor="val_Best_FIOU",
                                    ),
                                ],
                                logger=CSVLogger("logs", name="concat"),
                            )

                            # Model without combining hidden states
                            model_classic = LSTMLightningModelClassic(
                                lr=learning_rate,
                                input_frames=in_frames,
                                output_frames=ou_frames,
                                batch_size=batch_size,
                                hidden_size=hidden_size,
                                hidden_depth=hidden_depth,
                                dropout=dropout,
                                scheduler_factor=scheduler_factor,
                                scheduler_patience=scheduler_patience,
                            )

                            """ # Model (Hidden State Sum)
                            model_sum = LSTMLightningModelSum(
                                lr=learning_rate,
                                input_frames=in_frames,
                                output_frames=ou_frames,
                                batch_size=batch_size,
                                hidden_size=hidden_size,
                                hidden_depth=hidden_depth,
                                dropout=dropout,
                                scheduler_factor=scheduler_factor,
                                scheduler_patience=scheduler_patience,
                            )

                            # Model (Hidden State Average)
                            model_average = LSTMLightningModelAverage(
                                lr=learning_rate,
                                input_frames=in_frames,
                                output_frames=ou_frames,
                                batch_size=batch_size,
                                hidden_size=hidden_size,
                                hidden_depth=hidden_depth,
                                dropout=dropout,
                                scheduler_factor=scheduler_factor,
                                scheduler_patience=scheduler_patience,
                            ) """

                            # Model (Hidden State Concatenation)
                            model_concat = LSTMLightningModelConcat(
                                lr=learning_rate,
                                input_frames=in_frames,
                                output_frames=ou_frames,
                                batch_size=batch_size,
                                hidden_size=hidden_size,
                                hidden_depth=hidden_depth,
                                dropout=dropout,
                                scheduler_factor=scheduler_factor,
                                scheduler_patience=scheduler_patience,
                            )

                            # Training phase
                            trainer_classic_model.fit(
                                model_classic, datamodule=data_module
                            )
                            """ trainer_sum_model.fit(model_sum, datamodule=data_module)
                            trainer_average_model.fit(
                                model_average, datamodule=data_module
                            ) """
                            trainer_concat_model.fit(
                                model_concat, datamodule=data_module
                            )

                            # Setup data module for testing
                            data_module.setup("test")

                            for i in range(2):
                                # Select the model to evaluate
                                if i == 0:
                                    model = model_classic
                                    model_name = "classic"
                                    trainer = trainer_classic_model
                                elif i == 1:
                                    model = model_concat
                                    model_name = "concat"
                                    trainer = trainer_concat_model
                                """ elif i == 1:
                                    model = model_sum
                                    model_name = "sum"
                                    trainer = trainer_sum_model
                                elif i == 2:
                                    model = model_average
                                    model_name = "average"
                                    trainer = trainer_average_model """
                                

                                # Compute the metrics over the test dataset
                                test_metrics = trainer.test(
                                    model, datamodule=data_module, ckpt_path="best"
                                )[0]

                                # Create a dictionary with the hyperparameters and metrics
                                results_row = {
                                    "model": model_name,
                                    "batch_size": batch_size,
                                    "hidden_size": hidden_size,
                                    "hidden_depth": hidden_depth,
                                    "learning_rate": learning_rate,
                                    "scheduler_patience": scheduler_patience,
                                    "scheduler_factor": scheduler_factor,
                                    "max_epochs": max_epochs,
                                    "dropout": dropout,
                                    "input_frames": in_frames,
                                    "output_frames": ou_frames,
                                }

                                # Add test metrics to the results row
                                for key, value in test_metrics.items():
                                    results_row[f"{key}"] = value

                                # Append the results to the dataframe
                                results = pd.concat(
                                    [results, pd.DataFrame([results_row])],
                                    ignore_index=True,
                                )

                                # Save the results to a CSV file
                                results.to_csv(
                                    "results_final_LSTM_SizPos.csv", index=False
                                )
