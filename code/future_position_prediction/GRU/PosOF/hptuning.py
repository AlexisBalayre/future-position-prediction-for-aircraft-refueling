import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

# Importing necessary modules from the custom TurbulenceModel package and utilities
from LSTMLightningDataModule import LSTMLightningDataModule
from GRULightningModelWithOpticalFlow import GRULightningModelWithOpticalFlow


# Main execution block
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    L.seed_everything(42)

    # Model initialization with specified architecture parameters
    data_folder = "/Volumes/ALEXIS/Thesis/OF2"
    num_workers = 8  # Number of workers for data loading
    batch_size = 16  # Number of samples per batch
    input_frames = [15]  # Number of input frames
    output_frames = [30]  # Number of output frames
    hidden_sizes = [128, 256]  # Size of the model's hidden layers
    hidden_depths = [1, 2, 3]  # Number of hidden layers
    learning_rate = 1e-4  # Initial learning rate
    scheduler_patiences = [5]
    scheduler_factors = [0.1]
    max_epochs = 60  # Maximum number of training epochs
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

    torch.set_float32_matmul_precision('medium')

    # Data Module
    data_module = LSTMLightningDataModule(
        data_folder=data_folder,
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

                            trainer_sum_model = L.Trainer(
                                max_epochs=max_epochs,  # Maximum number of epochs for training
                                accelerator="gpu",  # Specifies the training will be on CPU
                                devices=1,  # Automatically selects the available devices
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

                            # Model (Hidden State Sum)
                            model_sum = GRULightningModelWithOpticalFlow(
                                lr=learning_rate,
                                input_frames=in_frames,
                                output_frames=ou_frames,
                                batch_size=batch_size,
                                hidden_dim=hidden_size,
                                hidden_depth=hidden_depth,
                                dropout=dropout,
                                scheduler_factor=scheduler_factor,
                                scheduler_patience=scheduler_patience,
                            )
    
                            trainer_sum_model.fit(model_sum, datamodule=data_module)
                        
                            # Setup data module for testing
                            data_module.setup("test")

                            # Compute the metrics over the test dataset
                            test_metrics = trainer_sum_model.test(
                                model_sum, datamodule=data_module, ckpt_path="best"
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
                                "results.csv", index=False
                            )
