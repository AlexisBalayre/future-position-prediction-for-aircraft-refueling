import os
import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from LSTMLightningDataModule import LSTMLightningDataModule
from LSTMLightningModel import LSTMLightningModel
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def train_lstm(config, checkpoint_dir=None):
    # Paths to dataset files
    train_dataset_path = "/mnt/beegfs/home/s425500/Thesis/LSTM/data/train.json"
    val_dataset_path = "/mnt/beegfs/home/s425500/Thesis/LSTM/data/validation.json"
    test_dataset_path = "/mnt/beegfs/home/s425500/Thesis/LSTM/data/test.json"

    num_workers = 4  # Number of workers for data loading, adjust as needed

    # Fixed random seed for reproducibility of results
    torch.manual_seed(123)

    # Data Module
    data_module = LSTMLightningDataModule(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        test_dataset_path=test_dataset_path,
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        num_workers=num_workers,
    )

    # Initialize the model
    model = LSTMLightningModel(
        lr=config["learning_rate"],
        batch_size=config["batch_size"],
        input_dim=4,
        hidden_dim=config["hidden_dim"],
        output_dim=4,
        hidden_depth=config["hidden_depth"],
    )

    # Logger setup for TensorBoard
    logger = TensorBoardLogger("tb_logs", name="lstm_model")

    # Checkpoint and Report Callbacks
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="val_loss"),
        TuneReportCallback({"val_loss": "val_loss"}, on="validation_end"),
    ]

    # Trainer initialization with configurations for training process
    trainer = L.Trainer(
        max_epochs=50,  # You can adjust this
        accelerator="gpu",  # Specifies the training will be on GPU
        devices=1,  # Number of GPUs to use per trial
        logger=logger,  # Integrates the TensorBoard logger for tracking experiments
        callbacks=callbacks,  # Adds the specified callbacks to the training process
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
    )

    if checkpoint_dir:
        model = LSTMLightningModel.load_from_checkpoint(
            checkpoint_path=os.path.join(checkpoint_dir, "checkpoint")
        )

    # Training phase
    trainer.fit(model, datamodule=data_module)


def main():
    # Configuration for hyperparameter search
    config = {
        "batch_size": tune.choice([16, 32, 64]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "hidden_dim": tune.randint(16, 256),
        "hidden_depth": tune.randint(1, 4),
        "seq_length": tune.randint(10, 30),
    }

    # Define the search space and scheduler
    scheduler = ASHAScheduler(
        metric="val_loss", mode="min", max_t=50, grace_period=1, reduction_factor=2
    )

    # Reporter for Ray Tune
    reporter = CLIReporter(
        parameter_columns=[
            "batch_size",
            "learning_rate",
            "hidden_dim",
            "hidden_depth",
            "seq_length",
        ],
        metric_columns=["val_loss", "training_iteration"],
    )

    ray.init(address="auto")  # Initialize Ray in cluster mode

    analysis = tune.run(
        train_lstm,
        resources_per_trial={"cpu": 8, "gpu": 1},  # Each trial uses 8 CPUs and 1 GPU
        config=config,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="~/ray_results",  # Adjust this to your needs
        name="tune_lstm",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    main()
