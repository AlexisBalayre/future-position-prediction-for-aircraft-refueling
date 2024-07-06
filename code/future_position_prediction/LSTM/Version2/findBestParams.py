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
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer


def train_lstm(config):
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

    # Loggers for TensorBoard
    logger = TensorBoardLogger(
        save_dir="logs/",
        name="lstm",
        version="0.1",
    )

    # Trainer initialization with configurations for training process
    trainer = L.Trainer(
        max_epochs=50,  # You can adjust this
        accelerator="cpu",  # Specifies the training will be on GPU
        devices="auto",  # Number of GPUs to use per trial
        logger=logger,
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
    )

    # Training phase
    trainer.fit(model, datamodule=data_module)


def main():
    # Configuration for hyperparameter search
    search_space = {
        "batch_size": tune.choice([16, 32, 64]),
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "hidden_dim": tune.randint(16, 256),
        "hidden_depth": tune.randint(1, 4),
        "seq_length": tune.randint(10, 30),
    }

    # Define the search space and scheduler
    scheduler = ASHAScheduler(max_t=50, grace_period=1, reduction_factor=2)
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
    trainable_with_resources = tune.with_resources(train_lstm, {"cpu": 16})
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        run_config=run_config,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=100,
            scheduler=scheduler,
        ),
    )
    analysis = tuner.fit()

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    main()
