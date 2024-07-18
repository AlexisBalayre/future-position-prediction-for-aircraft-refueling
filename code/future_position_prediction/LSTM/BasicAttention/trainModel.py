import torch

from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from LSTMLightningDataModule import LSTMLightningDataModule
from LSTMLightningModel import LSTMLightningModel

# Main execution block
if __name__ == "__main__":
    # Model initialization with specified architecture parameters
    train_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/train.json"
    val_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/val.json"
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/test.json"
    images_folder = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/images"
    num_workers = 8  # Number of workers for data loading
    batch_size = 16  # Number of samples per batch
    input_frames = 60 # Number of input frames
    output_frames = 15  # Number of output frames
    hidden_dim = 256  # Size of the model's hidden layers
    hidden_depth = 1  # Number of hidden layers
    learning_rate = 1e-3  # Initial learning rate
    max_epochs = 1000  # Maximum number of training epochs
    dropout = 0.1  # Dropout rate

    # Fixed random seed for reproducibility of results
    L.seed_everything(123)

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

    # Setup the data module
    data_module.setup(
        stage="train", 
    )

    # Initialize the model
    model = LSTMLightningModel(
        lr=learning_rate,
        input_frames=input_frames,
        output_frames=output_frames,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        dropout=dropout,
    )

    # Logger setup for TensorBoard
    logger = TensorBoardLogger("tb_logs", name="lstm_fpp_version3")

    # Early stopping and checkpointing callbacks
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="max", monitor="val_FIoU"),
    ]

    # Trainer initialization with configurations for training process
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
    )

    # Training phase
    trainer.fit(model, datamodule=data_module)

    # Setup the data module for testing
    data_module.setup(
        stage="test"
    )

    # Testing phase
    trainer.test(model, datamodule=data_module)
