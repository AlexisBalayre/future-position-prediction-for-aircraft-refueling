import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from TransformerLightningDataModule import TransformerLightningDataModule
from TransformerLightningModel import TransformerLightningModel

# Main execution block
if __name__ == "__main__":
    # Model initialization with specified architecture parameters
    train_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/train.json"
    val_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/validation.json"
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/test.json"
    num_workers = 8  # Number of workers for data loading
    batch_size = 16  # Number of samples per batch
    input_frames = 20  # Number of input frames
    output_frames = 1  # Number of output frames
    input_dim = 4  # Dimensionalityx of input features
    output_dim = 4  # Dimensionality of the model's output
    hidden_dim = 80  # Size of the model's hidden layers
    hidden_depth = 1  # Number of hidden layers
    learning_rate = 9e-4  # Initial learning rate
    max_epochs = 100000  # Maximum number of training epochs

    # Fixed random seed for reproducibility of results
    torch.manual_seed(123)

    # Data Module
    data_module = TransformerLightningDataModule(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        test_dataset_path=test_dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data_module.setup(
        input_frames, output_frames, stage="train"
    )  # Prepare data for training

    # Initialize the model
    model = TransformerLightningModel(
        lr=learning_rate,
        batch_size=batch_size,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        input_frames=input_frames,
        output_frames=output_frames,
    )

    # Logger setup for TensorBoard
    logger = TensorBoardLogger("tb_logs", name="transformer")

    # Early stopping and checkpointing callbacks
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="val_loss"),
    ]

    # Trainer initialization with configurations for training process
    trainer = L.Trainer(
        max_epochs=max_epochs,  # Maximum number of epochs for training
        accelerator="cpu",  # Specifies the training will be on CPU
        devices="auto",  # Automatically selects the available devices
        logger=logger,  # Integrates the TensorBoard logger for tracking experiments
        callbacks=callbacks,  # Adds the specified callbacks to the training process
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
    )

    # Training phase
    trainer.fit(
        model,
        datamodule=data_module,
    )  # Start training the model

    # Testing phase
    data_module.setup(
        input_frames, output_frames, stage="test"
    )  # Prepare data for testing
    trainer.test(model, datamodule=data_module)  # Evaluate the model on the test set
