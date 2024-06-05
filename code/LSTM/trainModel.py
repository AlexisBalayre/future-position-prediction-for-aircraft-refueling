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
    train_dataset_path = "./data/outdoor2/train.json"
    val_dataset_path = "./data/outdoor2/validation.json"
    test_dataset_path = "./data/outdoor2/test.json"
    num_workers = 8  # Number of workers for data loading
    batch_size = 16  # Number of samples per batch
    seq_length = 15  # Length of the input sequence
    input_dim = 4  # Dimensionalityx of input features
    output_dim = 4  # Dimensionality of the model's output
    hidden_dim = 256  # Size of the model's hidden layers
    hidden_depth = 2  # Number of hidden layers
    learning_rate = 5e-4  # Initial learning rate
    max_epochs = 100000  # Maximum number of training epochs

    # Fixed random seed for reproducibility of results
    torch.manual_seed(123)

    # Data Module
    data_module = LSTMLightningDataModule(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        test_dataset_path=test_dataset_path,
        seq_length=seq_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    data_module.setup("fit")  # Prepare data for the fitting process

    # Initialize the model
    model = LSTMLightningModel(
        lr=learning_rate,
        batch_size=batch_size,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        hidden_depth=hidden_depth,
    )

    # Logger setup for TensorBoard
    logger = TensorBoardLogger("tb_logs", name="lstm_model_indoor1")

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
    data_module.setup("test")  # Prepare data for testing
    trainer.test(model, datamodule=data_module)  # Evaluate the model on the test set
