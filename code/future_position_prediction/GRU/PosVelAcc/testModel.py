import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from .PosVelAccGRULightningDataModule import PosVelAccGRULightningDataModule
from .PosVelAccGRULightningModelConcat import PosVelAccGRULightningModelConcat

# Script to test the PosVelAcc-GRU model using trained weights.
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    L.seed_everything(42)

    # Model initialisation with specified architecture parameters
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/AARP/frames/full_dataset_annotated_fpp/test_filter_savgol.json"
    checkpoint_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/PosVelAcc/checkpoints/30input_60output/checkpoints/epoch=88-step=7565.ckpt"
    hparams_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/PosVelAcc/checkpoints/30input_60output/hparams.yaml"
    input_frames = 30  # Number of input frames
    output_frames = 60  # Number of output frames

    # Data Module initialisation
    data_module = PosVelAccGRULightningDataModule(
        train_dataset_path=test_dataset_path,
        val_dataset_path=test_dataset_path,
        test_dataset_path=test_dataset_path,
        input_frames=input_frames,
        output_frames=output_frames,
    )

    # Setup data module for testing
    data_module.setup("test")

    # Trainer initialisation with configurations for testing process
    trainer = L.Trainer(
        accelerator="cpu",  # Specifies the training will be on CPU
        devices="auto",  # Automatically selects the available devices
        deterministic=True,  # Ensures reproducibility of results
        precision=32,  # Use 32-bit floating point precision
        logger=CSVLogger("logs", name="concat"),
    )

    # Model initialisation
    model_concat = PosVelAccGRULightningModelConcat.load_from_checkpoint(
        checkpoint_path=checkpoint_path, hparams_file=hparams_path
    )

    # Test the model
    trainer.test(model_concat, datamodule=data_module, ckpt_path=checkpoint_path)
