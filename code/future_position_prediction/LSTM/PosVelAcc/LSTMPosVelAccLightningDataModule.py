import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from typing import Optional

from .LSTMPosVelAccLightningDataset import LSTMPosVelAccLightningDataset


class LSTMPosVelAccLightningDataModule(L.LightningDataModule):
    """
    A LightningDataModule for managing data loading for the PosVelAcc-LSTM model.
    This module handles the loading and preprocessing of datasets for training, validation, testing, and prediction.

    Args:
        train_dataset_path (str): Path to the training dataset JSON file.
        val_dataset_path (str): Path to the validation dataset JSON file.
        test_dataset_path (str): Path to the test dataset JSON file.
        input_frames (int): Number of frames to use as input.
        output_frames (int): Number of frames to predict as output.
        batch_size (int, optional): Batch size for data loading. Default is 16.
        num_workers (int, optional): Number of workers for data loading. Default is 4.
    """

    def __init__(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path: str,
        input_frames: int,
        output_frames: int,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_frames = input_frames
        self.output_frames = output_frames

    def setup(
        self, stage: Optional[str] = None, predict_dataset_path: Optional[str] = None
    ) -> None:
        """
        Setup the datasets for different stages (train, val, test, predict).

        Args:
            stage (str, optional): The stage for which to setup the dataset, either 'train', 'val', 'test', or 'predict'.
            predict_dataset_path (str, optional): Path to the dataset for prediction. Required if stage is 'predict'.
        """
        if stage == "train" or stage is None:
            self.train_dataset = LSTMPosVelAccLightningDataset(
                self.train_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="train",
            )
            self.val_dataset = LSTMPosVelAccLightningDataset(
                self.val_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="val",
            )
        if stage == "test" or stage is None:
            self.test_dataset = LSTMPosVelAccLightningDataset(
                self.test_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="test",
            )
        if stage == "predict" and predict_dataset_path is not None:
            self.predict_dataset = LSTMPosVelAccLightningDataset(
                predict_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="predict",
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader object for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader object for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader object for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the prediction dataset.

        Returns:
            DataLoader: DataLoader object for the prediction dataset.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
