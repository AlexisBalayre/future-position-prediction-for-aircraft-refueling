import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from typing import Optional

from .SizPosGRULightningDataset import SizPosGRULightningDataset


class SizPosGRULightningDataModule(L.LightningDataModule):
    """
    A LightningDataModule for managing data loading for the SizPos-GRU model.
    This module handles the loading and preprocessing of datasets for training, validation, testing, and prediction.

    Args:
        train_dataset_path (str): Path to the training dataset.
        val_dataset_path (str): Path to the validation dataset.
        test_dataset_path (str): Path to the testing dataset.
        input_frames (int): Number of input frames.
        output_frames (int): Number of output frames.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 16.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
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
    ) -> None:
        super().__init__()
        # Store initialisation parameters
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
        Sets up the datasets for the specified stage (train, val, test, or predict).

        Args:
            stage (Optional[str], optional): Stage name, can be 'train', 'val', 'test', or 'predict'. Defaults to None.
            predict_dataset_path (Optional[str], optional): Path to the dataset used for prediction. Only used if stage is 'predict'. Defaults to None.
        """
        if stage == "train" or stage is None:
            # Initialize training and validation datasets
            self.train_dataset = SizPosGRULightningDataset(
                self.train_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="train",
            )
            self.val_dataset = SizPosGRULightningDataset(
                self.val_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="val",
            )
        if stage == "test" or stage is None:
            # Initialize test dataset
            self.test_dataset = SizPosGRULightningDataset(
                self.test_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="test",
            )
        if stage == "predict" and predict_dataset_path is not None:
            # Initialize prediction dataset
            self.predict_dataset = SizPosGRULightningDataset(
                predict_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                stage="predict",
            )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
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
            DataLoader: DataLoader for the validation dataset.
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
            DataLoader: DataLoader for the test dataset.
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
            DataLoader: DataLoader for the prediction dataset.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
