import lightning as L
from torch.utils.data import DataLoader

from YOLOv10LightningDataset import YOLOv10LightningDataset


class YOLOv10LightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset_images_folder_path,
        train_dataset_labels_folder_path,
        val_dataset_images_folder_path,
        val_dataset_labels_folder_path,
        test_dataset_images_folder_path,
        test_dataset_labels_folder_path,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset_images_folder_path = train_dataset_images_folder_path
        self.train_dataset_labels_folder_path = train_dataset_labels_folder_path
        self.val_dataset_images_folder_path = val_dataset_images_folder_path
        self.val_dataset_labels_folder_path = val_dataset_labels_folder_path
        self.test_dataset_images_folder_path = test_dataset_images_folder_path
        self.test_dataset_labels_folder_path = test_dataset_labels_folder_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None, predict_dataset_path=None):
        """
        Setup the data for the given stage

        Args:
            stage: Stage to setup the data for (fit, test, val)
            predict_dataset_path: Path to the prediction dataset
        """

        # Prepare the data for the training and validation process
        if stage == "train":
            self.train_dataset = YOLOv10LightningDataset(
                label_dir=self.train_dataset_labels_folder_path,
                image_dir=self.train_dataset_images_folder_path,
                phase="train",
            )
            self.val_dataset = YOLOv10LightningDataset(
                label_dir=self.val_dataset_labels_folder_path,
                image_dir=self.val_dataset_images_folder_path,
                phase="val",
            )
        # Prepare the data for the testing process
        elif stage == "test":
            self.test_dataset = YOLOv10LightningDataset(
                label_dir=self.test_dataset_labels_folder_path,
                image_dir=self.test_dataset_images_folder_path,
                phase="test",
            )
        # Prepare the data for the inference process
        elif stage == "predict" and predict_dataset_path is not None:
            self.predict_dataset = YOLOv10LightningDataset(
                image_dir=predict_dataset_path, phase="predict"
            )

    def train_dataloader(self):
        """
        Returns the training data loader

        Returns:
            DataLoader: Training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns the validation data loader

        Returns:
            DataLoader: Validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Returns the test data loader

        Returns:
            DataLoader: Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        """
        Returns the inference data loader

        Returns:
            DataLoader: Inference data loader
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
