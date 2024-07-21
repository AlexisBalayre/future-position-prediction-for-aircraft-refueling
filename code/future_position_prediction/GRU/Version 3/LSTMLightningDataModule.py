import lightning as L
from torch.utils.data import DataLoader

from LSTMLightningDataset import LSTMLightningDataset


class LSTMLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_folder,
        input_frames,
        output_frames,
        target_size=(480, 640),
        batch_size=16,
        num_workers=4,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = LSTMLightningDataset(
                split="train",
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                data_folder=self.data_folder,
                target_size=self.target_size,
            )
            self.val_dataset = LSTMLightningDataset(
                split="val",
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                data_folder=self.data_folder,
                target_size=self.target_size,
            )
        if stage == "test" or stage is None:
            self.test_dataset = LSTMLightningDataset(
                split="test",
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                data_folder=self.data_folder,
                target_size=self.target_size,
            )
        if stage == "predict" and self.predict_dataset_path is not None:
            self.predict_dataset = LSTMLightningDataset(
                split="predict",
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                data_folder=self.data_folder,
                target_size=self.target_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
