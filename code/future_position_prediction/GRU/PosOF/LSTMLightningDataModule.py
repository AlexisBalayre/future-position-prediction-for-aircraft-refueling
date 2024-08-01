import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from LSTMLightningDataset import LSTMLightningDataset


class LSTMLightningDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_folder,
        input_frames,
        output_frames,
        batch_size=16,
        num_workers=4,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_frames = input_frames
        self.output_frames = output_frames

    def setup(self, stage, predict_dataset_path=None):
        if stage == "train" or stage is None:
            self.train_dataset = LSTMLightningDataset(
                data_folder=self.data_folder,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                split="train",
            )
            self.val_dataset = LSTMLightningDataset(
                data_folder=self.data_folder,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                split="val",
            )
        if stage == "test" or stage is None:
            self.test_dataset = LSTMLightningDataset(
                data_folder=self.data_folder,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                split="test",
            )
        if stage == "predict" and predict_dataset_path is not None:
            self.predict_dataset = LSTMLightningDataset(
                data_folder=self.data_folder,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                split="predict",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
