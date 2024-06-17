import lightning as L
from torch.utils.data import DataLoader

from TransformerLightningDataset import TransformerLightningDataset


class TransformerLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset_path,
        val_dataset_path,
        test_dataset_path,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, input_frames=1, output_frames=3, stage=None, predict_dataset_path=None):
        if stage == "train" or stage is None:
            self.train_dataset = TransformerLightningDataset(
                self.train_dataset_path,
                input_frames=input_frames,
                output_frames=output_frames,
                stage="train",
            )
            self.val_dataset = TransformerLightningDataset(
                self.val_dataset_path,
                input_frames=input_frames,
                output_frames=output_frames,
                stage="val",
            )
        if stage == "test" or stage is None:
            self.test_dataset = TransformerLightningDataset(
                self.test_dataset_path,
                input_frames=input_frames,
                output_frames=output_frames,
                stage="test",
            )
        if stage == "predict" and predict_dataset_path is not None:
            self.predict_dataset = TransformerLightningDataset(
                predict_dataset_path,
                input_frames=input_frames,
                output_frames=output_frames,
                stage="predict",
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
