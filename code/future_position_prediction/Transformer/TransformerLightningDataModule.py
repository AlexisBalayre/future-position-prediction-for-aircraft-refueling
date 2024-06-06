import lightning as L
from torch.utils.data import DataLoader

from TransformerLightningDataset import TransformerLightningDataset


class TransformerLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset_path,
        val_dataset_path,
        test_dataset_path,
        seq_length=10,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Create datasets
        self.train_dataset = TransformerLightningDataset(self.train_dataset_path, self.seq_length)
        self.val_dataset = TransformerLightningDataset(self.val_dataset_path, self.seq_length)
        self.test_dataset = TransformerLightningDataset(self.test_dataset_path, self.seq_length)

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
