import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from LSTMLightningDataset import LSTMLightningDataset


class LSTMLightningDataModule(L.LightningDataModule):

    def __init__(
        self,
        train_dataset_path,
        val_dataset_path,
        test_dataset_path,
        input_frames,
        output_frames,
        images_folder,
        batch_size=16,
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.images_folder = images_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_frames = input_frames
        self.output_frames = output_frames

    def custom_collate(batch):
        max_length = max(len(item[0]) for item in batch)
        for i, (data, target) in enumerate(batch):
            pad_size = max_length - len(data)
            batch[i] = (F.pad(data, (0, 0, 0, pad_size)), target)
        return default_collate(batch)

    def setup(self, stage, predict_dataset_path=None):
        if stage == "train" or stage is None:
            self.train_dataset = LSTMLightningDataset(
                self.train_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                images_folder=self.images_folder,
                stage="train",
            )
            self.val_dataset = LSTMLightningDataset(
                self.val_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                images_folder=self.images_folder,
                stage="val",
            )
        if stage == "test" or stage is None:
            self.test_dataset = LSTMLightningDataset(
                self.test_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                images_folder=self.images_folder,
                stage="test",
            )
        if stage == "predict" and predict_dataset_path is not None:
            self.predict_dataset = LSTMLightningDataset(
                predict_dataset_path,
                input_frames=self.input_frames,
                output_frames=self.output_frames,
                images_folder=self.images_folder,
                stage="predict",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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
