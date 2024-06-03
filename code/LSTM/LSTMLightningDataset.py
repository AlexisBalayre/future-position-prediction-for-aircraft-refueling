import json
import torch
from torch.utils.data import Dataset


class LSTMLightningDataset(Dataset):
    def __init__(self, json_file, seq_length=10, stage="train"):
        with open(json_file, "r") as f:
            annotations = json.load(f)

        self.seq_length = seq_length
        self.annotations = annotations
        self.data, self.targets = self._create_sequences()

    def _create_sequences(self):
        xs = []
        ys = []
        for i in range(len(self.annotations) - self.seq_length):
            seq = self.annotations[i : i + self.seq_length]
            target = self.annotations[i + self.seq_length]
            x = [anno["bbox"] for anno in seq]
            y = target["bbox"]
            xs.append(x)
            ys.append(y)

        return torch.tensor(xs, dtype=torch.float32), torch.tensor(
            ys, dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
