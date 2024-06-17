import json
import torch
from torch.utils.data import Dataset


class LSTMLightningDataset(Dataset):
    def __init__(self, json_file, input_frames, output_frames, stage="train"):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            for idx in range(len(frames) - input_frames - output_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                # Keep only the last frame for the output sequence
                output_seq = [frames[idx + input_frames + output_frames - 1]]
                self.samples.append((video_id, input_seq, output_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, input_seq, output_seq = self.samples[idx]

        input_bboxes = [frame["bbox"] for frame in input_seq]
        output_bboxes = [frame["bbox"] for frame in output_seq]

        input_bboxes = torch.tensor(input_bboxes, dtype=torch.float32)
        output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32).squeeze(
            0
        )  # Ensure correct dimensions

        return input_bboxes, output_bboxes
