import json
import torch
from torch.utils.data import Dataset


class CVLightningDataset(Dataset):
    def __init__(self, json_file, input_frames, output_frames, stage="train"):
        try:
            with open(json_file, "r") as f:
                self.data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {json_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        self.samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            total_frames = input_frames + output_frames
            for idx in range(0, len(frames) - total_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                output_seq = frames[idx + input_frames : idx + total_frames]
                self.samples.append((video_id, input_seq, output_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, input_seq, output_seq = self.samples[idx]

        input_bboxes_position = torch.tensor(
            [self.load_bbox(frame["bbox"]) for frame in input_seq], dtype=torch.float32
        ).transpose(0, 1)

        output_bboxes_position = torch.tensor(
            [self.load_bbox(frame["bbox"]) for frame in output_seq], dtype=torch.float32
        ).transpose(0, 1)

        return input_bboxes_position, output_bboxes_position

    @staticmethod
    def load_bbox(bbox):
        return [0 if coord is None else coord for coord in bbox]
