import json
import torch
from torch.utils.data import Dataset


class LSTMLightningDataset(Dataset):
    def __init__(self, json_file, input_frames, output_frames, stage="train"):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.input_frames = input_frames
        self.output_frames = output_frames
        self.samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            for idx in range(len(frames) - input_frames - output_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                output_seq = [frames[idx + input_frames + output_frames - 1]]
                self.samples.append((video_id, input_seq, output_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, input_seq, output_seq = self.samples[idx]

        input_bboxes = [
            self.extract_features(input_seq[i], input_seq[i - 1] if i > 0 else None)
            for i in range(len(input_seq))
        ]
        output_bboxes = [self.extract_features(output_seq[0], input_seq[-1])]

        input_bboxes = torch.tensor(input_bboxes, dtype=torch.float32)
        output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32).squeeze(0)

        return input_bboxes, output_bboxes

    def extract_features(self, current_frame, previous_frame):
        x, y, w, h = current_frame["bbox"]

        return [x, y, w, h]

        """ if previous_frame is not None:
            x_prev, y_prev, w_prev, h_prev = previous_frame["bbox"]
            vx = x - x_prev
            vy = y - y_prev
            dw = w - w_prev
            dh = h - h_prev
        else:
            vx, vy, dw, dh = 0, 0, 0, 0

        return [x, y, w, h, vx, vy, dw, dh] """
