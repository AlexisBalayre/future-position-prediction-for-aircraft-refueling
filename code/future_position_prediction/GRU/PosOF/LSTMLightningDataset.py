import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LSTMLightningDataset(Dataset):
    def __init__(
        self, split, input_frames, output_frames, data_folder, target_size=(480, 640)
    ):
        self.target_size = target_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.data_folder = data_folder
        self.split = split

        json_file = os.path.join(data_folder, split, "processed_data.json")
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.samples = self.preprocess_samples()

    def preprocess_samples(self):
        samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            total_frames = self.input_frames + self.output_frames
            for idx in range(len(frames) - total_frames + 1):
                input_seq = frames[idx : idx + self.input_frames]
                output_seq = frames[idx + self.input_frames : idx + total_frames]
                samples.append((video_id, input_seq, output_seq))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, input_seq, output_seq = self.samples[idx]

        try:
            input_data = self.process_sequence(input_seq, is_input=True)
            output_data = self.process_sequence(output_seq, is_input=False)

            return (video_id, *input_data, *output_data)

        except Exception as e:
            print(f"Error in sample: {idx}")
            print(f"Video ID: {video_id}")
            print(e)
            return self.get_dummy_data()

    def process_sequence(self, sequence, is_input):
        bboxes_position = []
        bboxes_size = []
        optical_flows = []

        for frame in sequence:
            bbox_position, bbox_size, optical_flow = (
                self.load_and_transform_image_with_bbox(
                    frame["image_name"],
                    frame.get("bbox_position", [0, 0, 0, 0, 0, 0]),
                    frame.get("bbox_size", [0, 0, 0, 0]),
                    frame.get("optical_flow_file"),
                    is_input=is_input,
                )
            )
            bboxes_position.append(bbox_position)
            bboxes_size.append(bbox_size)
            if is_input:
                optical_flows.append(optical_flow)

        bboxes_position = torch.tensor(bboxes_position, dtype=torch.float32)
        bboxes_size = torch.tensor(bboxes_size, dtype=torch.float32)

        if is_input:
            optical_flows = torch.tensor(np.array(optical_flows), dtype=torch.float32)
            return bboxes_position, bboxes_size, optical_flows
        else:
            return bboxes_position, bboxes_size

    def load_and_transform_image_with_bbox(
        self, image_name, bbox_position, bbox_size, optical_flow_file, is_input=True
    ):
        try:
            if is_input and optical_flow_file:
                optical_flow_path = os.path.join(
                    self.data_folder, self.split, "optical_flows", optical_flow_file
                )
                if os.path.exists(optical_flow_path):
                    optical_flow = np.load(optical_flow_path)
                else:
                    optical_flow = np.zeros((*self.target_size, 2), dtype=np.float32)
            else:
                optical_flow = np.zeros((*self.target_size, 2), dtype=np.float32)

            bbox_position = (
                bbox_position
                if isinstance(bbox_position, (list, tuple)) and len(bbox_position) == 6
                else [0, 0, 0, 0, 0, 0]
            )
            bbox_size = (
                bbox_size
                if isinstance(bbox_size, (list, tuple)) and len(bbox_size) == 4
                else [0, 0, 0, 0]
            )

            return bbox_position, bbox_size, optical_flow

        except Exception as e:
            print(f"Error loading and transforming image: {image_name}")
            print(e)
            return (
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0],
                np.zeros((*self.target_size, 2), dtype=np.float32),
            )

    def get_dummy_data(self):
        dummy_bbox_position = torch.zeros((self.input_frames, 6), dtype=torch.float32)
        dummy_bbox_size = torch.zeros((self.input_frames, 4), dtype=torch.float32)
        dummy_optical_flow = torch.zeros(
            (self.input_frames, *self.target_size, 2), dtype=torch.float32
        )
        dummy_output_bbox_position = torch.zeros(
            (self.output_frames, 6), dtype=torch.float32
        )
        dummy_output_bbox_size = torch.zeros(
            (self.output_frames, 4), dtype=torch.float32
        )
        return (
            "dummy_video_id",
            dummy_bbox_position,
            dummy_bbox_size,
            dummy_optical_flow,
            dummy_output_bbox_position,
            dummy_output_bbox_size,
        )
