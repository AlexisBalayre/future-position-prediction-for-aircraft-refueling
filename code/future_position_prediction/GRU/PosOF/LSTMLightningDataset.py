import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


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

        self.samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            total_frames = input_frames + output_frames
            for idx in range(len(frames) - total_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                output_seq = frames[idx + input_frames : idx + total_frames]
                self.samples.append((video_id, input_seq, output_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, input_seq, output_seq = self.samples[idx]

        try:
            input_bboxes_position, input_bboxes_velocity, input_optical_flow = zip(
                *[
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"],
                        frame.get("bbox_position", [0, 0, 0, 0]),
                        frame.get("bbox_velocity", [0, 0, 0, 0]),
                        frame.get("optical_flow_file"),
                        seq_idx=i,
                        sample_idx=idx,
                        is_input=True,
                    )
                    for i, frame in enumerate(input_seq)
                ]
            )
            output_bboxes_position, output_bboxes_velocity, _ = zip(
                *[
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"],
                        frame.get("bbox_position", [0, 0, 0, 0]),
                        frame.get("bbox_velocity", [0, 0, 0, 0]),
                        frame.get("optical_flow_file"),
                        seq_idx=i,
                        sample_idx=idx,
                        is_input=False,
                    )
                    for i, frame in enumerate(output_seq)
                ]
            )

            input_bboxes_position = torch.tensor(
                input_bboxes_position, dtype=torch.float32
            )
            input_bboxes_velocity = torch.tensor(
                input_bboxes_velocity, dtype=torch.float32
            )

            input_optical_flow = np.array(input_optical_flow)
            input_optical_flow = torch.tensor(input_optical_flow, dtype=torch.float32)

            output_bboxes_position = torch.tensor(
                output_bboxes_position, dtype=torch.float32
            )
            output_bboxes_velocity = torch.tensor(
                output_bboxes_velocity, dtype=torch.float32
            )

            return (
                video_id,
                input_bboxes_position,
                input_bboxes_velocity,
                input_optical_flow,
                output_bboxes_position,
                output_bboxes_velocity,
            )

        except Exception as e:
            print(f"Error in sample: {idx}")
            print(f"Video ID: {video_id}")
            print(e)
            # Create a dummy fallback tensor to avoid returning None
            dummy_tensor = torch.zeros((self.input_frames, 4), dtype=torch.float32)
            dummy_optical_flow = torch.zeros(
                (self.input_frames, *self.target_size, 2), dtype=torch.float32
            )
            return (
                video_id,
                dummy_tensor,
                dummy_tensor,
                dummy_optical_flow,
                dummy_tensor,
                dummy_tensor,
            )

    def load_and_transform_image_with_bbox(
        self,
        image_name,
        bbox_position,
        bbox_velocity,
        optical_flow_file,
        seq_idx=0,
        sample_idx=0,
        is_input=True,
    ):
        try:
            if optical_flow_file:
                optical_flow_path = os.path.join(
                    self.data_folder, self.split, "optical_flows", optical_flow_file
                )
                optical_flow = np.load(optical_flow_path)
            else:
                optical_flow = np.zeros(
                    (self.target_size[0], self.target_size[1], 2), dtype=np.float32
                )

            # Validate bbox_position and bbox_velocity
            if (
                not bbox_position
                or not isinstance(bbox_position, (list, tuple))
                or len(bbox_position) != 4
            ):
                bbox_position = [0, 0, 0, 0]
            if (
                not bbox_velocity
                or not isinstance(bbox_velocity, (list, tuple))
                or len(bbox_velocity) != 4
            ):
                bbox_velocity = [0, 0, 0, 0]

            return bbox_position, bbox_velocity, optical_flow

        except Exception as e:
            print(f"Error loading and transforming image: {image_name}")
            print(e)
            return (
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                np.zeros(
                    (self.target_size[1], self.target_size[0], 2), dtype=np.float32
                ),
            )
