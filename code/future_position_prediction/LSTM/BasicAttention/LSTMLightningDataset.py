import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class LSTMLightningDataset(Dataset):
    def __init__(
        self,
        json_file,
        input_frames,
        output_frames,
        images_folder,
        target_size=(480, 640),
        stage="train",
    ):
        self.target_size = target_size
        self.images_folder = images_folder
        self.input_frames = input_frames
        self.output_frames = output_frames

        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            total_frames = input_frames + output_frames
            for idx in range(0, len(frames) - total_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                output_seq = frames[idx + input_frames : idx + total_frames]
                self.samples.append((video_id, input_seq, output_seq))

        self.original_height, self.original_width = 480, 640
        self.target_height, self.target_width = self.target_size
        self.width_ratio = self.target_width / self.original_width
        self.height_ratio = self.target_height / self.original_height

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            video_id, input_seq, output_seq = self.samples[idx]

            input_bboxes = []
            input_velocities = []
            input_accelerations = []
            for i, frame in enumerate(input_seq):
                bbox_position, bbox_velocity, bbox_acceleration = (
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"],
                        frame["bbox"],
                        frame["class_id"],
                        seq_idx=i,
                        sample_idx=idx,
                        is_input=True,
                    )
                )
                input_bboxes.append(bbox_position)
                input_velocities.append(bbox_velocity)
                input_accelerations.append(bbox_acceleration)

            output_bboxes = []
            output_velocities = []
            for i, frame in enumerate(output_seq):
                bbox_position, bbox_velocity, _ = (
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"],
                        frame["bbox"],
                        frame["class_id"],
                        seq_idx=i,
                        sample_idx=idx,
                        is_input=False,
                    )
                )
                output_bboxes.append(bbox_position)
                output_velocities.append(bbox_velocity)

            input_bboxes = np.array(input_bboxes)
            input_velocities = np.array(input_velocities)
            input_accelerations = np.array(input_accelerations)
            output_bboxes = np.array(output_bboxes)
            output_velocities = np.array(output_velocities)

            input_bboxes = torch.tensor(input_bboxes, dtype=torch.float32)
            input_velocities = torch.tensor(input_velocities, dtype=torch.float32)
            input_accelerations = torch.tensor(input_accelerations, dtype=torch.float32)
            output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32)
            output_velocities = torch.tensor(output_velocities, dtype=torch.float32)

            return (
                video_id,
                input_bboxes,
                input_velocities,
                input_accelerations,
                output_bboxes,
                output_velocities,
            )

        except Exception as e:
            print(e)
            print("Error in sample: ", idx)
            print("Video ID: ", video_id)
            return None

    def load_and_transform_image_with_bbox(
        self, image_name, bbox, class_id, seq_idx=0, sample_idx=0, is_input=True
    ):
        if class_id is None or bbox == []:
            class_id = 3
            bbox = [0, 0, 0, 0]

        bbox = [0 if coord is None else coord for coord in bbox]

        x, y, w, h = bbox
        x = x * self.width_ratio
        y = y * self.height_ratio
        w = w * self.width_ratio
        h = h * self.height_ratio
        bbox_position = [x, y, w, h]

        if seq_idx == 0 and is_input:
            bbox_velocity = [0, 0, 0, 0]
            bbox_acceleration = [0, 0, 0, 0]
        else:
            if is_input:
                prev_bbox = self.samples[sample_idx][1][seq_idx - 1]["bbox"]
            else:
                prev_bbox = (
                    self.samples[sample_idx][1][-1]["bbox"]
                    if seq_idx == 0
                    else self.samples[sample_idx][2][seq_idx - 1]["bbox"]
                )

            prev_bbox = [
                0 if coord is None or np.isnan(float(coord)) else coord
                for coord in prev_bbox
            ]
            prev_x, prev_y, prev_w, prev_h = prev_bbox
            prev_x = prev_x * self.width_ratio
            prev_y = prev_y * self.height_ratio
            prev_w = prev_w * self.width_ratio
            prev_h = prev_h * self.height_ratio
            bbox_velocity = [x - prev_x, y - prev_y, w - prev_w, h - prev_h]

            if seq_idx <= 1:
                bbox_acceleration = [0, 0, 0, 0]
            else:
                if is_input:
                    prev_prev_bbox = self.samples[sample_idx][1][seq_idx - 2]["bbox"]
                else:
                    prev_prev_bbox = (
                        self.samples[sample_idx][1][-2]["bbox"]
                        if seq_idx == 1
                        else self.samples[sample_idx][2][seq_idx - 2]["bbox"]
                    )

                prev_prev_bbox = [
                    0 if coord is None or np.isnan(float(coord)) else coord
                    for coord in prev_prev_bbox
                ]
                prev_prev_x, prev_prev_y, prev_prev_w, prev_prev_h = prev_prev_bbox
                prev_prev_x = prev_prev_x * self.width_ratio
                prev_prev_y = prev_prev_y * self.height_ratio
                prev_prev_w = prev_prev_w * self.width_ratio
                prev_prev_h = prev_prev_h * self.height_ratio
                prev_velocity = [
                    prev_x - prev_prev_x,
                    prev_y - prev_prev_y,
                    prev_w - prev_prev_w,
                    prev_h - prev_prev_h,
                ]
                bbox_acceleration = [
                    v - pv for v, pv in zip(bbox_velocity, prev_velocity)
                ]

        return bbox_position, bbox_velocity, bbox_acceleration
