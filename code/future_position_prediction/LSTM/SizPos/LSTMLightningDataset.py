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
        double_train=False,
    ):
        self.target_size = target_size
        self.images_folder = images_folder
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stage = stage

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

                if stage == "train" and double_train:
                    reversed_input_seq = output_seq[::-1]
                    reversed_output_seq = input_seq[::-1]
                    all_seq = np.concatenate(
                        [reversed_input_seq, reversed_output_seq], axis=0
                    )
                    reversed_input_seq, reversed_output_seq = np.split(
                        all_seq, [input_frames]
                    )
                    self.samples.append(
                        (video_id, reversed_input_seq, reversed_output_seq)
                    )

        self.original_height, self.original_width = 480, 640
        self.target_height, self.target_width = self.target_size
        self.width_ratio = self.target_width / self.original_width
        self.height_ratio = self.target_height / self.original_height

    def __len__(self):
        return len(self.samples)

    def augment_bbox_sequence(self, bboxes):
        augmented_bboxes = bboxes.astype(np.float64)
        seq_length = len(augmented_bboxes)

        # Simulate camera movement (panning)
        pan_x = np.random.normal(0, 0.01, seq_length)
        pan_y = np.random.normal(0, 0.01, seq_length)
        pan_x = np.cumsum(pan_x) / 5
        pan_y = np.cumsum(pan_y) / 5

        # Simulate camera zoom
        zoom_factor = np.random.uniform(0.98, 1.02, seq_length)
        zoom_factor = np.cumprod(zoom_factor)

        # Apply augmentations
        augmented_bboxes[:, 0] += pan_x  # x center
        augmented_bboxes[:, 1] += pan_y  # y center
        augmented_bboxes[:, 2] *= zoom_factor  # width
        augmented_bboxes[:, 3] *= zoom_factor  # height

        # Add small noise
        noise = np.random.normal(0, 0.002, augmented_bboxes.shape)
        augmented_bboxes += noise

        # Clip values to ensure they remain in [0, 1] range
        return np.clip(augmented_bboxes, 0, 1)

    def __getitem__(self, idx):
        try:
            video_id, input_seq, output_seq = self.samples[idx]

            # Only convert to numpy arrays if we're going to augment
            if self.stage == "train" and np.random.random() < 0.5:
                input_bboxes = np.array(
                    [
                        [
                            float(0 if coord is None else coord)
                            for coord in frame["bbox"]
                        ]
                        for frame in input_seq
                    ],
                    dtype=np.float64,
                )
                output_bboxes = np.array(
                    [
                        [
                            float(0 if coord is None else coord)
                            for coord in frame["bbox"]
                        ]
                        for frame in output_seq
                    ],
                    dtype=np.float64,
                )
                all_bboxes = np.concatenate([input_bboxes, output_bboxes], axis=0)
                all_bboxes = self.augment_bbox_sequence(all_bboxes)
                input_bboxes, output_bboxes = np.split(all_bboxes, [len(input_seq)])

            else:
                input_bboxes = [frame["bbox"] for frame in input_seq]
                output_bboxes = [frame["bbox"] for frame in output_seq]

            input_positions = []
            input_sizes = []
            for i, bbox in enumerate(input_bboxes):
                pos, size = self.process_bbox(
                    bbox,
                    seq_idx=i,
                    sample_idx=idx,
                    is_input=True,
                )
                input_positions.append(pos)
                input_sizes.append(size)

            output_positions = []
            output_sizes = []
            for i, bbox in enumerate(output_bboxes):
                pos, size = self.process_bbox(
                    bbox,
                    seq_idx=i,
                    sample_idx=idx,
                    is_input=False,
                )
                output_positions.append(pos)
                output_sizes.append(size)

            input_positions = torch.tensor(input_positions, dtype=torch.float32)
            input_sizes = torch.tensor(input_sizes, dtype=torch.float32)
            output_positions = torch.tensor(output_positions, dtype=torch.float32)
            output_sizes = torch.tensor(output_sizes, dtype=torch.float32)

            return (
                video_id,
                input_positions,
                input_sizes,
                output_positions,
                output_sizes,
            )

        except Exception as e:
            print(e)
            print("Error in sample: ", idx)
            print("Video ID: ", video_id)
            return None

    def process_bbox(self, bbox, seq_idx=0, sample_idx=0, is_input=True):
        if bbox is None or len(bbox) == 0:
            bbox = [0, 0, 0, 0]

        bbox = [0 if coord is None else coord for coord in bbox]

        # YOLO format (xywh) normalized
        x, y, w, h = bbox

        bbox_position = [x, y, 0, 0, 0, 0]  # [x, y, velx, vely, accx, accy]
        bbox_size = [w, h, 0, 0]  # [w, h, deltaw, deltah]

        if seq_idx >= 0:
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

            # Calculate velocities (change in position)
            velx = x - prev_x
            vely = y - prev_y
            bbox_position[2] = velx
            bbox_position[3] = vely

            # Calculate change in size
            deltaw = w - prev_w
            deltah = h - prev_h
            bbox_size[2] = deltaw
            bbox_size[3] = deltah

            if seq_idx >= 1:
                prev_prev_bbox = (
                    self.samples[sample_idx][1][-2]["bbox"]
                    if seq_idx == 1
                    else self.samples[sample_idx][2][seq_idx - 2]["bbox"]
                )

                prev_prev_bbox = [
                    0 if coord is None or np.isnan(float(coord)) else coord
                    for coord in prev_prev_bbox
                ]

                prev_prev_x, prev_prev_y, _, _ = prev_prev_bbox

                # Calculate accelerations (change in velocity)
                prev_velx = prev_x - prev_prev_x
                prev_vely = prev_y - prev_prev_y
                accx = velx - prev_velx
                accy = vely - prev_vely
                bbox_position[4] = accx
                bbox_position[5] = accy

        return bbox_position, bbox_size
