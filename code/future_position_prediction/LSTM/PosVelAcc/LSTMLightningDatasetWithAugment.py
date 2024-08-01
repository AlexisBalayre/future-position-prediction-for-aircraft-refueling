import json
import numpy as np
import torch
from torch.utils.data import Dataset


class LSTMLightningDatasetWithAugment(Dataset):
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
        self.stage = stage

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

                # Double the dataset for train stage by reversing each sequence
                """ if stage == "train":
                    reversed_input_seq = output_seq[::-1]
                    reversed_output_seq = input_seq[::-1]
                    self.samples.append(
                        (video_id, reversed_input_seq, reversed_output_seq)
                    ) """

    def __len__(self):
        return len(self.samples)

    def augment_bbox_sequence(self, bboxes):
        augmented_bboxes = bboxes.astype(np.float64)  # Ensure float type
        seq_length = len(augmented_bboxes)

        # Simulate camera movement (panning)
        pan_x = np.random.normal(0, 0.01, seq_length)
        pan_y = np.random.normal(0, 0.01, seq_length)

        # Make the panning smooth
        pan_x = np.cumsum(pan_x) / 5
        pan_y = np.cumsum(pan_y) / 5

        augmented_bboxes[:, 0] += pan_x
        augmented_bboxes[:, 1] += pan_y

        # Simulate camera zoom
        zoom_factor = np.random.uniform(0.98, 1.02, seq_length)
        zoom_factor = np.cumprod(zoom_factor)  # Make zoom change smooth

        # Apply zoom to width and height
        augmented_bboxes[:, 2] *= zoom_factor
        augmented_bboxes[:, 3] *= zoom_factor

        # Add small noise to simulate detection inaccuracies
        noise = np.random.normal(0, 0.002, augmented_bboxes.shape)
        augmented_bboxes += noise

        # Clip values to ensure they remain in [0, 1] range
        augmented_bboxes = np.clip(augmented_bboxes, 0, 1)

        return augmented_bboxes

    def __getitem__(self, idx):
        try:
            video_id, input_seq, output_seq = self.samples[idx]

            input_bboxes = np.array(
                [
                    [float(0 if coord is None else coord) for coord in frame["bbox"]]
                    for frame in input_seq
                ],
                dtype=np.float32,
            )

            output_bboxes = np.array(
                [
                    [float(0 if coord is None else coord) for coord in frame["bbox"]]
                    for frame in output_seq
                ],
                dtype=np.float32,
            )

            if self.stage == "train":
                all_bboxes = np.concatenate([input_bboxes, output_bboxes], axis=0)
                if np.random.random() < 0.5:
                    all_bboxes = self.augment_bbox_sequence(all_bboxes)
                input_bboxes, output_bboxes = np.split(all_bboxes, [len(input_bboxes)])

            # Calculate velocities and accelerations
            input_velocities = np.diff(input_bboxes, axis=0, prepend=input_bboxes[:1])
            input_accelerations = np.diff(
                input_velocities, axis=0, prepend=input_velocities[:1]
            )
            output_velocities = np.diff(
                output_bboxes, axis=0, prepend=output_bboxes[:1]
            )

            # Convert to torch tensors
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
            print(f"Error in sample {idx}: {e}")
            print(f"Video ID: {video_id}")
            return None
