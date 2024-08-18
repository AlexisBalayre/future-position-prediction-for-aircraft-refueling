import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PosVelAccGRULightningDataset(Dataset):
    """
    A PyTorch Dataset for loading video sequences and generating input and output tensors
    for training and evaluating the PosVelAccGRU model.

    Args:
        json_file (str): Path to the JSON file containing video data.
        input_frames (int): Number of frames to use as input.
        output_frames (int): Number of frames to predict as output.
        stage (str, optional): Dataset stage, either 'train', 'val', or 'test'. Default is 'train'.
        double_train (bool, optional): If True, includes reversed sequences in the training data. Default is False.
    """

    def __init__(
        self, json_file, input_frames, output_frames, stage="train", double_train=False
    ):
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

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def augment_bbox_sequence(self, bboxes):
        """
        Applies augmentation to a sequence of bounding boxes by simulating camera movement
        and adding noise.

        Args:
            bboxes (np.ndarray): Array of bounding boxes with shape (seq_length, 4).

        Returns:
            np.ndarray: Augmented bounding boxes, clipped to the range [0, 1].
        """
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
        """
        Retrieves a single sample of input and output sequences, along with computed velocities and accelerations.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple: A tuple containing the video ID (str), input bounding boxes (torch.Tensor),
                   input velocities (torch.Tensor), input accelerations (torch.Tensor),
                   output bounding boxes (torch.Tensor), and output velocities (torch.Tensor).

        Raises:
            ValueError: If there is an issue with the sample data.
        """
        try:
            video_id, input_seq, output_seq = self.samples[idx]

            input_bboxes = np.array(
                [
                    [float(0 if coord is None else coord) for coord in frame["bbox"]]
                    for frame in input_seq
                ],
                dtype=np.float64,
            )

            output_bboxes = np.array(
                [
                    [float(0 if coord is None else coord) for coord in frame["bbox"]]
                    for frame in output_seq
                ],
                dtype=np.float64,
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
            raise ValueError(f"Error in sample {idx}: {e}")
