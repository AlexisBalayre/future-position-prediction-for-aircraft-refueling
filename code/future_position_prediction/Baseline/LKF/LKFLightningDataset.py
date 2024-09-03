import json
import torch
from torch.utils.data import Dataset


class LKFLightningDataset(Dataset):
    """Dataset for loading and processing video frame data, specifically for
    handling sequences of bounding box positions.

    Args:
        json_file (str): Path to the JSON file containing video data.
        input_frames (int): Number of frames to be used as input.
        output_frames (int): Number of frames to be predicted/output.
        stage (str, optional): Stage of the dataset, e.g., 'train', 'val', 'test'. Defaults to 'train'.
    """

    def __init__(
        self,
        json_file: str,
        input_frames: int,
        output_frames: int,
        stage: str = "train",
    ):
        """Initializes the LKFLightningDataset with the provided parameters.

        Args:
            json_file (str): Path to the JSON file containing video data.
            input_frames (int): Number of frames to be used as input.
            output_frames (int): Number of frames to be predicted/output.
            stage (str, optional): Stage of the dataset, e.g., 'train', 'val', 'test'. Defaults to 'train'.

        Raises:
            ValueError: If the JSON file is invalid.
            FileNotFoundError: If the JSON file is not found.
        """
        try:
            with open(json_file, "r") as f:
                self.data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {json_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        self.samples = []

        # Create samples from the data by extracting input and output sequences.
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            total_frames = input_frames + output_frames

            # Extract sequences of frames for each video entry.
            for idx in range(0, len(frames) - total_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                output_seq = frames[idx + input_frames : idx + total_frames]
                self.samples.append((video_id, input_seq, output_seq))

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a sample from the dataset based on the index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input bounding box positions and the output bounding box positions.
        """
        video_id, input_seq, output_seq = self.samples[idx]

        # Load bounding box positions for the input sequence.
        input_bboxes_position = torch.tensor(
            [self.load_bbox(frame["bbox"]) for frame in input_seq], dtype=torch.float32
        ).transpose(0, 1)

        # Load bounding box positions for the output sequence.
        output_bboxes_position = torch.tensor(
            [self.load_bbox(frame["bbox"]) for frame in output_seq], dtype=torch.float32
        ).transpose(0, 1)

        return input_bboxes_position, output_bboxes_position

    @staticmethod
    def load_bbox(bbox: list) -> list:
        """Converts a bounding box representation into a list, replacing any None values with zeros.

        Args:
            bbox (list): A list representing the bounding box coordinates.

        Returns:
            list: Processed list of bounding box coordinates.
        """
        return [0 if coord is None else coord for coord in bbox]
