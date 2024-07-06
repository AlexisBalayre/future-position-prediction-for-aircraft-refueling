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
        target_size=(640, 640),
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
            for idx in range(0, len(frames) - total_frames + 1, total_frames):
                input_seq = frames[idx : idx + input_frames]
                output_seq = frames[idx + input_frames : idx + total_frames]
                self.samples.append((video_id, input_seq, output_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            video_id, input_seq, output_seq = self.samples[idx]

            input_bboxes, input_delta_bboxes = zip(
                *[
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"], frame["bbox"], frame["class_id"], idx=i
                    )
                    for i, frame in enumerate(input_seq)
                ]
            )
            output_bboxes, output_delta_bboxes = zip(
                *[
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"], frame["bbox"], frame["class_id"], idx=i
                    )
                    for i, frame in enumerate(output_seq)
                ]
            )

            input_bboxes = torch.tensor(input_bboxes, dtype=torch.float32)
            input_delta_bboxes = torch.tensor(input_delta_bboxes, dtype=torch.float32)
            output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32)
            output_delta_bboxes = torch.tensor(output_delta_bboxes, dtype=torch.float32)

            return (
                input_bboxes,
                input_delta_bboxes,
                output_bboxes,
                output_delta_bboxes,
            )

        except Exception as e:
            print(e)
            print("Error in sample: ", idx)
            print("Video ID: ", video_id)

    def load_and_transform_image_with_bbox(
        self, image_name, bbox, class_id, feature_vector=False, idx=0
    ):
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Handle No Detection
        if class_id == None or bbox == []:
            class_id = 3
            bbox = [0, 0, 0, 0]

        # Resize bbox
        original_height, original_width = image.shape[:2]
        target_height, target_width = self.target_size
        x, y, w, h = bbox
        x = x * target_width / original_width
        y = y * target_height / original_height
        w = w * target_width / original_width
        h = h * target_height / original_height
        transformed_bbox = [x, y, w, h]

        # Calculate delta bbox
        if idx == 0:
            delta_bbox = [0, 0, 0, 0]
        else:
            prev_bbox = self.samples[idx - 1][1][-1]["bbox"]
            prev_x, prev_y, prev_w, prev_h = prev_bbox
            prev_x = prev_x * target_width / original_width
            prev_y = prev_y * target_height / original_height
            prev_w = prev_w * target_width / original_width
            prev_h = prev_h * target_height / original_height
            delta_bbox = [x - prev_x, y - prev_y, w - prev_w, h - prev_h]

        return transformed_bbox, delta_bbox
