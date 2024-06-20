import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


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

        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            for idx in range(len(frames) - input_frames - output_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                output_seq = frames[
                    idx + input_frames : idx + input_frames + output_frames
                ]
                self.samples.append((video_id, input_seq, output_seq))

        self.transform = A.Compose(
            [
                A.Resize(width=self.target_size[0], height=self.target_size[1], p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],
            bbox_params=A.BboxParams(format="albumentations", label_fields=[]),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, input_seq, output_seq = self.samples[idx]

        input_images, input_bboxes = zip(
            *[
                self.load_and_transform_image_with_bbox(
                    frame["image_name"], frame["bbox"]
                )
                for frame in input_seq
            ]
        )
        output_images, output_bboxes = zip(
            *[
                self.load_and_transform_image_with_bbox(
                    frame["image_name"], frame["bbox"]
                )
                for frame in output_seq
            ]
        )

        input_images = torch.stack(input_images)
        input_bboxes = torch.tensor(input_bboxes, dtype=torch.float32)

        output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32).squeeze(0)

        return input_images, input_bboxes, output_bboxes

    def load_and_transform_image_with_bbox(self, image_name, bbox):
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        transformed = self.transform(image=image, bboxes=[bbox])
        transformed_image = transformed["image"]
        transformed_bbox = transformed["bboxes"][0]

        return transformed_image, transformed_bbox
