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
        yolo_feature_vector_folder = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/features/23.cv3.2.2"
    ):
        self.target_size = target_size
        self.images_folder = images_folder
        self.yolo_feature_vector_folder = yolo_feature_vector_folder

        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            video_id = entry["video_id"]
            frames = entry["frames"]
            for idx in range(len(frames) - input_frames - output_frames + 1):
                input_seq = frames[idx : idx + input_frames]
                output_seq = [frames[idx + input_frames + output_frames - 1]]
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
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            video_id, input_seq, output_seq = self.samples[idx]

            input_images, input_bboxes, input_class_ids, input_feature_vectors = zip(
                *[
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"],
                        frame["bbox"],
                        frame["class_id"],
                        feature_vector=True,
                    )
                    for frame in input_seq
                ]
            )
            _, output_bboxes, _, _ = zip(
                *[
                    self.load_and_transform_image_with_bbox(
                        frame["image_name"], frame["bbox"], frame["class_id"]
                    )
                    for frame in output_seq
                ]
            )

            input_images = torch.stack(input_images)
            input_bboxes = torch.tensor(input_bboxes, dtype=torch.float32)
            input_class_ids = torch.tensor(input_class_ids, dtype=torch.long)
            input_feature_vectors = torch.tensor(
                np.array(input_feature_vectors), dtype=torch.float32
            )
            output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32).squeeze(0)

            return (
                input_images,
                input_bboxes,
                input_class_ids,
                input_feature_vectors,
                output_bboxes,
            )

        except Exception as e:
            print(e)
            print("Error in sample: ", idx)
            print("Video ID: ", video_id)

    def load_and_transform_image_with_bbox(
        self, image_name, bbox, class_id, feature_vector=False
    ):
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Handle No Detection
        if class_id == None or bbox == []:
            class_id = 3
            bbox = [0, 0, 0, 0]

        transformed_image = self.transform(image=image)["image"]

        # Resize bbox
        original_height, original_width = image.shape[:2]
        target_height, target_width = self.target_size
        x, y, w, h = bbox
        x = x * target_width / original_width
        y = y * target_height / original_height
        w = w * target_width / original_width
        h = h * target_height / original_height
        transformed_bbox = [x, y, w, h]

        if feature_vector:
            image_name = os.path.splitext(image_name)[0]
            feature_vector = torch.load(
                os.path.join(self.yolo_feature_vector_folder, image_name + ".pt")
            )
            return (transformed_image, transformed_bbox, class_id, feature_vector)
        return transformed_image, transformed_bbox, class_id, None
