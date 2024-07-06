import os
import torch
import json
import numpy as np
from PIL import Image
from YOLOv10.YOLOv10DetectionPredictor import YOLOv10DetectionPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tqdm  # tqdm is used to show the progress of the feature extraction process


class FeatureExtractor:
    def __init__(
        self,
        yolo_checkpoint_path,
        yolo_intermediate_layer,
        images_folder,
        output_folder,
    ):
        self.yolo_predictor = YOLOv10DetectionPredictor(yolo_checkpoint_path)
        self.yolo_intermediate_layer = yolo_intermediate_layer
        self.images_folder = images_folder
        self.output_folder = output_folder

        self.transform = A.Compose(
            [
                A.Resize(width=640, height=640, p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )

    def extract_and_store_features(self):
        # For each image in the images_folder folder, extract the feature vector and store it
        for frame_name in tqdm.tqdm(os.listdir(self.images_folder)):
            image_path = os.path.join(self.images_folder, frame_name)
            feature_vector = self._extract_feature_vector(image_path)
            self._store_feature_vector(frame_name, feature_vector)

    def _extract_feature_vector(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        transformed = self.transform(image=image)
        transformed_image = transformed["image"].unsqueeze(0)  # Add batch dimension

        feature_vector, _ = self.yolo_predictor.predict_with_intermediate_output(
            transformed_image, self.yolo_intermediate_layer
        )
        return feature_vector  # Remove batch dimension

    def _store_feature_vector(self, frame_name, feature_vector):
        output_path = os.path.join(
            self.output_folder, f"{os.path.splitext(frame_name)[0]}.pt"
        )
        torch.save(feature_vector, output_path)


if __name__ == "__main__":
    yolo_checkpoint_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/yolov10s.pth"
    yolo_intermediate_layer = "22"
    images_folder = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/images"
    output_folder = "/Volumes/ALEXIS/Thesis/Features/22"

    extractor = FeatureExtractor(
        yolo_checkpoint_path, yolo_intermediate_layer, images_folder, output_folder
    )

    extractor.extract_and_store_features()
