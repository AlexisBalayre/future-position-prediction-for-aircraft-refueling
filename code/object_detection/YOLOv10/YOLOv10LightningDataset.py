import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class YOLOv10LightningDataset(Dataset):
    def __init__(self, img_dir, label_dir="", target_size=(640, 640), stage="train"):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.stage = stage
        self.imgs = list(sorted(os.listdir(img_dir)))

        if stage != "predict":
            self.labels = list(sorted(os.listdir(label_dir)))

        common_transforms = [
            A.Resize(width=self.target_size[0], height=self.target_size[1], p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]

        train_transforms = common_transforms + [A.HorizontalFlip(p=0.5)]

        self.transform = A.Compose(
            train_transforms if stage == "train" else common_transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = cv2.imread(img_path)

        if self.stage != "predict":
            label_path = os.path.join(self.label_dir, self.labels[idx])
            with open(label_path, "r") as f:
                line = f.readline()
                class_id, cx, cy, bw, bh = map(float, line.strip().split())

                bbox = [cx, cy, bw, bh]
                class_labels = [class_id]

            transformed = self.transform(
                image=image, bboxes=[bbox], class_labels=class_labels
            )
            image = transformed["image"]
            bbox = torch.tensor(transformed["bboxes"][0], dtype=torch.float32)
            class_label = torch.tensor(transformed["class_labels"][0], dtype=torch.long)

            return image, bbox, class_label
        else:
            transformed = self.transform(image=image)
            image = transformed["image"]

            return image
