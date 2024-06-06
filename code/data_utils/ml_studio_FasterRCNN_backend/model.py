import json
import os
from dotenv import load_dotenv
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.transforms import ToTensor
from label_studio_ml.model import LabelStudioMLBase
from PIL import Image


# Load environment variables
load_dotenv()


class CustomDataset(Dataset):
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = annotation["image_path"]
        boxes = annotation["boxes"]
        labels = annotation["labels"]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return image, target


class AirplaneFuelPortDetector(LabelStudioMLBase):
    def __init__(self, batch_size=16, num_epochs=10, lr=5e-4, **kwargs):
        """
        Initialise the model.

        Args:
            batch_size (int, optional): Batch size for training. Defaults to 16.
            num_epochs (int, optional): Number of epochs for training. Defaults to 10.
            lr (_type_, optional): Learning rate for training. Defaults to 5e-4.
        """

        super(AirplaneFuelPortDetector, self).__init__(**kwargs)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, tasks, **kwargs):
        """
        Make predictions on the given tasks.

        Args:
            tasks: List of tasks to make predictions on

        Returns:
            List of predictions
        """
        self.model.eval()

        predictions = []
        for task in tasks:
            image_url = task["data"]["image"]
            image_path = self.get_local_path(image_url)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image: {e}")
                continue
            image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)

            results = []
            for box, score, label in zip(
                outputs[0]["boxes"], outputs[0]["scores"], outputs[0]["labels"]
            ):
                if score > 0.1:
                    x_min, y_min, x_max, y_max = box.tolist()
                    width, height = image.size

                    x_min_ratio = x_min / width
                    y_min_ratio = y_min / height
                    width_ratio = (x_max - x_min) / width
                    height_ratio = (y_max - y_min) / height

                    result = {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": x_min_ratio * 100,
                            "y": y_min_ratio * 100,
                            "width": width_ratio * 100,
                            "height": height_ratio * 100,
                            "rectanglelabels": ["Fuel Port"],
                            "score": score.item(),
                        },
                    }
                    results.append(result)

            predictions.append({"result": results})
        return predictions

    def _get_annotated_dataset(self, project_id):
        """Retrieve annotated data from Label Studio API"""
        HOSTNAME = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
        API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "my_api_key")
        download_url = f'{HOSTNAME.rstrip("/")}/api/projects/{project_id}/export'

        for _ in range(3):  # Retry mechanism: 3 attempts
            response = requests.get(
                download_url, headers={"Authorization": f"Token {API_KEY}"}
            )
            if response.status_code == 200:
                return json.loads(response.content)
            else:
                print(f"Failed to fetch data: {response.status_code}. Retrying...")

        raise Exception(
            f"Can't load task data using {download_url}, response status_code = {response.status_code}"
        )

    def fit(self, event, data, **kwargs):
        """
        Fit the model based on labeled data from Label Studio.

        Args:
            event: Event name
            data: Data from Label Studio
        """
        self.model.train()

        project_id = data["project"]["id"]
        tasks = self._get_annotated_dataset(project_id)

        annotations = []
        for task in tasks:
            if not task.get("annotations"):
                continue

            annotation = task["annotations"][0]
            if annotation.get("skipped") or annotation.get("was_cancelled"):
                continue

            image_url = task["data"]["image"]
            try:
                image_path = self.get_local_path(image_url)
                width, height = Image.open(image_path).size
            except requests.HTTPError as e:
                print(f"HTTP error for {image_url}: {e}")
                continue
            except Exception as e:
                print(f"Failed to process {image_url}: {e}")
                continue

            if len(annotation["result"]) != 1:
                continue
            result = annotation["result"][0]
            value = result["value"]
            label = value["rectanglelabels"][0]

            if label == "Fuel Port":
                x = value["x"] / 100 * width
                y = value["y"] / 100 * height
                w = value["width"] / 100 * width
                h = value["height"] / 100 * height

                if x != 0 or y != 0 or w != 0 or h != 0:
                    annotations.append(
                        {
                            "image_path": image_path,
                            "boxes": [[x, y, x + w, y + h]],
                            "labels": [1],
                        }
                    )

        dataset = CustomDataset(annotations, transform=ToTensor())
        data_loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for epoch in range(self.num_epochs):
            for images, targets in data_loader:
                images = [image.to(self.device) for image in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

        self.model.eval()
