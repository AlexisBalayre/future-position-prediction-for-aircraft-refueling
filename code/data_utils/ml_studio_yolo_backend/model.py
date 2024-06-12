from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from PIL import Image
from ultralytics import YOLOv10


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model"""

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "0.0.1")
        self.yolo_model = YOLOv10(
            "/Users/alexis/Programmation/label-studio-ml-backend/yolo_backend/best.pt"
        )  # Initialize the model once during setup

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(
            f"""\
    Run prediction on {tasks}
    Received context: {context}
    Project ID: {self.project_id}
    Label config: {self.label_config}
    Parsed JSON Label config: {self.parsed_label_config}
    Extra params: {self.extra_params}"""
        )

        predictions = []
        for task in tasks:
            image_url = task["data"]["image"]
            image_path = self.get_local_path(
                image_url
            )  # Ensure this method is implemented in LabelStudioMLBase
            try:
                image = Image.open(image_path)
                width, height = image.size
                results = self.yolo_model(image_path)
                formatted_results = self.format_results(
                    results=results, width=width, height=height
                )  # Assuming another method to format results as per Label Studio requirement
                predictions.append({"result": formatted_results})

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        return ModelResponse(predictions=predictions)

    def format_results(self, width, height, results):
        """Formats YOLO results to Label Studio's prediction format"""
        formatted_predictions = []
        for result in results:
            for bbox in result.boxes:
                x_min, y_min, x_max, y_max = bbox.xyxy[0].tolist()

                x_min_ratio = x_min / width
                y_min_ratio = y_min / height
                width_ratio = (x_max - x_min) / width
                height_ratio = (y_max - y_min) / height

                formatted_predictions.append(
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": x_min_ratio * 100,
                            "y": y_min_ratio * 100,
                            "width": width_ratio * 100,
                            "height": height_ratio * 100,
                            "rectanglelabels": ["Fuel Port"],  # Assuming a fixed label
                            "score": bbox.conf[0].item(),  # Extract confidence score
                        },
                    }
                )
        return formatted_predictions
