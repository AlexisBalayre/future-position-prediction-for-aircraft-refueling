from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from PIL import Image
from ultralytics import YOLOv10


class Yolov10RefuellingPortDetector(LabelStudioMLBase):
    """
    A custom model class that integrates YOLOv10 with Label Studio for detecting refuelling ports in images.

    This class provides methods to set up the model, perform predictions on a set of tasks, and format the model's
    predictions to be compatible with Label Studio's format.

    Attributes:
        model_version (str): The version of the model.
        yolo_model (YOLOv10): The YOLOv10 model used for detection.
    """

    def setup(self):
        """
        Set up the model by configuring any necessary parameters.

        This method is called once when the model is initialized. The YOLOv10 model is loaded here.
        """
        self.set("model_version", "0.0.1")
        self.yolo_model = YOLOv10(
            ""
        )  # Initialize the model once during setup

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """
        Perform predictions on a list of tasks using the YOLOv10 model.

        Args:
            tasks (List[Dict]): A list of tasks in JSON format from Label Studio.
            context (Optional[Dict]): Additional context in JSON format from Label Studio.

        Returns:
            ModelResponse: The model's predictions in Label Studio's format.
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

                # Format the results according to Label Studio's format
                formatted_results = self.format_results(
                    results=results, width=width, height=height
                )
                predictions.append({"result": formatted_results})

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        return ModelResponse(predictions=predictions)

    def format_results(self, results, width: int, height: int) -> List[Dict]:
        """
        Formats YOLO results to Label Studio's prediction format.

        Args:
            results: The results returned by the YOLO model.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            List[Dict]: A list of formatted predictions compatible with Label Studio.
        """
        formatted_predictions = []
        for result in results:
            for bbox in result.boxes:
                x_min, y_min, x_max, y_max = bbox.xyxy[0].tolist()

                # Calculate normalized coordinates
                x_min_ratio = x_min / width
                y_min_ratio = y_min / height
                width_ratio = (x_max - x_min) / width
                height_ratio = (y_max - y_min) / height

                # Append the formatted result
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
