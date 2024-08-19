import json
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLOv10

from filters import (
    sg_filter_smoothing,
)


def handle_null_values(trajectory):
    """
    Handle null values in a trajectory by linear interpolation.

    Args:
        trajectory (numpy.ndarray): A trajectory with potential null values in x, y, width, and height columns.

    Returns:
        numpy.ndarray: The trajectory with null values interpolated.
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(trajectory, columns=["x_center", "y_center", "width", "height"])

    # Set types
    df["x_center"] = df["x_center"].astype(np.float64)
    df["y_center"] = df["y_center"].astype(np.float64)
    df["width"] = df["width"].astype(np.float64)
    df["height"] = df["height"].astype(np.float64)

    # Interpolate missing values
    df.interpolate(method="linear", limit_direction="both", inplace=True)
    return df.values


def run_detections(
    input_video_path,
    output_json_path,
    yolo_weights_path,
    output_frame=30,
    smooth_filter=True,
):
    """
    Run object detection on a video using YOLOv10 and save the results to a JSON file.

    Args:
        input_video_path (str): Path to the input video file.
        output_json_path (str): Path to save the output JSON file with detections.
        yolo_weights_path (str): Path to the YOLOv10 weights file.
        output_frame (int): The number of frames to look ahead for future bounding box ground truth.
        smooth_filter (bool): Whether to apply a smoothing filter to the trajectory.

    Returns:
        None
    """
    model = YOLOv10(yolo_weights_path)
    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), "Error reading video file"

    frame_count = 0
    object_id = None
    detections = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = model.track(frame, verbose=False, persist=True)

        if results and len(results) > 0:
            boxes = results[0].boxes
            if len(boxes) > 0:
                box = boxes[0]
                b = box.xywhn[0].cpu().numpy().astype(float)
                current_detection = {
                    "frame_id": frame_count,
                    "x_center": b[0],
                    "y_center": b[1],
                    "w": b[2],
                    "h": b[3],
                }
            else:
                current_detection = {
                    "frame_id": frame_count,
                    "x_center": None,
                    "y_center": None,
                    "w": None,
                    "h": None,
                }
        else:
            current_detection = {
                "frame_id": frame_count,
                "x_center": None,
                "y_center": None,
                "w": None,
                "h": None,
            }
        detections.append(current_detection)

    cap.release()

    detections.sort(key=lambda x: x["frame_id"])

    # Convert detections to numpy array for processing
    detection_array = np.array(
        [[d["x_center"], d["y_center"], d["w"], d["h"]] for d in detections]
    )

    # Handle null values and smoothing
    detection_array = handle_null_values(detection_array)
    if smooth_filter:
        smoothed_detections = sg_filter_smoothing(detection_array, window_length=20)
    else:
        smoothed_detections = detection_array

    # Clip values to [0, 1]
    smoothed_detections = np.clip(smoothed_detections, 0, 1)

    output_detections = []
    for i, detection in enumerate(smoothed_detections):
        future_frame_index = i + output_frame  # Look 'output_frame' frames ahead

        if future_frame_index < len(smoothed_detections):
            future_bbox_gt = np.array(smoothed_detections[future_frame_index])
        else:
            future_bbox_gt = np.array([0, 0, 0, 0])

        try:
            output_detections.append(
                {
                    "frame_id": int(detections[i]["frame_id"]),  # Convert to int
                    "current_bbox": {
                        "x_center": float(detection[0]),  # Convert to float
                        "y_center": float(detection[1]),
                        "w": float(detection[2]),
                        "h": float(detection[3]),
                    },
                    "future_bbox_gt": {
                        "x_center": float(future_bbox_gt[0]),  # Convert to float
                        "y_center": float(future_bbox_gt[1]),
                        "w": float(future_bbox_gt[2]),
                        "h": float(future_bbox_gt[3]),
                    },
                }
            )
        except Exception as e:
            print(detection)

    with open(output_json_path, "w") as f:
        json.dump(output_detections, f, indent=2)

    print(f"Saved smoothed detections to {output_json_path}")


def load_detections(input_json_path):
    """
    Load detection results from a JSON file.

    Args:
        input_json_path (str): Path to the input JSON file with detection results.

    Returns:
        list: A list of detections loaded from the JSON file.
    """
    with open(input_json_path, "r") as f:
        return json.load(f)
