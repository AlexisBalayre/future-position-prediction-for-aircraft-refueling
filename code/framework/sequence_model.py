import sys
import os
import json
import torch
import cv2
import numpy as np
import random
from ultralytics.utils.plotting import Annotator
from collections import deque

from code.future_position_prediction.GRU.SizPos.GRULightningModelConcat import (
    GRULightningModelConcat,
)
from code.future_position_prediction.GRU.SizPos.utils import (
    convert_velocity_to_positions,
    convert_PosSize_to_PosVel,
)
from code.framework.filters import (
    sg_filter_smoothing,
    moving_average_smoothing,
    exponential_smoothing,
    adaptive_smoothing,
    hybrid_smoothing,
    kalman_filter_smoothing,
    modified_exponential_smoothing,
    gaussian_filter_smoothing,
)


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list): The first bounding box in the format [x_center, y_center, width, height].
        boxB (list): The second bounding box in the format [x_center, y_center, width, height].

    Returns:
        float: The IoU between the two bounding boxes.
    """
    # Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2) in normalized coordinates
    boxA_x1 = boxA[0] - boxA[2] / 2
    boxA_y1 = boxA[1] - boxA[3] / 2
    boxA_x2 = boxA[0] + boxA[2] / 2
    boxA_y2 = boxA[1] + boxA[3] / 2

    boxB_x1 = boxB[0] - boxB[2] / 2
    boxB_y1 = boxB[1] - boxB[3] / 2
    boxB_x2 = boxB[0] + boxB[2] / 2
    boxB_y2 = boxB[1] + boxB[3] / 2

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA_x1, boxB_x1)
    yA = max(boxA_y1, boxB_y1)
    xB = min(boxA_x2, boxB_x2)
    yB = min(boxA_y2, boxB_y2)

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA_x2 - boxA_x1) * (boxA_y2 - boxA_y1)
    boxBArea = (boxB_x2 - boxB_x1) * (boxB_y2 - boxB_y1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of the prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou


def set_fixed_seed(seed: int = 42) -> None:
    """
    Set a fixed seed for reproducibility.

    Args:
        seed (int, optional): The seed value. Defaults to 42.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_future_positions_pred(
    detections: List[Dict[str, Dict[str, float]]],
    gru_model_path: str,
    hparams_file: str,
    input_video_path: str,
    output_video_path: str,
    input_frames: int = 30,
    future_frames: int = 60,
    smooth_filter: str = "sa",
    lkf: bool = False,
    smoothing_params: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Predict future object positions in a video using a GRU model and visualize the predictions.

    Args:
        detections (list): A list of detections, each containing the frame ID and bounding box data.
        gru_model_path (str): Path to the trained GRU model checkpoint.
        hparams_file (str): Path to the hyperparameters file associated with the GRU model.
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video with visualized predictions.
        input_frames (int): Number of input frames used to predict future positions.
        future_frames (int): Number of future frames to predict.
        smooth_filter (str): Type of smoothing filter to apply ('sa', 'ma', 'es', etc.).
        lkf (bool): Whether to apply a Kalman filter to smooth the predictions.
        smoothing_params (dict): Additional parameters for the chosen smoothing filter.

    Returns:
        tuple: A tuple containing:
            - prediction_data (list): A list of dictionaries containing frame ID, predicted, and ground truth positions.
            - mean_metrics (dict): A dictionary of mean metrics such as ADE, FDE, AIoU, etc.
    """
    # Set Seed
    set_fixed_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GRU model
    gru_model = (
        GRULightningModelConcat.load_from_checkpoint(
            gru_model_path, hparams_file=hparams_file
        )
        .to(device)
        .eval()
    )

    # Load video and initialise output video writer
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Error reading video file")

    frame_w, frame_h, fps = [
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    ]

    result = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_w, frame_h)
    )

    track_history_bbox = deque(maxlen=input_frames)
    frame_counter = 0

    prediction_data = []
    metrics = {
        "ade": [],
        "fde": [],
        "ade_percent": [],
        "fde_percent": [],
        "aiou": [],
        "fiou": [],
    }

    frame_diagonal = np.sqrt(frame_w**2 + frame_h**2)

    future_box_pred = [0, 0, 0, 0]
    future_box_gt = [0, 0, 0, 0]
    next_frame_id = input_frames

    for detection in detections:
        success, frame = cap.read()
        if not success:
            break

        annotator = Annotator(frame, line_width=2)

        current_bbox = detection["current_bbox"]
        current_frame_id = detection["frame_id"]

        if track_history_bbox:
            prev_bbox = track_history_bbox[-1]
            velx = current_bbox["x_center"] - prev_bbox[0]
            vely = current_bbox["y_center"] - prev_bbox[1]
            deltaw = current_bbox["w"] - prev_bbox[6]
            deltah = current_bbox["h"] - prev_bbox[7]

            accx, accy = (
                (0, 0)
                if len(track_history_bbox) <= 1
                else (velx - prev_bbox[2], vely - prev_bbox[3])
            )
        else:
            velx = vely = accx = accy = deltaw = deltah = 0

        track_history_bbox.append(
            [
                current_bbox["x_center"],
                current_bbox["y_center"],
                velx,
                vely,
                accx,
                accy,
                current_bbox["w"],
                current_bbox["h"],
                deltaw,
                deltah,
            ]
        )

        if len(track_history_bbox) == input_frames:
            if next_frame_id == current_frame_id:
                try:
                    smoothed_history = np.array(track_history_bbox)
                    smoothed_positions = smoothed_history[:, :6]
                    smoothed_sizes = smoothed_history[:, 6:]

                    positions = (
                        torch.tensor(smoothed_positions, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                    sizes = (
                        torch.tensor(smoothed_sizes, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )

                    with torch.no_grad():
                        predicted_positions, predicted_sizes = gru_model(
                            positions, sizes
                        )

                    if lkf:
                        combined_positions = np.concatenate(
                            (smoothed_positions, predicted_positions.cpu().numpy()[0])
                        )
                        smoothed_combined_positions = kalman_filter_smoothing(
                            combined_positions
                        )
                        smoothed_predicted_positions = torch.tensor(
                            smoothed_combined_positions[np.newaxis, input_frames:, :],
                            dtype=torch.float32,
                            device=device,
                        )
                        predicted_bboxes, predicted_velocities = (
                            convert_PosSize_to_PosVel(
                                smoothed_predicted_positions, predicted_sizes
                            )
                        )
                    else:
                        predicted_bboxes, predicted_velocities = (
                            convert_PosSize_to_PosVel(
                                predicted_positions, predicted_sizes
                            )
                        )

                    past_bboxes = (
                        torch.tensor(
                            [[t[0], t[1], t[6], t[7]] for t in track_history_bbox],
                            dtype=torch.float32,
                        )
                        .unsqueeze(0)
                        .to(device)
                    )

                    predicted_future_positions = convert_velocity_to_positions(
                        predicted_velocities, past_bboxes
                    )

                    smoothed_combined_bboxes = np.concatenate(
                        (
                            past_bboxes.cpu().numpy(),
                            predicted_future_positions.cpu().numpy(),
                        ),
                        axis=1,
                    )

                    smoothing_functions = {
                        "sa": sg_filter_smoothing,
                        "ma": moving_average_smoothing,
                        "gaussian": gaussian_filter_smoothing,
                        "es": exponential_smoothing,
                        "mes": modified_exponential_smoothing,
                        "hybrid": hybrid_smoothing,
                        "adaptive": adaptive_smoothing,
                    }

                    if smooth_filter in smoothing_functions:
                        smoothed_combined_bboxes = smoothing_functions[smooth_filter](
                            smoothed_combined_bboxes, **smoothing_params
                        )
                    else:
                        print(
                            f"Warning: Unknown smoothing filter '{smooth_filter}'. No smoothing applied."
                        )

                    smoothed_combined_bboxes = np.squeeze(smoothed_combined_bboxes)

                    pred_metrics = {"iou": [], "de": [], "de_percent": []}

                    for i in range(future_frames):
                        future_frame_id = detection["frame_id"] + i + 1
                        x, y, w, h = smoothed_combined_bboxes[input_frames + i]
                        future_box_pred = [x, y, w, h]

                        future_gt = next(
                            (
                                d["current_bbox"]
                                for d in detections
                                if d["frame_id"] == future_frame_id
                            ),
                            None,
                        )

                        if not future_gt:
                            break

                        future_box_gt = [
                            future_gt["x_center"],
                            future_gt["y_center"],
                            future_gt["w"],
                            future_gt["h"],
                        ]

                        prediction_data.append(
                            {
                                "frame_id": future_frame_id,
                                "predicted_position": future_box_pred,
                                "ground_truth_position": future_box_gt,
                            }
                        )

                        pred_metrics["iou"].append(
                            calculate_iou(future_box_pred, future_box_gt)
                        )

                        pred_x_pixel, pred_y_pixel = int(x * frame_w), int(y * frame_h)
                        gt_center_x_pixel, gt_center_y_pixel = int(
                            future_box_gt[0] * frame_w
                        ), int(future_box_gt[1] * frame_h)
                        de = np.sqrt(
                            (pred_x_pixel - gt_center_x_pixel) ** 2
                            + (pred_y_pixel - gt_center_y_pixel) ** 2
                        )
                        pred_metrics["de"].append(de)
                        pred_metrics["de_percent"].append((de / frame_diagonal) * 100)

                    # Skip if there are not enough future frames
                    if len(pred_metrics["de"]) < future_frames:
                        continue

                    # Update overall metrics
                    metrics["ade"].append(np.mean(pred_metrics["de"]))
                    metrics["ade_percent"].append(np.mean(pred_metrics["de_percent"]))
                    metrics["aiou"].append(np.mean(pred_metrics["iou"]))
                    metrics["fde"].append(pred_metrics["de"][-1])
                    metrics["fde_percent"].append(pred_metrics["de_percent"][-1])
                    metrics["fiou"].append(pred_metrics["iou"][-1])

                    # Visualize only the last frame prediction
                    pred_center_x, pred_center_y, pred_w, pred_h = (
                        smoothed_combined_bboxes[-1]
                    )
                    future_box_pred = [
                        int((pred_center_x - pred_w / 2) * frame_w),
                        int((pred_center_y - pred_h / 2) * frame_h),
                        int((pred_center_x + pred_w / 2) * frame_w),
                        int((pred_center_y + pred_h / 2) * frame_h),
                    ]

                    gt_center_x, gt_center_y, gt_w, gt_h = future_box_gt
                    future_box_gt = [
                        int((gt_center_x - gt_w / 2) * frame_w),
                        int((gt_center_y - gt_h / 2) * frame_h),
                        int((gt_center_x + gt_w / 2) * frame_w),
                        int((gt_center_y + gt_h / 2) * frame_h),
                    ]

                    next_frame_id += input_frames + future_frames

                except Exception as e:
                    print(f"Error in GRU prediction: {e}")

        annotator.box_label(
            future_box_pred, "Predicted Future Position", color=(0, 255, 0)
        )

        current_box = [
            int((current_bbox["x_center"] - current_bbox["w"] / 2) * frame_w),
            int((current_bbox["y_center"] - current_bbox["h"] / 2) * frame_h),
            int((current_bbox["x_center"] + current_bbox["w"] / 2) * frame_w),
            int((current_bbox["y_center"] + current_bbox["h"] / 2) * frame_h),
        ]

        annotator.box_label(current_box, "Refueling Port", color=(0, 0, 255))
        annotator.box_label(future_box_gt, "Future Position GT", color=(255, 0, 0))
        annotated_frame = annotator.result()
        result.write(annotated_frame)

        cv2.imshow("frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_counter += 1

    cap.release()
    result.release()
    cv2.destroyAllWindows()

    mean_metrics = {k: np.mean(v) if v else 0 for k, v in metrics.items()}

    return prediction_data, mean_metrics


def save_predictions_to_json(prediction_data, output_json_path):
    # Convert prediction data to a more readable format
    formatted_data = []
    for pred in prediction_data:
        formatted_pred = {
            "frame_id": pred["frame_id"],
            "predicted_position": {
                "x_center": float(pred["predicted_position"][0]),
                "y_center": float(pred["predicted_position"][1]),
                "width": float(pred["predicted_position"][2]),
                "height": float(pred["predicted_position"][3]),
            },
        }
        if pred["ground_truth_position"]:
            formatted_pred["ground_truth_position"] = {
                "x_center": float(pred["ground_truth_position"][0]),
                "y_center": float(pred["ground_truth_position"][1]),
                "width": float(pred["ground_truth_position"][2]),
                "height": float(pred["ground_truth_position"][3]),
            }
        else:
            formatted_pred["ground_truth_position"] = None
        formatted_data.append(formatted_pred)

    with open(output_json_path, "w") as f:
        json.dump(formatted_data, f, indent=2)
    print(f"Saved predictions and ground truth to {output_json_path}")
