import json
import torch
import cv2
import numpy as np
import random
from ultralytics import YOLOv10
from ultralytics.utils.plotting import Annotator
from collections import deque
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter

from future_position_prediction.GRU.SizPos.GRULightningModelConcat import (
    GRULightningModelConcat,
)
from future_position_prediction.GRU.SizPos.utils import (
    convert_velocity_to_positions,
    convert_PosSize_to_PosVel,
)

from filters import (
    smooth_trajectory,
    moving_average_smoothing,
    exponential_smoothing,
    adaptive_smoothing,
    hybrid_smoothing,
    kalman_filter_smoothing,
    modified_exponential_smoothing,
)


def run_detections(
    input_video_path,
    output_json_path,
    yolo_weights_path,
    output_frame=30,
    smooth_filter="",
):
    model = YOLOv10(yolo_weights_path)
    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), "Error reading video file"

    frame_count = 0
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
                    "x_center": 0,
                    "y_center": 0,
                    "w": 0,
                    "h": 0,
                }
        else:
            current_detection = {
                "frame_id": frame_count,
                "x_center": 0,
                "y_center": 0,
                "w": 0,
                "h": 0,
            }
        detections.append(current_detection)

    cap.release()

    # Convert detections to numpy array for smoothing
    detection_array = np.array(
        [[d["x_center"], d["y_center"], d["w"], d["h"]] for d in detections]
    )

    # Apply smoothing
    smoothing_functions = {
        "sa": smooth_trajectory,
        "ma": moving_average_smoothing,
        "es": exponential_smoothing,
        "hybrid": hybrid_smoothing,
        "adaptive": adaptive_smoothing,
    }

    if smooth_filter in smoothing_functions:
        smoothed_detections = smoothing_functions[smooth_filter](
            detection_array[np.newaxis, ...]
        )[0]
    else:
        print(
            f"Warning: Unknown smoothing filter '{smooth_filter}'. No smoothing applied."
        )
        smoothed_detections = detection_array

    output_detections = []
    for i, detection in enumerate(smoothed_detections):
        future_frame_index = i + output_frame  # Look 'output_frame' frames ahead

        if future_frame_index < len(smoothed_detections):
            future_bbox_gt = smoothed_detections[future_frame_index]
        else:
            future_bbox_gt = np.array([0, 0, 0, 0])

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

    output_detections.sort(key=lambda x: x["frame_id"])

    with open(output_json_path, "w") as f:
        json.dump(output_detections, f, indent=2)

    print(f"Saved smoothed detections to {output_json_path}")


def load_detections(input_json_path):
    with open(input_json_path, "r") as f:
        return json.load(f)


def calculate_iou(boxA, boxB):
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


def set_fixed_seed(seed=42):
    """Set fixed seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_future_positions_pred(
    detections,
    gru_model_path,
    hparams_file,
    input_video_path,
    output_video_path,
    input_frames=30,
    future_frames=60,
    smooth_filter="sa",
    lkf=False,
    smoothing_params=None,
):
    # Set Seed
    set_fixed_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gru_model = (
        GRULightningModelConcat.load_from_checkpoint(
            gru_model_path, hparams_file=hparams_file
        )
        .to(device)
        .eval()
    )

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
                        "sa": smooth_trajectory,
                        "ma": moving_average_smoothing,
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


if __name__ == "__main__":
    # Set up your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained GRU model
    yolo_weights_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/runs/detect/train15/weights/best_yolov10s.pt"
    gru_model_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/SizPos/version_153/checkpoints/epoch=58-step=5546.ckpt"
    hparams_file = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/SizPos/version_153/hparams.yaml"
    input_frames = 15
    output_frames = 30
    input_video_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/videos/test/test_indoor1.avi"
    video_name = "test_indoor1"

    smooth_filters = ["sa", "ma", "mes", "es", "hybrid", "adaptive", ""]
    smoothing_params = {
        "sa": {"window_length": 15},
        "ma": {"window_size": 5},
        "mes": {},
        "es": {},
        "hybrid": {"savgol_window_length": 15},
        "adaptive": {"initial_window": 10},
    }
    lkfs = [True, False]

    output_batch = []

    for lkf in lkfs:
        for smooth_filter in smooth_filters:
            print(
                f"Running prediction with smooth filter: {smooth_filter} and LKF: {lkf}"
            )
            output_json_file = f"./test/data_{video_name}.json"
            output_video_path = (
                f"./test/output_lkf_{smooth_filter}_{video_name}.avi"
                if lkf
                else f"./test/output_{smooth_filter}_{video_name}.avi"
            )
            output_predictions_json = (
                f"./test/predictions_lkf_{smooth_filter}_{video_name}.json"
                if lkf
                else f"./test/predictions_{smooth_filter}_{video_name}.json"
            )

            # Step 1: Perform object detection and save detections with future bbox GT
            run_detections(
                input_video_path,
                output_json_file,
                yolo_weights_path,
                output_frame=output_frames,
                smooth_filter="sa"
            ) 

            # Step 2: Load detections
            detections = load_detections(output_json_file)

            # Step 3: Perform future position prediction and visualize
            prediction_data, metrics = run_future_positions_pred(
                detections,
                gru_model_path,
                hparams_file,
                input_video_path,
                output_video_path,
                input_frames=input_frames,
                future_frames=output_frames,
                smooth_filter=smooth_filter,
                lkf=lkf,
                smoothing_params=smoothing_params.get(smooth_filter, {}),
            )

            output_batch.append(
                {
                    "smooth_filter": smooth_filter,
                    "lkf": lkf,
                    "mean_fde": metrics["fde"],
                    "mean_fiou": metrics["fiou"],
                    "mean_ade": metrics["ade"],
                    "mean_aiou": metrics["aiou"],
                    "mean_fde_percent": metrics["fde_percent"],
                    "mean_ade_percent": metrics["ade_percent"],
                }
            )

            # Step 4: Save predictions and ground truth to JSON
            save_predictions_to_json(prediction_data, output_predictions_json)


print(output_batch)
