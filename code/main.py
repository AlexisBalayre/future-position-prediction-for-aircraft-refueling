import json
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import deque
from scipy.signal import savgol_filter
from statsmodels.tsa.arima.model import ARIMA

from future_position_prediction.GRU.SizPos.GRULightningModelConcat import (
    GRULightningModelConcat,
)
from future_position_prediction.GRU.SizPos.utils import (
    convert_velocity_to_positions,
    convert_PosSize_to_PosVel,
)


class AdvancedKalmanFilter:
    def __init__(
        self, initial_state, initial_covariance, process_noise, measurement_noise
    ):
        self.state = np.array(initial_state, dtype=np.float32)
        self.covariance = np.array(initial_covariance, dtype=np.float32)
        self.process_noise = np.array(process_noise, dtype=np.float32)
        self.measurement_noise = np.array(measurement_noise, dtype=np.float32)
        self.previous_measurement = None

    def predict(self):
        # No control input, so prediction is just maintaining the state
        self.state = self.state
        self.covariance = self.covariance + self.process_noise

    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float32)
        if self.previous_measurement is not None:
            residual = measurement - self.previous_measurement
            adaptive_measurement_noise = np.var(residual)
            R = np.diag([adaptive_measurement_noise] * len(measurement))
        else:
            R = self.measurement_noise

        S = self.covariance + R  # Innovation covariance
        K = np.dot(self.covariance, np.linalg.inv(S))  # Kalman gain

        self.state = self.state + np.dot(K, (measurement - self.state))
        I = np.eye(self.covariance.shape[0])
        self.covariance = np.dot((I - K), self.covariance)

        self.previous_measurement = measurement

    def get_state(self):
        return self.state  # Return the current state of the filter


def apply_advanced_kalman_filter(predicted_bboxes):
    # Ensure predicted_bboxes is at least 2D (seq_len, features)
    predicted_bboxes = np.atleast_2d(predicted_bboxes)

    # Initialize the filter parameters with the first predicted bbox
    initial_state = predicted_bboxes[0]
    state_dim = len(initial_state)
    initial_covariance = np.eye(state_dim) * 0.1
    process_noise = np.eye(state_dim) * 1e-5
    measurement_noise = np.eye(state_dim) * 0.1

    akf = AdvancedKalmanFilter(
        initial_state, initial_covariance, process_noise, measurement_noise
    )

    smoothed_bboxes = []
    for i, bbox in enumerate(predicted_bboxes):
        # Ensure bbox is a 1D array
        bbox = np.squeeze(bbox)

        if bbox.ndim != 1 or len(bbox) != state_dim:
            raise ValueError(
                f"Unexpected bbox shape or size: {bbox.shape}, expected: ({state_dim},)"
            )

        try:
            akf.predict()
            akf.update(bbox)
        except Exception as e:
            print(f"Error during Kalman Filter update at bbox {i}: {e}")
            break

        smoothed_bboxes.append(akf.get_state())

    print("Reached the end of Kalman Filter processing.")

    return np.array(smoothed_bboxes)


def smooth_trajectory(trajectory, window_length=5, polyorder=2):
    if len(trajectory) < window_length:
        return trajectory
    return savgol_filter(trajectory, window_length, polyorder, axis=0)


def moving_average_smoothing(trajectory, window_size=5):
    if len(trajectory) < window_size:
        return trajectory
    return np.convolve(trajectory, np.ones(window_size) / window_size, mode="valid")


def exponential_smoothing(trajectory, alpha=0.1):
    smoothed_trajectory = np.zeros_like(trajectory)
    smoothed_trajectory[0] = trajectory[0]
    for t in range(1, len(trajectory)):
        smoothed_trajectory[t] = (
            alpha * trajectory[t] + (1 - alpha) * smoothed_trajectory[t - 1]
        )
    return smoothed_trajectory


def arima_smoothing(trajectory, order=(1, 1, 1)):
    model = ARIMA(trajectory, order=order)
    fitted_model = model.fit()
    smoothed_trajectory = fitted_model.predict(start=0, end=len(trajectory) - 1)
    return smoothed_trajectory


def run_detections(input_video_path, output_json_path, yolo_weights_path):
    model = YOLO(yolo_weights_path)
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

    if detections:
        positions = np.array([[d["x_center"], d["y_center"]] for d in detections])
        sizes = np.array([[d["w"], d["h"]] for d in detections])

        smoothed_positions = smooth_trajectory(positions, window_length=10)
        smoothed_sizes = smooth_trajectory(sizes)

        for i, detection in enumerate(detections):
            detection["x_center"], detection["y_center"] = smoothed_positions[i]
            detection["w"], detection["h"] = smoothed_sizes[i]

    output_detections = []
    for i, detection in enumerate(detections):
        future_frame_index = min(i + 59, len(detections) - 1)
        future_bbox_gt = detections[future_frame_index]

        output_detections.append(
            {
                "frame_id": detection["frame_id"],
                "current_bbox": {
                    k: detection[k] for k in ["x_center", "y_center", "w", "h"]
                },
                "future_bbox_gt": {
                    k: future_bbox_gt[k] for k in ["x_center", "y_center", "w", "h"]
                },
            }
        )

    output_detections.sort(key=lambda x: x["frame_id"])

    with open(output_json_path, "w") as f:
        json.dump(output_detections, f)

    print(f"Saved detections to {output_json_path}")


def load_detections(input_json_path):
    with open(input_json_path, "r") as f:
        return json.load(f)


def run_future_positions_pred(
    detections,
    gru_model_path,
    hparams_file,
    input_video_path,
    output_video_path,
    input_frames=30,
    future_frames=60,
    smooth_predictions=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gru_model = GRULightningModelConcat.load_from_checkpoint(
        gru_model_path, hparms_file=hparams_file
    )
    gru_model = gru_model.to(device)
    gru_model.eval()

    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = [
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    ]

    result = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h)
    )

    all_history_bbox = []
    track_history_bbox = deque(maxlen=input_frames)
    frame_counter = 0

    prediction_data = []  # List to store prediction and ground truth data

    future_box_pred = [0, 0, 0, 0]  # Initialize future box prediction
    future_box_gt = [0, 0, 0, 0]  # Initialize future box ground truth

    for detection in detections:
        success, frame = cap.read()
        if not success:
            break

        current_bbox = detection["current_bbox"]
        annotator = Annotator(frame, line_width=2)

        if len(track_history_bbox) == input_frames:
            if frame_counter % 90 == 30:  # After 30 frames
                try:
                    future_bbox_gt = detection["future_bbox_gt"]

                    smoothed_history = np.array(track_history_bbox)
                    smoothed_positions = smoothed_history[:, :6]
                    smoothed_sizes = smoothed_history[:, 6:]

                    # Ensure that the positions and sizes are converted to NumPy arrays first
                    positions = torch.tensor(
                        np.array([smoothed_positions]), dtype=torch.float32
                    ).to(device)
                    sizes = torch.tensor(
                        np.array([smoothed_sizes]), dtype=torch.float32
                    ).to(device)

                    with torch.no_grad():
                        predicted_positions, predicted_sizes = gru_model(
                            positions, sizes
                        )

                    if smooth_predictions:
                        combined_positions = np.concatenate(
                            (smoothed_positions, predicted_positions.cpu().numpy()[0])
                        )
                        smoothed_combined_positions = moving_average_smoothing(
                            combined_positions
                        )
                        # torch.Size([1, 90, 4]) -> torch.Size([1, 60, 4])
                        smoothed_predicted_positions = torch.tensor(
                            smoothed_combined_positions[np.newaxis, input_frames:, :],
                            dtype=torch.float32,
                            device=device,
                        )

                        predicted_bboxes, predicted_velocities = (
                            convert_PosSize_to_PosVel(
                                smoothed_predicted_positions,
                                predicted_sizes,
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

                    # Combine the smoothed future positions
                    smoothed_combined_bboxes = np.concatenate(
                        (
                            past_bboxes.cpu().numpy(),
                            predicted_future_positions.cpu().numpy(),
                        ),
                        axis=1,
                    )
                    # (1, 90, 4) -> (90, 4)
                    """smoothed_combined_bboxes = smooth_trajectory(
                        smoothed_combined_bboxes, window_length=30
                    ) """
                    """ smoothed_combined_bboxes = moving_average_smoothing(
                        smoothed_combined_bboxes, window_size=30
                    )  """
                    smoothed_combined_bboxes = exponential_smoothing(
                        smoothed_combined_bboxes, alpha=0.1
                    )
                    smoothed_combined_bboxes = np.squeeze(smoothed_combined_bboxes)

                    # Store prediction and ground truth data for all future frames
                    for i in range(future_frames):
                        future_frame_id = detection["frame_id"] + i + 1
                        x, y, w, h = smoothed_combined_bboxes[input_frames + i]
                        future_box_pred = [x, y, w, h]  # Already in YOLO format

                        # Find the corresponding ground truth for this future frame
                        future_gt = next(
                            (
                                d["current_bbox"]
                                for d in detections
                                if d["frame_id"] == future_frame_id
                            ),
                            None,
                        )
                        if future_gt:
                            future_box_gt = [
                                future_gt["x_center"],
                                future_gt["y_center"],
                                future_gt["w"],
                                future_gt["h"],
                            ]
                        else:
                            future_box_gt = None

                        prediction_data.append(
                            {
                                "frame_id": future_frame_id,
                                "predicted_position": future_box_pred,
                                "ground_truth_position": future_box_gt,
                            }
                        )

                    # Visualize only the 60th frame prediction
                    x, y, w, h = smoothed_combined_bboxes[-1]
                    x_pixel, y_pixel = int(x * frame.shape[1]), int(y * frame.shape[0])
                    w_pixel, h_pixel = int(w * frame.shape[1]), int(h * frame.shape[0])
                    future_box_pred = [
                        x_pixel - w_pixel // 2,
                        y_pixel - h_pixel // 2,
                        x_pixel + w_pixel // 2,
                        y_pixel + h_pixel // 2,
                    ]

                    future_box_gt = [
                        int(
                            (future_bbox_gt["x_center"] - future_bbox_gt["w"] / 2)
                            * frame.shape[1]
                        ),
                        int(
                            (future_bbox_gt["y_center"] - future_bbox_gt["h"] / 2)
                            * frame.shape[0]
                        ),
                        int(
                            (future_bbox_gt["x_center"] + future_bbox_gt["w"] / 2)
                            * frame.shape[1]
                        ),
                        int(
                            (future_bbox_gt["y_center"] + future_bbox_gt["h"] / 2)
                            * frame.shape[0]
                        ),
                    ]

                except Exception as e:
                    print(f"Error in GRU prediction: {e}")

        annotator.box_label(
            future_box_pred, "Predicted Future Position", color=(0, 255, 0)
        )

        current_box = [
            int((current_bbox["x_center"] - current_bbox["w"] / 2) * frame.shape[1]),
            int((current_bbox["y_center"] - current_bbox["h"] / 2) * frame.shape[0]),
            int((current_bbox["x_center"] + current_bbox["w"] / 2) * frame.shape[1]),
            int((current_bbox["y_center"] + current_bbox["h"] / 2) * frame.shape[0]),
        ]
        annotator.box_label(current_box, "Refueling Port", color=(0, 0, 255))
        annotator.box_label(future_box_gt, "Future Position GT", color=(255, 0, 0))
        annotated_frame = annotator.result()
        result.write(annotated_frame)

        cv2.imshow("frame", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if track_history_bbox:
            prev_bbox = track_history_bbox[-1]
            velx = current_bbox["x_center"] - prev_bbox[0]
            vely = current_bbox["y_center"] - prev_bbox[1]
            deltaw = current_bbox["w"] - prev_bbox[6]
            deltah = current_bbox["h"] - prev_bbox[7]

            if len(track_history_bbox) > 1:
                prev_velx, prev_vely = prev_bbox[2:4]
                accx, accy = velx - prev_velx, vely - prev_vely
            else:
                accx, accy = 0, 0
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

        frame_counter += 1

    cap.release()
    result.release()
    cv2.destroyAllWindows()

    print("Video processing completed.")
    return prediction_data


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
    gru_model_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/SizPos/version_121/checkpoints/epoch=79-step=6800.ckpt"
    hparams_file = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/SizPos/version_121/hparams.yaml"
    input_frames = 30
    output_frames = 60
    input_video_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/videos/video_lab_semiopen/video_lab_semiopen_1______3.avi"
    video_name = "video_lab_semiopen_1______3"
    is_smooth_predictions = True
    output_json_file = f"./data_{video_name}.json"
    output_video_path = (
        f"./output_smooth_ema{video_name}.avi"
        if is_smooth_predictions
        else f"./output_{video_name}.avi"
    )
    output_predictions_json = (
        f"./predictions_smooth_ema{video_name}.json"
        if is_smooth_predictions
        else f"./predictions_{video_name}.json"
    )

    # Step 1: Perform object detection and save detections with future bbox GT
    # run_detections(input_video_path, output_json_file, yolo_weights_path)

    # Step 2: Load detections
    detections = load_detections(output_json_file)

    # Step 3: Perform future position prediction and visualize
    prediction_data = run_future_positions_pred(
        detections,
        gru_model_path,
        hparams_file,
        input_video_path,
        output_video_path,
        smooth_predictions=False,
    )

    # Step 4: Save predictions and ground truth to JSON
    save_predictions_to_json(prediction_data, output_predictions_json)
