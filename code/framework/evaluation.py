import json
import torch

from .object_detection_model import load_detections, run_detections
from .sequence_model import (
    run_future_positions_pred,
    save_predictions_to_json,
)


# Script to run object detection and future position prediction (Evaluation Purpose)
if __name__ == "__main__":
    # Set up your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained GRU model
    yolo_weights_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/runs/detect/train24/weights/best.pt"  # Path to the YOLOv10 weights
    gru_model_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/SizPos/checkpoints/input15_output30/checkpoints/epoch=57-step=5452.ckpt"  # Path to the trained GRU model
    hparams_file = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/GRU/SizPos/checkpoints/input15_output30/hparams.yaml"  # Path to the hyperparameters file
    input_frames = 15  # Number of input frames
    output_frames = 30  # Number of output frames
    input_video_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/AARP/videos/video_lab_semiopen/video_lab_semiopen_1______3.avi"  # Path to the video
    video_name = "video_lab_semiopen_1______3"  # Name of the video

    # DO NOT MODIFY BELOW THIS LINE
    smooth_filters = ["sa", "ma", "mes", "es", "hybrid", "adaptive", "gaussian", ""] 
    smoothing_params = {
        "sa": {"window_length": 20},
        "ma": {"window_size": 2},
        "mes": {},
        "es": {},
        "hybrid": {"savgol_window_length": 20},
        "adaptive": {"initial_window": 10},
        "gaussian": {"sigma": 2},
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
                smooth_filter=False,
            )

            # Step 2: Load detections
            detections = load_detections(output_json_file)

            # Step 3: Perform future position prediction and visualise
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


# Display the output batch
print(output_batch)
