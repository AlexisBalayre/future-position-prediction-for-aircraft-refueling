import time
import cv2
import torch
import psutil
import numpy as np
import json
from ultralytics import YOLOv10
from ultralytics.utils.plotting import Annotator, colors

# Load your fine-tuned YOLO model
model = YOLOv10(
    "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/runs/detect/train13/weights/best_yolov10n.pt"
)


def detect_and_track(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_time = 0
    power_usage = []
    video_results = {"video_id": video_path, "frames": []}

    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a video writer to save the annotated video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"{video_path.split('.')[0]}_annotated.mp4", fourcc, fps, (width, height)
    )

    names = model.names

    prev_frame = None  # Initialize the previous frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        # Resize the frame if the previous frame is not None and has different size
        if prev_frame is not None and (
            prev_frame.shape[1] != frame.shape[1]
            or prev_frame.shape[0] != frame.shape[0]
        ):
            frame = cv2.resize(frame, (prev_frame.shape[1], prev_frame.shape[0]))

        # Perform detection on the frame
        results = model.track(frame, persist=True, verbose=False)

        end_time = time.time()
        detection_time = end_time - start_time
        total_time += detection_time

        # Measure power usage of the process
        process = psutil.Process()
        power = process.cpu_percent(interval=None)
        power_usage.append(power)

        # Collect results for each frame
        frame_results = {
            "frame_id": frame_count-1,
            "image_name": f"{video_path.split('.')[0]}_frame_{frame_count-1}.jpg",
            "detections": [],
        }

        if results and results[0].boxes is not None:
            boxes_xyxyn = results[0].boxes.xywhn.cpu().numpy()
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

            if results[0].boxes.id is not None:
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotated_frame = frame.copy()
                annotator = Annotator(annotated_frame, line_width=2)

                for box_xyxyn, cls, track_id in zip(boxes_xyxyn, clss, track_ids):
                    box_xyxy = boxes_xyxy[track_ids.index(track_id)]
                    annotator.box_label(
                        box_xyxy, color=colors(int(cls), True), label=names[int(cls)]
                    )
                    frame_results["detections"].append(
                        {"class_id": int(cls), "bbox": box_xyxyn.tolist()}
                    )

                out.write(annotated_frame)

        video_results["frames"].append(frame_results)

        # Update the previous frame
        prev_frame = frame.copy()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_detection_time = total_time / frame_count
    avg_power_usage = np.mean(power_usage)

    return video_results, avg_detection_time, avg_power_usage


if __name__ == "__main__":
    # Test videos
    video_paths = [
        #"/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/videos/test/test_outdoor1.mp4",
        "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/videos/video_lab_closed_1/video_lab_platform_6.avi",
        "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/videos/test/test_indoor1.avi",
        "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/videos/video_lab_semiopen/video_lab_semiopen_1______3.avi",
    ]

    all_results = []

    for video_path in video_paths:
        print(f"Processing {video_path}")
        video_results, avg_detection_time, avg_power_usage = detect_and_track(
            video_path
        )
        all_results.append(video_results)
        print(f"Average Detection Time per Frame: {avg_detection_time:.4f} seconds")
        print(f"Average Power Usage: {avg_power_usage:.2f}% CPU")

    # Save the results to a JSON file
    with open(
        "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/evalutation/results_nano.json",
        "w",
    ) as f:
        json.dump(all_results, f, indent=4)
