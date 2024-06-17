import torch
import cv2
import numpy as np
from ultralytics import YOLOv10
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

from future_position_prediction.LSTM.LSTMLightningModel import LSTMLightningModel

# Set up your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained LSTM model
lstm_model_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/LSTM/tb_logs/lstm_model_test1/version_5/checkpoints/epoch=2943-step=1254144.ckpt"
lstm_model = LSTMLightningModel.load_from_checkpoint(lstm_model_path)
lstm_model = lstm_model.to(device)
lstm_model.eval()

# LSTM input and output settings
input_frames = 3
output_frames = 1

# Set up the object detection model
track_history = defaultdict(lambda: [])
future_predictions = defaultdict(lambda: [])
model = YOLOv10(
    "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/runs/detect/train2_final/weights/best.pt"
)
names = model.model.names

# Open the video file
video_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/test/video_lab_semiopen_1______3.avi"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

result = cv2.VideoWriter(
    "video_lab_semiopen_1______3_with_predictions.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=False)
        boxes_xyxy = results[0].boxes.xyxy.cpu()
        boxes_xyxyn = results[0].boxes.xyxyn.cpu()

        if results[0].boxes.id is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Annotator Init
            annotator = Annotator(frame, line_width=2)

            for box_xyxyn, cls, track_id in zip(boxes_xyxyn, clss, track_ids):
                box_xyxy = boxes_xyxy[track_ids.index(track_id)]
                annotator.box_label(
                    box_xyxy, color=colors(int(cls), True), label=names[int(cls)]
                )

                bbox = box_xyxyn.tolist()

                # Store tracking history
                track_history[track_id].append(bbox)
                if len(track_history[track_id]) > input_frames:
                    track_history[track_id].pop(0)

            # Make predictions with LSTM model
            for track_id, history in track_history.items():
                if len(history) == input_frames:
                    input_sequence = (
                        torch.tensor(history[-input_frames:], dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )
                    with torch.no_grad():
                        prediction = lstm_model(input_sequence).cpu().numpy().squeeze()

                    pred_x1, pred_y1, pred_x2, pred_y2 = prediction

                    # Adjust the predicted bounding box coordinates
                    pred_x1 = max(0, pred_x1)
                    pred_y1 = max(0, pred_y1)
                    pred_x2 = min(1, pred_x2)
                    pred_y2 = min(1, pred_y2)

                    # Store the prediction for the future frame
                    future_predictions[track_id].append(
                        (
                            frame_count + output_frames,
                            [pred_x1, pred_y1, pred_x2, pred_y2],
                        )
                    )

            # Draw the predicted bounding box for the current frame
            for track_id, predictions in future_predictions.items():
                for prediction in predictions:
                    if prediction[0] == frame_count:
                        pred_x1, pred_y1, pred_x2, pred_y2 = prediction[1]
                        if pred_x2 > pred_x1 and pred_y2 > pred_y1:
                            annotator.box_label(
                                [pred_x1 * w, pred_y1 * h, pred_x2 * w, pred_y2 * h],
                                color=(0, 255, 0),
                                label="Future Position",
                            )
                        predictions.remove(prediction)

        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
    else:
        break

result.release()
cap.release()
cv2.destroyAllWindows()
