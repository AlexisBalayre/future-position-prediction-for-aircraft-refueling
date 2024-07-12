import torch
import cv2
import numpy as np
from ultralytics import YOLOv10
from ultralytics.utils.plotting import Annotator, colors
from collections import deque

from future_position_prediction.LSTM.Version3.LSTMLightningModel import LSTMLightningModel

# Set up your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained LSTM model
lstm_model_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/future_position_prediction/LSTM/Version3/tb_logs/lstm_fpp_version3/version_19/checkpoints/epoch=0-step=175.ckpt"
lstm_model = LSTMLightningModel.load_from_checkpoint(lstm_model_path)
lstm_model = lstm_model.to(device)
lstm_model.eval()

# LSTM input and output settings
input_frames = 15
output_frames = 60

# Load the YOLO model
model = YOLOv10(
    "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/runs/detect/train15/weights/best_yolov10s.pt"
)
names = model.model.names

# Video input setup
video_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/code/object_detection/YOLOv10/test/video_lab_semiopen_1______3.avi"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

result = cv2.VideoWriter("video_lab_semiopen_1___object_tracking.avi",
                       cv2.VideoWriter_fourcc(*'XVID'),
                       fps,
                       (w, h))

# Tracking and prediction setup
track_history = deque(maxlen=input_frames)

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Perform object detection
    results = model.track(frame, verbose=False, persist=True)

    # Process detections
    annotator = Annotator(frame, line_width=2)
    if results and len(results) > 0:
        boxes = results[0].boxes
        if len(boxes) > 0:
            box = boxes[0]  # Get the first (and only) box
            b = box.xyxy[0].cpu().numpy()
            c = int(box.cls)
            id = int(box.id.item()) if box.id is not None else None
            
            if id is not None:
                x_center = (b[0] + b[2]) / 2
                y_center = (b[1] + b[3]) / 2
                w = b[2] - b[0]
                h = b[3] - b[1]

                # Normalize the bounding box coordinates
                x_center /= frame.shape[1]
                y_center /= frame.shape[0]
                w /= frame.shape[1]
                h /= frame.shape[0]
                
                # Update track history
                track_history.append((x_center, y_center, w, h))
                
                # Draw current bounding box
                color = colors(c, True)
                annotator.box_label(b, f"{names[c]} {id}", color=color)
                
                # Perform future position prediction if we have enough history
                if len(track_history) == input_frames:
                    input_sequence = torch.tensor(list(track_history), dtype=torch.float32).unsqueeze(0).to(device)
                    
                    try:
                        with torch.no_grad():
                            predicted_positions, _ = lstm_model(input_sequence, input_sequence)
                        
                        print(f"Predicted positions: {predicted_positions}")
                        
                        # Draw the last predicted position
                        x, y, w, h = predicted_positions[0, -1].cpu().numpy()
                        # Green color for future position
                        annotator.box_label([x - w / 2, y - h / 2, x + w / 2, y + h / 2], "Future Position", color=(0, 255, 0))
                    except Exception as e:
                        print(f"Error in LSTM prediction: {e}")

    annotated_frame = annotator.result()
    result.write(annotated_frame)

    # Display the frame
    cv2.imshow("frame", annotated_frame)

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
result.release()
cv2.destroyAllWindows()

print("Video processing completed.")