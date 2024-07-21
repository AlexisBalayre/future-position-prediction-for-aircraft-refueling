import json
import os
import numpy as np
import cv2
from tqdm import tqdm

def preprocess_data(json_file, images_folder, output_folder, target_size=(480, 640)):
    with open(json_file, "r") as f:
        data = json.load(f)

    processed_data = []

    original_height, original_width = 480, 640
    target_height, target_width = target_size
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    optical_flow_folder = os.path.join(output_folder, "optical_flows")
    if not os.path.exists(optical_flow_folder):
        os.makedirs(optical_flow_folder)

    for entry in tqdm(data):
        video_id = entry["video_id"]
        frames = entry["frames"]
        processed_frames = []

        for idx, frame in enumerate(frames):
            image_name = frame["image_name"]
            bbox = frame.get("bbox", [])
            class_id = frame.get("class_id", None)

            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if class_id is None or bbox == []:
                class_id = 3
                bbox = [0, 0, 0, 0]

            bbox = [0 if coord is None else coord for coord in bbox]

            x, y, w, h = bbox
            x = x * width_ratio
            y = y * height_ratio
            w = w * width_ratio
            h = h * height_ratio
            bbox_position = [x, y, w, h]

            if idx == 0:
                bbox_velocity = [0, 0, 0, 0]
                optical_flow = np.zeros((target_height, target_width, 2), dtype=np.float32)
            else:
                prev_frame = frames[idx - 1]["image_name"]
                prev_image_path = os.path.join(images_folder, prev_frame)
                prev_image = cv2.imread(prev_image_path)
                prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

                optical_flow = cv2.calcOpticalFlowFarneback(
                    prev_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                optical_flow_file = os.path.join(optical_flow_folder, f"{video_id}_{idx}.npy")
                np.save(optical_flow_file, optical_flow)

                prev_bbox = frames[idx - 1].get("bbox", [])
                prev_bbox = [0 if coord is None or np.isnan(float(coord)) else coord for coord in prev_bbox]
                prev_x, prev_y, prev_w, prev_h = prev_bbox
                prev_x = prev_x * width_ratio
                prev_y = prev_y * height_ratio
                prev_w = prev_w * width_ratio
                prev_h = prev_h * height_ratio
                bbox_velocity = [x - prev_x, y - prev_y, w - prev_w, h - prev_h]

            processed_frames.append({
                "image_name": image_name,
                "bbox_position": bbox_position,
                "bbox_velocity": bbox_velocity,
                "optical_flow_file": f"{video_id}_{idx}.npy" if idx > 0 else None,
                "class_id": class_id,
            })

        processed_data.append({
            "video_id": video_id,
            "frames": processed_frames,
        })

    output_file = os.path.join(output_folder, "processed_data.json")
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)

# Example usage for each split
splits = ["train", "val", "test"]
images_folder = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_YOLO/all/images"

for split in splits:
    json_file = f"/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/{split}.json"
    output_folder = f"/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/processed_data/{split}"
    preprocess_data(json_file, images_folder, output_folder)