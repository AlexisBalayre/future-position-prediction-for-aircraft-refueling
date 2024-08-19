import json
import os
import numpy as np
import cv2
from tqdm import tqdm


def preprocess_data_with_of(
    json_file, images_folder, output_folder, target_size=(480, 640)
):
    """
    Preprocess video data, compute optical flow, and save processed data.

    Args:
        json_file (str): Path to the JSON file containing video frame annotations.
        images_folder (str): Directory containing the video frames.
        output_folder (str): Directory where processed data and optical flows will be saved.
        target_size (tuple): Target size to resize images (height, width).

    Returns:
        None
    """
    # Load the annotated data from the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    processed_data = []

    original_height, original_width = 480, 640
    target_height, target_width = target_size
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # Create output directories if they do not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    optical_flow_folder = os.path.join(output_folder, "optical_flows")
    if not os.path.exists(optical_flow_folder):
        os.makedirs(optical_flow_folder)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    for entry in tqdm(data):
        video_id = entry["video_id"]
        frames = entry["frames"]
        processed_frames = []

        prev_gray = None
        prev_points = None

        prev_bbox = None
        prev_velx, prev_vely = 0, 0

        for idx, frame in enumerate(frames):
            image_name = frame["image_name"]
            bbox = frame.get("bbox", [])
            class_id = frame.get("class_id", None)

            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if class_id is None or not bbox:
                class_id = 3
                bbox = [0, 0, 0, 0]

            bbox = [0 if coord is None else coord for coord in bbox]

            x, y, w, h = bbox
            x = x * width_ratio
            y = y * height_ratio
            w = w * width_ratio
            h = h * height_ratio

            if idx == 0:
                # Initialize values for the first frame
                velx, vely, accx, accy = 0, 0, 0, 0
                deltaw, deltah = 0, 0
                optical_flow = np.zeros(
                    (target_height, target_width, 2), dtype=np.float32
                )
                prev_gray = gray
                prev_points = cv2.goodFeaturesToTrack(
                    gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
                )
            else:
                # Calculate Lucas-Kanade optical flow for subsequent frames
                if prev_points is not None and len(prev_points) > 0:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, prev_points, None, **lk_params
                    )

                    # Select good points
                    good_new = next_points[status == 1]
                    good_old = prev_points[status == 1]

                    # Create optical flow array
                    optical_flow = np.zeros(
                        (target_height, target_width, 2), dtype=np.float32
                    )
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        if 0 <= int(b) < target_height and 0 <= int(a) < target_width:
                            optical_flow[int(b), int(a)] = [a - c, b - d]

                    # Save optical flow to a file
                    optical_flow_file = os.path.join(
                        optical_flow_folder, f"{video_id}_{idx}.npy"
                    )
                    try:
                        np.save(optical_flow_file, optical_flow)
                    except Exception as e:
                        print(
                            f"Error saving optical flow file {optical_flow_file}: {e}"
                        )

                    prev_x, prev_y, prev_w, prev_h = prev_bbox
                    velx = x - prev_x
                    vely = y - prev_y
                    deltaw = w - prev_w
                    deltah = h - prev_h

                    accx = velx - prev_velx
                    accy = vely - prev_vely

                    prev_velx, prev_vely = velx, vely

                    # Update the previous frame and points
                    prev_gray = gray.copy()
                    prev_points = good_new.reshape(-1, 1, 2)
                else:
                    # Handle the case where no good points are found
                    optical_flow = np.zeros(
                        (target_height, target_width, 2), dtype=np.float32
                    )
                    velx, vely, accx, accy = 0, 0, 0, 0
                    deltaw, deltah = 0, 0
                    prev_gray = gray.copy()
                    prev_points = cv2.goodFeaturesToTrack(
                        gray,
                        maxCorners=100,
                        qualityLevel=0.3,
                        minDistance=7,
                        blockSize=7,
                    )

            bbox_position = [x, y, velx, vely, accx, accy]
            bbox_size = [w, h, deltaw, deltah]
            prev_bbox = [x, y, w, h]

            processed_frames.append(
                {
                    "image_name": image_name,
                    "bbox_position": bbox_position,
                    "bbox_size": bbox_size,
                    "optical_flow_file": f"{video_id}_{idx}.npy" if idx > 0 else None,
                    "class_id": class_id,
                }
            )

        processed_data.append(
            {
                "video_id": video_id,
                "frames": processed_frames,
            }
        )

    # Save the processed data to a JSON file
    output_file = os.path.join(output_folder, "processed_data.json")
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)


if __name__ == "__main__":
    # Example usage for each split
    splits = ["train", "val", "test"]
    images_folder = "/path/to/images/folder"

    for split in splits:
        json_file = f"/path/to/json/files/{split}.json"
        output_folder = f"/path/to/output/folder/{split}"
        preprocess_data_with_of(json_file, images_folder, output_folder)
