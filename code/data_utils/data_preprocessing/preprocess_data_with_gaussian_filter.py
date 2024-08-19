import json
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_smoothing(trajectory, sigma=2):
    """
    Apply Gaussian smoothing to a trajectory.

    Args:
        trajectory (numpy.ndarray): A 2D array with shape (n_frames, 4) representing the trajectory data
                                     with columns ["x_center", "y_center", "width", "height"].
        sigma (int, optional): The standard deviation for Gaussian kernel. Default is 2.

    Returns:
        numpy.ndarray: The smoothed trajectory as a 2D array with the same shape as the input.
    """
    # Apply Gaussian filter separately to each column (x_center, y_center, width, height)
    smoothed_x = gaussian_filter(trajectory[:, 0], sigma=sigma)
    smoothed_y = gaussian_filter(trajectory[:, 1], sigma=sigma)
    smoothed_width = gaussian_filter(trajectory[:, 2], sigma=sigma)
    smoothed_height = gaussian_filter(trajectory[:, 3], sigma=sigma)

    # Stack the smoothed components back into a single array
    return np.column_stack((smoothed_x, smoothed_y, smoothed_width, smoothed_height))


def handle_null_values(trajectory):
    """
    Handle missing values in the trajectory data by linear interpolation.

    Args:
        trajectory (numpy.ndarray): A 2D array with shape (n_frames, 4) representing the trajectory data
                                     with columns ["x_center", "y_center", "width", "height"].

    Returns:
        numpy.ndarray: The trajectory with missing values filled in by linear interpolation.
    """
    df = pd.DataFrame(trajectory, columns=["x_center", "y_center", "width", "height"])
    # Interpolate missing values
    df.interpolate(method="linear", limit_direction="both", inplace=True)
    return df.values


if __name__ == "__main__":

    stages = ["train", "val", "test"]  # Stages of the training process

    for stage in stages:
        # Load the data
        file_path = f"./data/AARP/frames/full_dataset_annotated_fpp/{stage}.json"
        with open(file_path) as f:
            data = json.load(f)

        # Extract the bounding box coordinates and apply smoothing
        for entry in data:
            bbox_trajectory = []
            for frame in entry["frames"]:
                bbox = frame.get("bbox", [])
                bbox_trajectory.append(bbox)

            # Handle null values by interpolation
            bbox_trajectory = handle_null_values(bbox_trajectory)

            # Convert to numpy array for processing
            bbox_trajectory_np = np.array(bbox_trajectory)

            # Apply temporal smoothing to bbox trajectory
            smoothed_trajectory = gaussian_smoothing(bbox_trajectory_np)

            # Clip the values to be within the range [0, 1]
            smoothed_trajectory = np.clip(smoothed_trajectory, 0, 1)

            # Update the bbox coordinates in the data
            for idx, frame in enumerate(entry["frames"]):
                frame["bbox"] = smoothed_trajectory[idx].tolist()

        # Save the updated data back to a JSON file
        output_file_path = f"./data/AARP/frames/full_dataset_annotated_fpp/{stage}_filter_gaussian.json"
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=4)
