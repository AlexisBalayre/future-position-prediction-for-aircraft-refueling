import json
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter


def rolling_mean(trajectory, window_size=20):
    """
    Apply rolling mean smoothing to a trajectory.

    Args:
        trajectory (numpy.ndarray): A 2D array with shape (n_frames, 4) representing the trajectory data
                                     with columns ["x_center", "y_center", "width", "height"].
        window_size (int, optional): The size of the moving window. Default is 20.

    Returns:
        numpy.ndarray: The smoothed trajectory as a 2D array with the same shape as the input.
    """
    df = pd.DataFrame(trajectory, columns=["x_center", "y_center", "width", "height"])
    rolling_df = df.rolling(window=window_size, min_periods=1, center=True).mean()
    return rolling_df.values


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
            smoothed_trajectory = rolling_mean(bbox_trajectory_np)

            # Clip the values to be within the range [0, 1]
            smoothed_trajectory = np.clip(smoothed_trajectory, 0, 1)

            # Update the bbox coordinates in the data
            for idx, frame in enumerate(entry["frames"]):
                frame["bbox"] = smoothed_trajectory[idx].tolist()

        # Save the updated data back to a JSON file
        output_file_path = f"./data/AARP/frames/full_dataset_annotated_fpp/{stage}_filter_rolling_mean.json"
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=4)
