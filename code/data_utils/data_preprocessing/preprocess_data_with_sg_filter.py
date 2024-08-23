import json
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def smooth_trajectory(trajectory, window_length=20, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth the trajectory data.

    Args:
        trajectory (numpy.ndarray): A 2D array with shape (n_frames, 4) representing the trajectory data
                                     with columns ["x_center", "y_center", "width", "height"].
        window_length (int, optional): The length of the filter window (i.e., the number of coefficients). Default is 20.
        polyorder (int, optional): The order of the polynomial used to fit the samples. Default is 2.

    Returns:
        numpy.ndarray: The smoothed trajectory as a 2D array with the same shape as the input.
                       If the input trajectory has fewer points than the window length, the original trajectory is returned.
    """
    if len(trajectory) < window_length:
        return trajectory  # Return original if not enough data points
    return savgol_filter(trajectory, window_length, polyorder, axis=0)


def handle_null_values(trajectory):
    """
    Handle missing values in the trajectory data by linear interpolation.

    Args:
        trajectory (numpy.ndarray): A 2D array with shape (n_frames, 4) representing the trajectory data
                                     with columns ["x_center", "y_center", "width", "height"].

    Returns:
        numpy.ndarray: The trajectory with missing values filled in by linear interpolation.
    """
    # Convert to DataFrame for easier manipulation
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
            smoothed_trajectory = smooth_trajectory(bbox_trajectory_np)

            # Clip the values to be within the range [0, 1]
            smoothed_trajectory = np.clip(smoothed_trajectory, 0, 1)

            # Update the bbox coordinates in the data
            for idx, frame in enumerate(entry["frames"]):
                frame["bbox"] = smoothed_trajectory[idx].tolist()

        # Save the updated data back to a JSON file
        output_file_path = (
            f"./data/AARP/frames/full_dataset_annotated_fpp/{stage}_filter_savgol.json"
        )
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=4)
