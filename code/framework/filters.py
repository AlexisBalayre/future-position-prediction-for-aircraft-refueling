import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter
from typing import Tuple, Union


def sg_filter_smoothing(
    trajectory: np.ndarray, window_length: int = 5, polyorder: int = 2
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth the trajectory data.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, num_features) or (1, seq_len, num_features).
        window_length (int): The length of the filter window (i.e., the number of coefficients). Must be a positive odd integer.
        polyorder (int): The order of the polynomial used to fit the samples. Must be less than window_length.

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

        if len(trajectory) < window_length:
            return trajectory[np.newaxis, ...]

        smoothed_trajectory = savgol_filter(
            trajectory, window_length, polyorder, axis=0
        )
        return smoothed_trajectory[np.newaxis, ...]
    else:
        smoothed_trajectory = savgol_filter(
            trajectory, window_length, polyorder, axis=0
        )
        return smoothed_trajectory


def gaussian_filter_smoothing(trajectory: np.ndarray, sigma: float = 2) -> np.ndarray:
    """
    Apply Gaussian filter to smooth the trajectory data.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, num_features) or (1, seq_len, num_features).
        sigma (float): The standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)
        smoothed_x = gaussian_filter(trajectory[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter(trajectory[:, 1], sigma=sigma)
        smoothed_width = gaussian_filter(trajectory[:, 2], sigma=sigma)
        smoothed_height = gaussian_filter(trajectory[:, 3], sigma=sigma)
        smoothed_trajectory = np.column_stack(
            (smoothed_x, smoothed_y, smoothed_width, smoothed_height)
        )
        return smoothed_trajectory[np.newaxis, ...]
    else:
        smoothed_x = gaussian_filter(trajectory[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter(trajectory[:, 1], sigma=sigma)
        smoothed_width = gaussian_filter(trajectory[:, 2], sigma=sigma)
        smoothed_height = gaussian_filter(trajectory[:, 3], sigma=sigma)
        smoothed_trajectory = np.column_stack(
            (smoothed_x, smoothed_y, smoothed_width, smoothed_height)
        )
        return smoothed_trajectory


def moving_average_smoothing(
    trajectory: np.ndarray, window_size: int = 20
) -> np.ndarray:
    """
    Apply moving average filter to smooth the trajectory data.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, num_features) or (1, seq_len, num_features).
        window_size (int): The size of the moving window for calculating the average.

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    trajectory = trajectory.squeeze(0)
    if len(trajectory) < window_size:
        return trajectory[np.newaxis, ...]

    smoothed_trajectory = np.stack(
        [
            np.convolve(
                trajectory[:, i], np.ones(window_size) / window_size, mode="same"
            )
            for i in range(trajectory.shape[1])
        ],
        axis=1,
    )

    return smoothed_trajectory[np.newaxis, ...]


def exponential_smoothing(trajectory: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply exponential smoothing to the trajectory data.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, num_features) or (1, seq_len, num_features).
        alpha (float): The smoothing factor. Higher values give more weight to recent observations.

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    trajectory = trajectory.squeeze(0)
    smoothed_trajectory = np.zeros_like(trajectory)
    smoothed_trajectory[0] = trajectory[0]

    for t in range(1, len(trajectory)):
        smoothed_trajectory[t] = (
            alpha * trajectory[t] + (1 - alpha) * smoothed_trajectory[t - 1]
        )

    return smoothed_trajectory[np.newaxis, ...]


def modified_exponential_smoothing(
    trajectory: np.ndarray, alpha: float = 0.1, beta: float = 0.9
) -> np.ndarray:
    """
    Apply modified exponential smoothing to the trajectory data.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, num_features) or (1, seq_len, num_features).
        alpha (float): The smoothing factor for the current observation.
        beta (float): The smoothing factor for the previous smoothed value.

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    trajectory = trajectory.squeeze(0)
    smoothed_trajectory = np.zeros_like(trajectory)
    smoothed_trajectory[0] = trajectory[0]

    for t in range(1, len(trajectory)):
        smoothed_trajectory[t] = (
            alpha * trajectory[t] + beta * smoothed_trajectory[t - 1]
        )

    return smoothed_trajectory[np.newaxis, ...]


def adaptive_smoothing(
    trajectory: np.ndarray,
    initial_window: int = 10,
    polyorder: int = 2,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Apply adaptive smoothing to the trajectory data using Savitzky-Golay filter based on changes in the data.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, num_features) or (1, seq_len, num_features).
        initial_window (int): Initial window size for the Savitzky-Golay filter.
        polyorder (int): The order of the polynomial used to fit the samples.
        threshold (float): The threshold to detect significant changes in the trajectory.

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    trajectory = trajectory.squeeze(0)
    smoothed_trajectory = np.copy(trajectory)

    for i in range(1, len(trajectory)):
        diff = np.abs(trajectory[i] - trajectory[i - 1])
        if np.any(diff > threshold):
            smoothed_traj = savgol_filter(
                trajectory[max(0, i - initial_window) : i + 1],
                min(
                    initial_window, len(trajectory[max(0, i - initial_window) : i + 1])
                ),
                polyorder,
                axis=0,
            )[-1]
            smoothed_trajectory[i] = smoothed_traj
        else:
            smoothed_trajectory[i] = trajectory[i]

    return smoothed_trajectory[np.newaxis, ...]


def hybrid_smoothing(
    trajectory: np.ndarray,
    savgol_window_length: int = 15,
    savgol_polyorder: int = 2,
    ma_window_size: int = 3,
) -> np.ndarray:
    """
    Apply a hybrid smoothing approach that combines Savitzky-Golay and moving average filters.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, num_features) or (1, seq_len, num_features).
        savgol_window_length (int): The length of the Savitzky-Golay filter window.
        savgol_polyorder (int): The order of the polynomial used in the Savitzky-Golay filter.
        ma_window_size (int): The window size for the moving average filter.

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    trajectory = trajectory.squeeze(0)
    if len(trajectory) < savgol_window_length:
        return trajectory[np.newaxis, ...]

    savgol_smoothed = savgol_filter(
        trajectory, savgol_window_length, savgol_polyorder, axis=0
    )
    ma_smoothed = moving_average_smoothing(
        savgol_smoothed[np.newaxis, ...], window_size=ma_window_size
    ).squeeze(0)

    return ma_smoothed[np.newaxis, ...]


def kalman_filter_smoothing(trajectory: np.ndarray) -> np.ndarray:
    """
    Apply Kalman filter to smooth the trajectory data, assuming a constant velocity model.

    Args:
        trajectory (np.ndarray): The trajectory data to be smoothed. Expected shape (seq_len, 6).

    Returns:
        np.ndarray: The smoothed trajectory. Shape is the same as input.
    """
    seq_len, num_features = trajectory.shape
    assert num_features == 6, "Expected input shape (seq_len, 6)"

    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=6, dim_z=6)

    # State Transition Matrix (assuming constant velocity model for simplicity)
    dt = 1.0  # Time step
    kf.F = np.array(
        [
            [1, 0, dt, 0, 0.5 * dt**2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    # Measurement Matrix (assuming we measure all states directly)
    kf.H = np.eye(6)

    # Process Noise Covariance Matrix
    kf.Q = np.array(
        [
            [1e-3, 0, 0, 0, 0, 0],
            [0, 1e-3, 0, 0, 0, 0],
            [0, 0, 1e-2, 0, 0, 0],
            [0, 0, 0, 1e-2, 0, 0],
            [0, 0, 0, 0, 1e-1, 0],
            [0, 0, 0, 0, 0, 1e-1],
        ]
    )

    # Measurement Noise Covariance Matrix
    kf.R = np.eye(6) * 1e-1  # Increase to reduce sensitivity to measurement noise

    # Initial State Covariance Matrix
    kf.P = np.eye(6) * 1e2  # Start with a reasonable initial uncertainty

    # Initial state - use the first point as the initial state
    kf.x[:6] = trajectory[0, :6].reshape(6, 1)

    smoothed_trajectory = []

    for t in range(seq_len):
        kf.predict()
        kf.update(trajectory[t, :6].reshape(6, 1))
        smoothed_trajectory.append(kf.x[:6].reshape(-1))  # Flatten to (6,)

    return np.array(smoothed_trajectory)
