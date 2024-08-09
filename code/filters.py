import numpy as np
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter


def smooth_trajectory(trajectory, window_length=5, polyorder=2):
    trajectory = trajectory.squeeze(0)  # Remove the batch dimension
    if len(trajectory) < window_length:
        return trajectory[np.newaxis, ...]

    # Apply smoothing only to the x and y coordinates (first two columns)
    smoothed_xy = savgol_filter(trajectory[:, :2], window_length, polyorder, axis=0)

    # Combine smoothed x, y with original w, h
    smoothed_trajectory = np.hstack((smoothed_xy, trajectory[:, 2:]))

    return smoothed_trajectory[np.newaxis, ...]


def moving_average_smoothing(trajectory, window_size=20):
    trajectory = trajectory.squeeze(0)  # Remove the batch dimension
    if len(trajectory) < window_size:
        return trajectory[np.newaxis, ...]

    # Apply moving average only to x and y coordinates (first two columns)
    smoothed_xy = np.stack(
        [
            np.convolve(
                trajectory[:, i], np.ones(window_size) / window_size, mode="same"
            )
            for i in range(2)
        ],
        axis=1,
    )

    # Combine smoothed x, y with original w, h
    smoothed_trajectory = np.hstack((smoothed_xy, trajectory[:, 2:]))

    return smoothed_trajectory[np.newaxis, ...]


def exponential_smoothing(trajectory, alpha=0.1):
    trajectory = trajectory.squeeze(0)  # Remove the batch dimension
    smoothed_trajectory = np.zeros_like(trajectory)
    smoothed_trajectory[0, :2] = trajectory[0, :2]

    # Apply exponential smoothing only to x and y coordinates (first two columns)
    for t in range(1, len(trajectory)):
        smoothed_trajectory[t, :2] = (
            alpha * trajectory[t, :2] + (1 - alpha) * smoothed_trajectory[t - 1, :2]
        )

    # Retain original w and h columns
    smoothed_trajectory[:, 2:] = trajectory[:, 2:]

    return smoothed_trajectory[np.newaxis, ...]


def modified_exponential_smoothing(trajectory, alpha=0.1, beta=0.9):
    trajectory = trajectory.squeeze(0)  # Remove the batch dimension
    smoothed_trajectory = np.zeros_like(trajectory)
    smoothed_trajectory[0, :2] = trajectory[0, :2]

    # Apply modified exponential smoothing only to x and y coordinates (first two columns)
    for t in range(1, len(trajectory)):
        smoothed_trajectory[t, :2] = (
            alpha * trajectory[t, :2] + beta * smoothed_trajectory[t - 1, :2]
        )

    # Retain original w and h columns
    smoothed_trajectory[:, 2:] = trajectory[:, 2:]

    return smoothed_trajectory[np.newaxis, ...]


def adaptive_smoothing(trajectory, initial_window=10, polyorder=2, threshold=0.1):
    trajectory = trajectory.squeeze(0)  # Remove the batch dimension
    smoothed_trajectory = np.copy(trajectory)

    # Apply adaptive smoothing only to x and y coordinates (first two columns)
    for i in range(1, len(trajectory)):
        diff = np.abs(trajectory[i, :2] - trajectory[i - 1, :2])
        if np.any(diff > threshold):
            smoothed_xy = savgol_filter(
                trajectory[max(0, i - initial_window) : i + 1, :2],
                min(
                    initial_window, len(trajectory[max(0, i - initial_window) : i + 1])
                ),
                polyorder,
                axis=0,
            )[-1]
            smoothed_trajectory[i, :2] = smoothed_xy
        else:
            smoothed_trajectory[i, :2] = trajectory[i, :2]

    return smoothed_trajectory[np.newaxis, ...]


def hybrid_smoothing(
    trajectory, savgol_window_length=15, savgol_polyorder=2, ma_window_size=3
):
    trajectory = trajectory.squeeze(0)  # Remove the batch dimension
    if len(trajectory) < savgol_window_length:
        return trajectory[np.newaxis, ...]

    # Apply Savitzky-Golay smoothing only to x and y coordinates (first two columns)
    savgol_smoothed = savgol_filter(
        trajectory[:, :2], savgol_window_length, savgol_polyorder, axis=0
    )

    # Apply moving average smoothing on top of Savitzky-Golay smoothing
    ma_smoothed_xy = moving_average_smoothing(
        savgol_smoothed[np.newaxis, ...], window_size=ma_window_size
    ).squeeze(0)

    # Combine smoothed x, y with original w, h
    smoothed_trajectory = np.hstack((ma_smoothed_xy, trajectory[:, 2:]))

    return smoothed_trajectory[np.newaxis, ...]


def kalman_filter_smoothing(trajectory):
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
