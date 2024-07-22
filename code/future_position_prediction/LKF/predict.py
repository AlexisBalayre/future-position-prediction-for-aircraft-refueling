import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from LKFLightningDataset import LKFLightningDataset


def linear_kalman_filter(
    past_positions: torch.Tensor, max_future_frames: int
) -> torch.Tensor:
    """
    Linear Kalman Filter for predicting future positions.

    Args:
        past_positions (torch.Tensor): Past positions of the object. Shape: (batch_size, num_frames, 4)
        max_future_frames (int): Maximum number of future frames to predict.

    Returns:
        torch.Tensor: Predicted future positions. Shape: (batch_size, max_future_frames, 4)
    """
    batch_size, num_frames, _ = past_positions.shape

    dt = 1.0

    # State transition matrix
    F = (
        torch.tensor(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )

    # Measurement matrix
    H = (
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )

    # Process noise covariance
    Q = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1) * 0.1

    # Measurement noise covariance
    R = torch.eye(2).unsqueeze(0).repeat(batch_size, 1, 1) * 1.0

    # Initial state estimate
    x = torch.zeros(batch_size, 4, 1, dtype=torch.float32)
    x[:, :2, 0] = past_positions[:, -1, :2]

    # Initial error covariance
    P = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1) * 1.0

    # Kalman Filter loop
    for i in range(num_frames):
        # Predict
        x = torch.bmm(F, x)
        P = torch.bmm(torch.bmm(F, P), F.transpose(1, 2)) + Q

        # Update
        y = past_positions[:, i, :2].unsqueeze(2) - torch.bmm(H, x)
        S = torch.bmm(torch.bmm(H, P), H.transpose(1, 2)) + R
        K = torch.bmm(torch.bmm(P, H.transpose(1, 2)), torch.inverse(S))
        x = x + torch.bmm(K, y)
        P = P - torch.bmm(torch.bmm(K, H), P)

    # Predict future frames
    future_predictions = []
    for _ in range(max_future_frames):
        x = torch.bmm(F, x)
        future_predictions.append(x[:, :2, 0])

    future_predictions = torch.stack(future_predictions, dim=1)
    future_positions = torch.zeros(
        batch_size, max_future_frames, 4, dtype=torch.float32
    )
    future_positions[:, :, :2] = future_predictions
    future_positions[:, :, 2:] = (
        past_positions[:, -1, 2:].unsqueeze(1).repeat(1, max_future_frames, 1)
    )

    return future_positions


@torch.no_grad()
def compute_IoU_batch(
    boxes1: torch.Tensor, boxes2: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    x1, y1, w1, h1 = boxes1.unbind(-1)
    x2, y2, w2, h2 = boxes2.unbind(-1)

    area1 = w1 * h1
    area2 = w2 * h2

    left = torch.max(x1, x2)
    right = torch.min(x1 + w1, x2 + w2)
    top = torch.max(y1, y2)
    bottom = torch.min(y1 + h1, y2 + h2)

    inter_area = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    union_area = area1 + area2 - inter_area

    iou = (inter_area + epsilon) / (union_area + epsilon)
    return torch.clamp(iou, min=0, max=1)


def evaluate_predictions(dataloader, output_seq_size):
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        if batch is None:
            continue

        input_bboxes, target_bboxes = batch
        input_bboxes = input_bboxes.float().permute(
            0, 2, 1
        )  # (batch_size, bbox_dim, input_frames) -> (batch_size, input_frames, bbox_dim)
        target_bboxes = target_bboxes.float().permute(
            0, 2, 1
        )  # (batch_size, bbox_dim, output_frames) -> (batch_size, output_frames, bbox_dim)

        # Use Kalman Filter for prediction
        predictions = linear_kalman_filter(input_bboxes, output_seq_size)

        all_predictions.append(predictions)
        all_targets.append(target_bboxes)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute FIOU using only the last frame
    fiou = compute_IoU_batch(all_predictions[:, -1], all_targets[:, -1]).mean()
    aiou = compute_IoU_batch(
        all_predictions.view(-1, 4), all_targets.view(-1, 4)
    ).mean()

    return fiou.item(), aiou.item()


if __name__ == "__main__":
    train_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/train.json"
    val_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/val.json"
    test_dataset_path = "/Users/alexis/Library/CloudStorage/OneDrive-Balayre&Co/Cranfield/Thesis/thesis-github-repository/data/frames/full_dataset_annotated_fpp/test.json"

    input_seq_sizes = [5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60]
    output_seq_sizes = [5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60]

    results = pd.DataFrame(
        columns=["stage", "input_seq_size", "output_seq_size", "FIOU", "AIOU"]
    )

    for stage in ["test"]:
        dataset_path = locals()[f"{stage}_dataset_path"]

        for input_seq_size in tqdm(
            input_seq_sizes, desc=f"{stage.capitalize()} - Input Seq Sizes"
        ):
            for output_seq_size in tqdm(
                output_seq_sizes, desc="Output Seq Sizes", leave=False
            ):
                dataset = LKFLightningDataset(
                    dataset_path, input_seq_size, output_seq_size, stage
                )
                dataloader = DataLoader(
                    dataset, batch_size=16, num_workers=8, shuffle=False
                )

                fiou, aiou = evaluate_predictions(dataloader, output_seq_size)

                # Save results to DataFrame
                results = pd.concat(
                    [
                        results,
                        pd.DataFrame(
                            {
                                "stage": [stage],
                                "input_seq_size": [input_seq_size],
                                "output_seq_size": [output_seq_size],
                                "FIOU": [fiou],
                                "AIOU": [aiou],
                            }
                        ),
                    ]
                )

                # Save results to disk
                results.to_csv("results.csv", index=False)


