import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from CVLightningDataset import CVLightningDataset


def predict_constant_velocity_batch(
    past_positions: torch.Tensor, max_future_frames: int
) -> torch.Tensor:
    velocities = past_positions[:, 1:] - past_positions[:, :-1]
    avg_velocity = torch.mean(velocities, dim=1, keepdim=True)
    last_position = past_positions[:, -1:]
    future_frames = torch.arange(
        1, max_future_frames + 1, device=past_positions.device
    ).view(1, -1, 1)
    future_positions = last_position + avg_velocity * future_frames
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
        input_bboxes = input_bboxes.float().permute(0, 2, 1)
        target_bboxes = target_bboxes.float().permute(0, 2, 1)

        predictions = predict_constant_velocity_batch(input_bboxes, output_seq_size)

        all_predictions.append(predictions)
        all_targets.append(target_bboxes)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

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
                dataset = CVLightningDataset(
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

