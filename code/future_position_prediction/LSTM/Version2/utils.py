import torch
from typing import Tuple


def convert_velocity_to_positions(
    predicted_velocity: torch.Tensor,
    past_positions: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert predicted velocity to positions.

    Args:
        predicted_velocity (torch.Tensor): Predicted velocity of shape (batch_size, seq_len, 4).
        past_positions (torch.Tensor): Past positions of shape (batch_size, past_seq_len, 4).
        image_size (Tuple[int, int]): Tuple of (width, height) representing the image dimensions.

    Returns:
        torch.Tensor: Predicted positions of shape (batch_size, seq_len, 4) with denormalized coordinates.
    """
    batch_size, seq_len, _ = predicted_velocity.shape
    predicted_positions = torch.zeros(
        batch_size, seq_len, 4, device=predicted_velocity.device
    )

    current = past_positions[:, -1, :].clone()
    current = torch.cat(
        [
            current[:, :2]
            * torch.tensor([image_size[0], image_size[1]], device=current.device),
            current[:, 2:]
            * torch.tensor([image_size[0], image_size[1]], device=current.device),
        ],
        dim=1,
    )

    for i in range(seq_len):
        next_position = current.clone() + predicted_velocity[:, i, :]
        next_position[:, 0::2] = torch.clamp(next_position[:, 0::2], 0, image_size[0])
        next_position[:, 1::2] = torch.clamp(next_position[:, 1::2], 0, image_size[1])
        predicted_positions[:, i, :] = next_position
        current = next_position.clone()

    return predicted_positions


@torch.no_grad()
def compute_ADE(
    predicted_positions: torch.Tensor, ground_truth_positions: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Average Displacement Error (ADE) for denormalized coordinates.

    Args:
        predicted_positions (torch.Tensor): Predicted positions of shape (batch_size, seq_len, 4).
        ground_truth_positions (torch.Tensor): Ground truth positions of shape (batch_size, seq_len, 4).

    Returns:
        torch.Tensor: The Average Displacement Error.
    """
    return torch.mean(
        torch.norm(
            predicted_positions[:, :, :2] - ground_truth_positions[:, :, :2], dim=2
        )
    )


@torch.no_grad()
def compute_FDE(
    predicted_positions: torch.Tensor, ground_truth_positions: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Final Displacement Error (FDE) for denormalized coordinates.

    Args:
        predicted_positions (torch.Tensor): Predicted positions of shape (batch_size, seq_len, 4).
        ground_truth_positions (torch.Tensor): Ground truth positions of shape (batch_size, seq_len, 4).

    Returns:
        torch.Tensor: The Final Displacement Error.
    """
    return torch.mean(
        torch.norm(
            predicted_positions[:, -1, :2] - ground_truth_positions[:, -1, :2], dim=1
        )
    )


@torch.no_grad()
def compute_IoU(
    predicted_boxes: torch.Tensor,
    ground_truth_boxes: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth boxes.

    Args:
        predicted_boxes (torch.Tensor): Predicted boxes in xywh format (x, y, w, h).
        ground_truth_boxes (torch.Tensor): Ground truth boxes in xywh format (x, y, w, h).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU between predicted and ground truth boxes.
    """
    # Ensure positive dimensions without modifying the original tensors
    pred_wh = torch.abs(predicted_boxes[:, 2:])
    gt_wh = torch.abs(ground_truth_boxes[:, 2:])

    pred_xyxy = torch.cat(
        [
            predicted_boxes[:, :2] - pred_wh / 2,
            predicted_boxes[:, :2] + pred_wh / 2,
        ],
        dim=-1,
    )
    gt_xyxy = torch.cat(
        [
            ground_truth_boxes[:, :2] - gt_wh / 2,
            ground_truth_boxes[:, :2] + gt_wh / 2,
        ],
        dim=-1,
    )

    inter_tl = torch.max(pred_xyxy[:, :2], gt_xyxy[:, :2])
    inter_br = torch.min(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
    inter_area = torch.prod(torch.clamp(inter_br - inter_tl, min=0), dim=1)

    pred_area = torch.prod(pred_wh, dim=1)
    gt_area = torch.prod(gt_wh, dim=1)
    union_area = pred_area + gt_area - inter_area

    iou = (inter_area + epsilon) / (union_area + epsilon)
    return torch.clamp(iou, min=0, max=1)  # Ensure IoU is between 0 and 1


@torch.no_grad()
def compute_AIOU(
    predicted_positions: torch.Tensor, ground_truth_positions: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Average Intersection over Union (AIoU) for denormalized coordinates.

    Args:
        predicted_positions (torch.Tensor): Predicted positions of shape (batch_size, seq_len, 4).
        ground_truth_positions (torch.Tensor): Ground truth positions of shape (batch_size, seq_len, 4).

    Returns:
        torch.Tensor: The Average Intersection over Union.
    """
    iou = compute_IoU(
        predicted_positions.view(-1, 4), ground_truth_positions.view(-1, 4)
    )
    aiou = torch.mean(iou)
    if torch.isnan(aiou) or torch.isinf(aiou):
        print(
            f"Warning: AIOU is {aiou}. Predicted: {predicted_positions}, Ground Truth: {ground_truth_positions}"
        )
        aiou = torch.tensor(0.0, device=predicted_positions.device)
    return aiou


@torch.no_grad()
def compute_FIOU(
    predicted_positions: torch.Tensor, ground_truth_positions: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Final Intersection over Union (FIoU) for denormalized coordinates.

    Args:
        predicted_positions (torch.Tensor): Predicted positions of shape (batch_size, seq_len, 4).
        ground_truth_positions (torch.Tensor): Ground truth positions of shape (batch_size, seq_len, 4).

    Returns:
        torch.Tensor: The Final Intersection over Union.
    """
    fiou = torch.mean(
        compute_IoU(predicted_positions[:, -1], ground_truth_positions[:, -1])
    )
    if torch.isnan(fiou) or torch.isinf(fiou):
        print(
            f"Warning: FIOU is {fiou}. Predicted: {predicted_positions[:, -1]}, Ground Truth: {ground_truth_positions[:, -1]}"
        )
        fiou = torch.tensor(0.0, device=predicted_positions.device)
    return fiou
