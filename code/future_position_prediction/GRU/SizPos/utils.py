import torch
from typing import Tuple


def convert_velocity_to_positions(
    predicted_velocity: torch.Tensor,
    past_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Convert predicted normalized velocity to normalized positions.

    Args:
        predicted_velocity (torch.Tensor): Predicted normalized velocity of shape (batch_size, seq_len, 4).
        past_positions (torch.Tensor): Past normalized positions of shape (batch_size, past_seq_len, 4).

    Returns:
        torch.Tensor: Predicted normalized positions of shape (batch_size, seq_len, 4).
    """
    batch_size, seq_len, _ = predicted_velocity.shape
    predicted_positions = torch.zeros(
        batch_size, seq_len, 4, device=predicted_velocity.device
    )

    current = past_positions[:, -1, :].clone()

    for i in range(seq_len):
        next_position = current.clone()

        # Update center coordinates (x, y)
        next_position[:, :2] += predicted_velocity[:, i, :2]

        # Update width and height
        next_position[:, 2:] += predicted_velocity[:, i, 2:]

        # Clamp the normalized positions to [0, 1]
        next_position = torch.clamp(next_position, 0, 1)

        predicted_positions[:, i, :] = next_position
        current = next_position.clone()

    return predicted_positions


@torch.no_grad()
def compute_ADE(
    predicted_positions: torch.Tensor,
    ground_truth_positions: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate the Average Displacement Error (ADE) for denormalized coordinates.

    Args:
        predicted_positions (torch.Tensor): Predicted positions of shape (batch_size, seq_len, 4).
        ground_truth_positions (torch.Tensor): Ground truth positions of shape (batch_size, seq_len, 4).
        image_size (Tuple[int, int]): Tuple of (width, height) representing the image dimensions.

    Returns:
        torch.Tensor: The Average Displacement Error.
    """
    denorm_factor = torch.tensor(
        [image_size[0], image_size[1], image_size[0], image_size[1]],
        device=predicted_positions.device,
    )

    # Denormalize the predicted and ground truth positions
    predicted_positions = predicted_positions * denorm_factor
    ground_truth_positions = ground_truth_positions * denorm_factor

    # Calculate the Euclidean distance between predicted and ground truth positions
    return torch.mean(
        torch.norm(
            predicted_positions[:, :, :2] - ground_truth_positions[:, :, :2], dim=2
        )
    )


@torch.no_grad()
def compute_FDE(
    predicted_positions: torch.Tensor,
    ground_truth_positions: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate the Final Displacement Error (FDE) for denormalized coordinates.

    Args:
        predicted_positions (torch.Tensor): Predicted positions of shape (batch_size, seq_len, 4).
        ground_truth_positions (torch.Tensor): Ground truth positions of shape (batch_size, seq_len, 4).

    Returns:
        torch.Tensor: The Final Displacement Error.
    """
    denorm_factor = torch.tensor(
        [
            image_size[0],
            image_size[1],
            image_size[0],
            image_size[1],
        ],
        device=predicted_positions.device,
    )

    # Denormalize the predicted and ground truth positions
    predicted_positions = predicted_positions * denorm_factor
    ground_truth_positions = ground_truth_positions * denorm_factor

    # Calculate the Euclidean distance between predicted and ground truth positions
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
    Calculate Intersection over Union (IoU) between predicted and ground truth boxes in YOLO format.

    Args:
        predicted_boxes (torch.Tensor): Predicted boxes in xywh format (center_x, center_y, width, height).
        ground_truth_boxes (torch.Tensor): Ground truth boxes in xywh format (center_x, center_y, width, height).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU between predicted and ground truth boxes.
    """
    # Extract coordinates
    pred_x1 = predicted_boxes[:, 0] - predicted_boxes[:, 2] / 2
    pred_y1 = predicted_boxes[:, 1] - predicted_boxes[:, 3] / 2
    pred_x2 = predicted_boxes[:, 0] + predicted_boxes[:, 2] / 2
    pred_y2 = predicted_boxes[:, 1] + predicted_boxes[:, 3] / 2

    gt_x1 = ground_truth_boxes[:, 0] - ground_truth_boxes[:, 2] / 2
    gt_y1 = ground_truth_boxes[:, 1] - ground_truth_boxes[:, 3] / 2
    gt_x2 = ground_truth_boxes[:, 0] + ground_truth_boxes[:, 2] / 2
    gt_y2 = ground_truth_boxes[:, 1] + ground_truth_boxes[:, 3] / 2

    # Intersection coordinates
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    # Intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # Box areas
    pred_area = predicted_boxes[:, 2] * predicted_boxes[:, 3]
    gt_area = ground_truth_boxes[:, 2] * ground_truth_boxes[:, 3]

    # Union area
    union_area = pred_area + gt_area - inter_area

    # IoU
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


@torch.no_grad()
def convert_PosSize_to_PosVel(
    positions: torch.Tensor, sizes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert positions and sizes to bounding boxes and velocities

    Args:
        positions (torch.Tensor): Positions of shape (batch_size, seq_len, 6).
        sizes (torch.Tensor): Sizes of shape (batch_size, seq_len, 4).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Bounding boxes and velocities of shape (batch_size, seq_len, 4).
    """
    # Convert positions and sizes to YOLO format (xcenter, ycenter, width, height)
    bboxes = torch.zeros_like(positions[:, :, :4])  # (batch_size, seq_len, 4)
    velocities = torch.zeros_like(positions[:, :, :4])  # (batch_size, seq_len, 4)

    # Bounding boxes: xcenter, ycenter, width, height
    bboxes[:, :, 0] = positions[:, :, 0]  # xcenter
    bboxes[:, :, 1] = positions[:, :, 1]  # ycenter
    bboxes[:, :, 2] = sizes[:, :, 0]  # width
    bboxes[:, :, 3] = sizes[:, :, 1]  # height

    # Velocities: velx, vely, deltaw, deltah
    velocities[:, :, 0] = positions[:, :, 2]  # velx
    velocities[:, :, 1] = positions[:, :, 3]  # vely
    velocities[:, :, 2] = sizes[:, :, 2]  # deltaw
    velocities[:, :, 3] = sizes[:, :, 3]  # deltah

    return bboxes, velocities
