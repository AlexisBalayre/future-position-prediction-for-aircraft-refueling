import torch
from torch import nn, Tensor
from typing import Tuple


def convert_velocity_to_positions(
    predicted_velocity: torch.Tensor,
    past_positions: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert predicted normalized velocity to denormalized positions.

    Args:
        predicted_velocity (torch.Tensor): Predicted normalized velocity of shape (batch_size, seq_len, 4).
        past_positions (torch.Tensor): Past normalized positions of shape (batch_size, past_seq_len, 4).
        image_size (Tuple[int, int]): Tuple of (width, height) representing the image dimensions.

    Returns:
        torch.Tensor: Predicted denormalized positions of shape (batch_size, seq_len, 4).
    """
    batch_size, seq_len, _ = predicted_velocity.shape
    predicted_positions = torch.zeros(
        batch_size, seq_len, 4, device=predicted_velocity.device
    )

    current = past_positions[:, -1, :].clone()

    for i in range(seq_len):
        next_position = current.clone()
        next_position[:, :2] += predicted_velocity[
            :, i, :2
        ]  # Update center coordinates
        next_position[:, 2:] *= torch.exp(
            predicted_velocity[:, i, 2:]
        )  # Update width and height

        # Clamp the normalized positions to [0, 1]
        next_position = torch.clamp(next_position, 0, 1)

        predicted_positions[:, i, :] = next_position
        current = next_position.clone()

    # Denormalize the predicted positions
    denorm_factor = torch.tensor(
        [image_size[0], image_size[1], image_size[0], image_size[1]],
        device=predicted_positions.device,
    )
    denormalized_positions = predicted_positions * denorm_factor.unsqueeze(0).unsqueeze(
        0
    )

    return denormalized_positions


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
        predicted_positions.reshape(-1, 4), ground_truth_positions.reshape(-1, 4)
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


def get_src_trg(
    sequence: torch.Tensor, enc_seq_len: int, target_seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate the src (encoder input), trg (decoder input) and trg_y (the target)
    sequences from a sequence.
    Args:
        sequence: tensor, a 3D tensor of shape [batch_size, seq_len, feature_size] where
                  seq_len = encoder input length + target sequence length
        enc_seq_len: int, the desired length of the input to the transformer encoder
        target_seq_len: int, the desired length of the target sequence (the
                        one against which the model output is compared)
    Return:
        src: tensor, 3D, used as input to the transformer encoder
        trg: tensor, 3D, used as input to the transformer decoder
        trg_y: tensor, 3D, the target sequence against which the model output
               is compared when computing loss.
    """
    assert (
        sequence.shape[1] == enc_seq_len + target_seq_len
    ), "Sequence length does not equal (input length + target length)"

    # Encoder input
    src = sequence[:, :enc_seq_len, :]

    # Decoder input. It should contain the last value of src and all
    # values of trg_y except the last (i.e. it must be shifted right by 1)
    trg = sequence[:, enc_seq_len - 1 : -1, :]

    assert (
        trg.shape[1] == target_seq_len
    ), "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[:, -target_seq_len:, :]

    assert (
        trg_y.shape[1] == target_seq_len
    ), "Length of trg_y does not match target sequence length"

    return src, trg, trg_y


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float("-inf"), diagonal=1)
