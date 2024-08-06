import torch
from torch import Tensor
from typing import Dict, Tuple

from future_position_prediction.GRU.SizPos.utils import (
    compute_ADE,
    compute_FDE,
    compute_AIOU,
    compute_FIOU,
)


class MetricsMonitoring:
    """
    A class for monitoring and computing various metrics for trajectory prediction.

    This class keeps track of several metrics including Average Displacement Error (ADE),
    Final Displacement Error (FDE), Average Intersection over Union (AIOU), and
    Final Intersection over Union (FIOU) for both direct predictions and predictions
    derived from velocity.

    Attributes:
        image_size (Tuple[int, int]): The size of the image (width, height) used for normalization.
    """

    def __init__(self, image_size: Tuple[int, int]):
        """
        Initialize the MetricsMonitoring object.

        Args:
            image_size (Tuple[int, int]): The size of the image (width, height) used for normalization.
        """
        self.image_size = image_size
        self.reset()

    def reset(self):
        """
        Reset all accumulated metrics to their initial state.

        This method should be called at the beginning of each epoch or when starting a new evaluation.
        """
        self.total_ade = 0.0
        self.total_fde = 0.0
        self.total_aiou = 0.0
        self.total_fiou = 0.0

        self.total_ade_from_vel = 0.0
        self.total_fde_from_vel = 0.0
        self.total_aiou_from_vel = 0.0
        self.total_fiou_from_vel = 0.0

        self.total_samples = 0

    def update(
        self,
        predicted_bbox: Tensor,
        predicted_bbox_from_vel: Tensor,
        ground_truth_bbox: Tensor,
    ):
        """
        Update the metrics with a new batch of predictions and ground truth.

        This method computes the metrics for the current batch and adds them to the running totals.

        Args:
            predicted_bbox (Tensor): Predicted bounding boxes of shape (batch_size, num_frames, 4)
            predicted_bbox_from_vel (Tensor): Predicted bounding boxes from velocities of shape (batch_size, num_frames, 4)
            ground_truth_bbox (Tensor): Ground truth bounding boxes of shape (batch_size, num_frames, 4)
        """
        batch_size = predicted_bbox.shape[0]

        self.total_ade += (
            compute_ADE(predicted_bbox, ground_truth_bbox, self.image_size) * batch_size
        )
        self.total_fde += (
            compute_FDE(predicted_bbox, ground_truth_bbox, self.image_size) * batch_size
        )
        self.total_aiou += compute_AIOU(predicted_bbox, ground_truth_bbox) * batch_size
        self.total_fiou += compute_FIOU(predicted_bbox, ground_truth_bbox) * batch_size
        self.total_ade_from_vel += (
            compute_ADE(
                predicted_bbox_from_vel,
                ground_truth_bbox,
                self.image_size,
            )
            * batch_size
        )
        self.total_fde_from_vel += (
            compute_FDE(
                predicted_bbox_from_vel,
                ground_truth_bbox,
                self.image_size,
            )
            * batch_size
        )
        self.total_aiou_from_vel += (
            compute_AIOU(predicted_bbox_from_vel, ground_truth_bbox) * batch_size
        )
        self.total_fiou_from_vel += (
            compute_FIOU(predicted_bbox_from_vel, ground_truth_bbox) * batch_size
        )

        self.total_samples += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute the average metrics over all samples seen since the last reset.

        This method calculates the mean values for all tracked metrics and returns them in a dictionary.

        Returns:
            Dict[str, float]: A dictionary containing the average metrics. The keys are:
                - "ADE": Average Displacement Error
                - "FDE": Final Displacement Error
                - "AIOU": Average Intersection over Union
                - "FIOU": Final Intersection over Union
                - "ADE_from_vel": ADE for predictions derived from velocity
                - "FDE_from_vel": FDE for predictions derived from velocity
                - "AIOU_from_vel": AIOU for predictions derived from velocity
                - "FIOU_from_vel": FIOU for predictions derived from velocity
                - "Best_FIOU": The better of FIOU and FIOU_from_vel

        Note:
            If no samples have been processed (total_samples == 0), all metrics will be returned as 0.0.
        """
        if self.total_samples == 0:
            return {
                "ADE": 0.0,
                "FDE": 0.0,
                "AIOU": 0.0,
                "FIOU": 0.0,
                "ADE_from_vel": 0.0,
                "FDE_from_vel": 0.0,
                "AIOU_from_vel": 0.0,
                "FIOU_from_vel": 0.0,
                "Best_FIOU": 0.0,
            }

        mean_ade = self.total_ade / self.total_samples
        mean_fde = self.total_fde / self.total_samples
        mean_aiou = self.total_aiou / self.total_samples
        mean_fiou = self.total_fiou / self.total_samples
        mean_ade_from_vel = self.total_ade_from_vel / self.total_samples
        mean_fde_from_vel = self.total_fde_from_vel / self.total_samples
        mean_aiou_from_vel = self.total_aiou_from_vel / self.total_samples
        mean_fiou_from_vel = self.total_fiou_from_vel / self.total_samples

        return {
            "ADE": mean_ade,
            "FDE": mean_fde,
            "AIOU": mean_aiou,
            "FIOU": mean_fiou,
            "ADE_from_vel": mean_ade_from_vel,
            "FDE_from_vel": mean_fde_from_vel,
            "AIOU_from_vel": mean_aiou_from_vel,
            "FIOU_from_vel": mean_fiou_from_vel,
            "Best_FIOU": max(mean_fiou, mean_fiou_from_vel),
        }
