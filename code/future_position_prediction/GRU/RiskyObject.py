import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .gru_cells import CustomGRUCell, GRUNet, CorGRU, decoderGRU
from .attention import SelfAttentionAggregation, SpatialAttention
from .predictors import BboxPredictor, MidPred

class RiskyObject(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, n_frames: int = 100, fps: float = 20.0):
        super(RiskyObject, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fps = fps
        self.n_frames = n_frames
        self.n_layers = 1
        self.time_steps = 4 + 5

        # Feature extraction layers
        self.rgb_feature_extractor = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.bbox_feature_extractor = nn.Sequential(nn.Linear(5, hidden_dim), nn.ReLU())
        self.distance_feature_extractor = nn.Sequential(nn.Linear(6, hidden_dim), nn.ReLU())

        # GRU networks
        self.gru_net = GRUNet(hidden_dim * 2, hidden_dim, 4 * self.time_steps, self.n_layers, 32, self.time_steps)
        self.gru_net_cor = CorGRU(32, 32, self.n_layers)

        # Attention mechanisms
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.aggregation = SelfAttentionAggregation(4, hidden_dim)

        # Decoder
        self.decoder = decoderGRU(hidden_dim, hidden_dim, 4, self.n_layers)

        # Loss functions
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor, toa: torch.Tensor, flow: torch.Tensor, 
                hidden_in: Optional[torch.Tensor] = None, testing: bool = False) -> Tuple[Dict, List, List, Dict, Dict]:
        """
        Forward pass of the RiskyObject model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_frames, max_boxes + 1, feature_dim)
            y (torch.Tensor): Ground truth tensor of shape (batch_size, n_frames, max_boxes, 6)
            toa (torch.Tensor): Time of arrival tensor of shape (batch_size, 1)
            flow (torch.Tensor): Optical flow tensor
            hidden_in (Optional[torch.Tensor]): Initial hidden state
            testing (bool): Whether in testing mode

        Returns:
            Tuple[Dict, List, List, Dict, Dict]: Losses, outputs, labels, predicted bounding boxes, and ground truth bounding boxes
        """
        losses = {'total_loss': 0, 'inter_loss': 0, 'bbox_loss': 0, 'object_loss': 0}
        all_outputs = []
        all_labels = []
        pred_bbox = {t: {} for t in range(x.size(1) - self.time_steps)}
        gt_bbox = {t: {} for t in range(x.size(1) - self.time_steps)}

        hidden_states = {}
        hidden_states_cor = {}

        # Process each frame
        for t in range(x.size(1) - self.time_steps):
            frame_outputs, frame_labels, frame_losses = self.process_frame(t, x, y, flow, hidden_states, hidden_states_cor)
            
            all_outputs.extend(frame_outputs)
            all_labels.extend(frame_labels)
            for key, value in frame_losses.items():
                losses[key] += value

            # Update hidden states with spatial attention
            hidden_states = self.spatial_attention(hidden_states)
            hidden_states_cor = self.spatial_attention(hidden_states_cor)

        return losses, all_outputs, all_labels, pred_bbox, gt_bbox

    def process_frame(self, t: int, x: torch.Tensor, y: torch.Tensor, flow: torch.Tensor, 
                      hidden_states: Dict[str, torch.Tensor], hidden_states_cor: Dict[str, torch.Tensor]) -> Tuple[List, List, Dict]:
        """
        Process a single frame.

        Args:
            t (int): Current time step
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Ground truth tensor
            flow (torch.Tensor): Optical flow tensor
            hidden_states (Dict[str, torch.Tensor]): Hidden states for each object
            hidden_states_cor (Dict[str, torch.Tensor]): Hidden states for coordinate GRU

        Returns:
            Tuple[List, List, Dict]: Frame outputs, labels, and losses
        """
        frame_outputs = []
        frame_labels = []
        frame_losses = {'total_loss': 0, 'inter_loss': 0, 'bbox_loss': 0, 'object_loss': 0}

        # Extract features
        rgb_features = self.rgb_feature_extractor(flow[:, t])
        img_embed = rgb_features[:, 0, :].unsqueeze(1).repeat(1, 30, 1)
        obj_embed = rgb_features[:, 1:, :]
        combined_features = torch.cat([obj_embed, img_embed], dim=-1)

        # Process each bounding box
        for bbox in range(30):
            if y[0][t][bbox][0] == 0:  # ignore if there is no bounding box
                continue

            track_id = str(y[0][t][bbox][0].cpu().detach().numpy())
            hidden_state = hidden_states.get(track_id, torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device))
            hidden_state_cor = hidden_states_cor.get(track_id, torch.zeros(self.n_layers, x.size(0), 32).to(x.device))

            # Process bounding box and distance information
            bbox_info = y[0][t][bbox]
            normalized_bbox = self.normalize_bbox(bbox_info[1:5])
            bbox_features = self.bbox_feature_extractor(torch.cat([normalized_bbox, bbox_info[6].unsqueeze(0)]))
            
            distance_info = bbox_info[7:13]
            distance_features = self.distance_feature_extractor(distance_info.unsqueeze(0))

            # GRU processing
            output_cor, hidden_state_cor = self.gru_net_cor(distance_features, hidden_state_cor)
            output, hidden_state, predicted_bbox = self.gru_net(combined_features[0][bbox].unsqueeze(0).unsqueeze(0),
                                                                hidden_state, bbox_features, output_cor)

            # Compute losses and update hidden states
            losses = self.compute_losses(output, y[0][t][bbox][5], predicted_bbox, self.get_future_bbox(y, t, track_id))
            for key, value in losses.items():
                frame_losses[key] += value

            hidden_states[track_id] = hidden_state
            hidden_states_cor[track_id] = hidden_state_cor

            frame_outputs.append(output.detach().cpu().numpy())
            frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())

        return frame_outputs, frame_labels, frame_losses

    def normalize_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        """Normalize bounding box coordinates."""
        width, height = 1920, 1200  # Assuming these are the image dimensions
        return torch.tensor([bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height])

    def get_future_bbox(self, y: torch.Tensor, t: int, track_id: str) -> List[float]:
        """Get future bounding box coordinates for the given track_id."""
        future_bbox = []
        for i in range(self.time_steps):
            if float(track_id) in y[0][t+i+1][:, 0]:
                bbox = y[0][t+i+1][y[0][t+i+1][:, 0] == float(track_id)][0][1:5]
                future_bbox.extend(self.normalize_bbox(bbox))
            else:
                break
        return future_bbox

    def compute_losses(self, output: torch.Tensor, target: torch.Tensor, predicted_bbox: torch.Tensor, target_bbox: List[float]) -> Dict[str, torch.Tensor]:
        """Compute various losses for the model."""
        losses = {}
        if len(target_bbox) > 0:
            target_bbox = torch.tensor(target_bbox, device=output.device).unsqueeze(0)
            predicted_bbox = predicted_bbox[0][:len(target_bbox[0])]
            losses['bbox_loss'] = self.bbox_loss(predicted_bbox, target_bbox)
            losses['inter_loss'] = self.bbox_loss(predicted_bbox, target_bbox) * 0.3
        else:
            losses['bbox_loss'] = torch.tensor(0.0, device=output.device)
            losses['inter_loss'] = torch.tensor(0.0, device=output.device)

        losses['object_loss'] = self.ce_loss(output, target.unsqueeze(0).unsqueeze(0)) * 0.05
        losses['total_loss'] = losses['object_loss'] + losses['bbox_loss'] + losses['inter_loss']
        
        return losses