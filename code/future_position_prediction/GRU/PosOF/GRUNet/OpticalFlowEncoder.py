import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowEncoder(nn.Module):
    """
    A convolutional neural network (CNN) module designed to encode optical flow inputs into a latent feature vector.

    The architecture consists of three convolutional layers followed by a global average pooling layer,
    which compresses the spatial dimensions to produce a fixed-size feature vector.

    Args:
        input_channels (int): Number of input channels (e.g., 2 for optical flow with x and y components).
        hidden_dim (int): Dimension of the latent feature vector produced by the encoder.
    """

    def __init__(self, input_channels, hidden_dim):
        super(OpticalFlowEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass through the OpticalFlowEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            torch.Tensor: Encoded feature vector of shape (batch_size, hidden_dim).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), x.size(1))