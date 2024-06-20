import torch
import torch.nn as nn
import torch.nn.functional as F


class CubicLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(CubicLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.temporal_conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.spatial_conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x_t, h_prev, c_prev, h_s_prev, c_s_prev):
        # Temporal branch
        combined = torch.cat([x_t, h_prev], dim=1)
        temp_conv = self.temporal_conv(combined)
        i, f, o, g = torch.split(temp_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        # Spatial branch
        combined_s = torch.cat([x_t, h_s_prev], dim=1)
        spatial_conv = self.spatial_conv(combined_s)
        i_s, f_s, o_s, g_s = torch.split(spatial_conv, self.hidden_dim, dim=1)

        i_s = torch.sigmoid(i_s)
        f_s = torch.sigmoid(f_s)
        o_s = torch.sigmoid(o_s)
        g_s = torch.tanh(g_s)

        c_s_t = f_s * c_s_prev + i_s * g_s
        h_s_t = o_s * torch.tanh(c_s_t)

        return h_t, c_t, h_s_t, c_s_t
