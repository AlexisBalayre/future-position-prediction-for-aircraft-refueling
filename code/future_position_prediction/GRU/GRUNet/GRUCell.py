import torch
import torch.nn as nn
import numpy as np


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bbox_dim, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bbox_dim = bbox_dim
        self.bias = bias

        self.input_to_hidden = nn.Linear(
            self.input_size, 3 * self.hidden_size, bias=self.bias
        )
        self.hidden_to_hidden = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=self.bias
        )
        self.vel_to_vel = nn.Linear(self.bbox_dim, 3 * self.hidden_size, bias=self.bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / (self.hidden_size**0.5)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)

    def forward(
        self, x: torch.Tensor, hx: torch.Tensor, vel: torch.Tensor
    ) -> torch.Tensor:
        if hx is None:
            hx = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        gate_x = self.input_to_hidden(x)
        gate_h = self.hidden_to_hidden(hx)
        gate_vel = self.vel_to_vel(vel)

        x_reset, x_upd, x_new = gate_x.chunk(3, 1)
        h_reset, h_upd, h_new = gate_h.chunk(3, 1)
        vel_reset, vel_upd, vel_new = gate_vel.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset + vel_reset)
        update_gate = torch.sigmoid(x_upd + h_upd + vel_upd)
        new_gate = torch.tanh(x_new + vel_new + reset_gate.mul(h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy
