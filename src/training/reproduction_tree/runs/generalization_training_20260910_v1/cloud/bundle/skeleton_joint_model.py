"""A controlled joint encoder with optional learned, semantic-free joint mixing."""
import torch
from torch import nn
from skeleton_joint_base import SensorTemporal as BaseSensorTemporal


class JointEncoder(nn.Module):
    def __init__(self, graph=False):
        super().__init__()
        self.graph = graph
        self.input = nn.Conv2d(8, 64, 1)
        self.joint_position = nn.Parameter(torch.zeros(1, 64, 1, 17))
        adjacency = torch.full((17, 17), -4.)
        adjacency.fill_diagonal_(4.)
        self.adjacency = nn.Parameter(adjacency, requires_grad=graph)
        self.pointwise = nn.Sequential(nn.GroupNorm(8, 64), nn.GELU(),
                                       nn.Conv2d(64, 64, 1), nn.GELU())
        self.output = nn.Sequential(nn.Conv1d(64, 128, 1), nn.GroupNorm(8, 128), nn.GELU(),
                                    nn.Conv1d(128, 128, 5, padding=2, groups=128), nn.GELU())

    def forward(self, s):
        b, _, t = s.shape
        position = s[:, :51].reshape(b, 17, 3, t).permute(0, 2, 3, 1)
        motion = s[:, 51:102].reshape(b, 17, 3, t).permute(0, 2, 3, 1)
        valid = s[:, 102:119].transpose(1, 2).unsqueeze(1)
        count = s[:, 119:120].unsqueeze(-1).expand(-1, -1, -1, 17)
        x = self.input(torch.cat([position, motion, valid, count], 1)) + self.joint_position
        if self.graph:
            # Only valid source joints contribute; empty frames produce zero messages.
            weights = self.adjacency.softmax(1)[None, None] * valid.permute(0, 2, 1, 3)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            message = torch.einsum('btij,bctj->bcti', weights, x)
        else:
            message = x * valid
        x = self.pointwise(x + message) * valid
        pooled = x.sum(-1) / valid.sum(-1).clamp_min(1.)
        return self.output(pooled)


class SensorTemporal(BaseSensorTemporal):
    def __init__(self, graph=False):
        super().__init__()
        self.skeleton = JointEncoder(graph)
