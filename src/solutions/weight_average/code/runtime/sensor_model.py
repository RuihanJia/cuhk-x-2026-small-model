"""Exact accepted measured-sensor architecture."""
import torch
from torch import nn
class SensorTemporal(nn.Module):

    def __init__(self):
        super().__init__()
        self.skeleton = nn.Sequential(nn.Conv1d(120, 128, 1), nn.GroupNorm(8, 128), nn.GELU(), nn.Conv1d(128, 128, 5, padding=2, groups=128), nn.GELU())
        self.imu = nn.Sequential(nn.Conv1d(7, 128, 9, stride=4, padding=4), nn.GroupNorm(8, 128), nn.GELU(), nn.Conv1d(128, 128, 5, padding=2, groups=128), nn.GELU())
        self.join = nn.Linear(256, 128)
        self.position = nn.Parameter(torch.zeros(1, 64, 128))
        layer = nn.TransformerEncoderLayer(128, 4, dim_feedforward=256, dropout=0.2, batch_first=True, activation='gelu', norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.2), nn.Linear(128, 40))

    def forward(self, s, i):
        z = torch.cat([self.skeleton(s), self.imu(i)], dim=1).transpose(1, 2)
        z = self.temporal(self.join(z) + self.position)
        return self.head(z.mean(1))
