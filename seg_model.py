import torch
import torch.nn as nn

class SegModel(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
