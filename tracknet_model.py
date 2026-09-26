"""TrackNet V1 architecture for 3-frame tennis-ball heatmap inference.

Architecture follows the public TrackNet PyTorch implementation and the
TrackNet paper (Huang et al., 2019). Model weights are downloaded separately.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_size = value.size(0)
        value = self.conv2(self.conv1(value))
        value = self.pool1(value)
        value = self.conv4(self.conv3(value))
        value = self.pool2(value)
        value = self.conv7(self.conv6(self.conv5(value)))
        value = self.pool3(value)
        value = self.conv10(self.conv9(self.conv8(value)))
        value = self.ups1(value)
        value = self.conv13(self.conv12(self.conv11(value)))
        value = self.ups2(value)
        value = self.conv15(self.conv14(value))
        value = self.ups3(value)
        value = self.conv18(self.conv17(self.conv16(value)))
        return value.reshape(batch_size, self.out_channels, -1)
