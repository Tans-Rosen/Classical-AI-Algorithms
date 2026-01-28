"""CNN architecture for Imagenette 10-class image classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ImagenetteCNN(nn.Module):
    """
    Small CNN for Imagenette: two conv blocks (Conv2d + ReLU + MaxPool2d)
    plus an MLP head. Expects 224×224 RGB input.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        spatial = self._conv_output_size(224)
        fc_in = spatial * spatial * 32
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _conv_output_size(size: int) -> int:
        for k, s in [(8, 4), (2, 2), (4, 2), (2, 2)]:
            size = math.floor((size - k) / s) + 1
        return int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
