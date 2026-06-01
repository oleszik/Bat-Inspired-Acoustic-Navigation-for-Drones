"""
Two-head CNN for acoustic obstacle + distance regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AcousticRegressionCNN(nn.Module):
    """
    Input:
      - grayscale spectrogram image [B, 1, 128, 128]

    Outputs:
      - obstacle_logit: [B] (binary obstacle presence logit)
      - distance_pred_m: [B] (continuous distance prediction in meters)

    Why two heads:
    - Obstacle presence and distance are related but different tasks.
    - A shared feature extractor learns common acoustic patterns,
      while separate heads specialize for classification vs regression.
    """

    def __init__(self) -> None:
        super().__init__()

        # Simple CNN feature extractor:
        # Conv2D -> ReLU -> MaxPool (x3)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # 128x128 -> 16x16 after 3 max-pool layers.
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
        )

        # Head A: binary obstacle logit.
        self.obstacle_head = nn.Linear(128, 1)

        # Head B: continuous distance in meters.
        self.distance_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        feat = self.shared_mlp(feat)
        obstacle_logit = self.obstacle_head(feat).squeeze(1)
        distance_pred_m = self.distance_head(feat).squeeze(1)
        return obstacle_logit, distance_pred_m
