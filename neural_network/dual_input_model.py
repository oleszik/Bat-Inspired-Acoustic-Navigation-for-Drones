"""
Dual-input model for acoustic obstacle + distance regression.

Inputs:
- Spectrogram image (grayscale 128x128)
- Matched-filter correlation vector (length 512)

Outputs:
- obstacle logit
- predicted distance (meters)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AcousticDualInputRegressionCNN(nn.Module):
    """
    Two-branch network:
    1) 2D CNN branch for spectrogram image
    2) 1D CNN branch for correlation feature vector

    Fused features are used by two heads:
    - obstacle_head: binary logit
    - distance_head: continuous distance in meters
    """

    def __init__(self) -> None:
        super().__init__()

        # Spectrogram branch:
        # Conv2D -> ReLU -> MaxPool (x3), then Flatten -> Linear -> ReLU
        self.spec_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
        )

        # Correlation branch:
        # Conv1D -> ReLU -> MaxPool -> Conv1D -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU
        # Input shape is [B, 512], converted to [B, 1, 512].
        self.corr_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 512 -> 256
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 256 -> 128
            nn.Flatten(),
            nn.Linear(32 * 128, 128),
            nn.ReLU(inplace=True),
        )

        # Fuse both branches and produce task-specific outputs.
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(inplace=True),
        )
        self.obstacle_head = nn.Linear(128, 1)
        self.distance_head = nn.Linear(128, 1)

    def forward(self, spec_image: torch.Tensor, corr_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec_feat = self.spec_branch(spec_image)
        corr_feat = self.corr_branch(corr_vec.unsqueeze(1))
        fused = self.fusion(torch.cat([spec_feat, corr_feat], dim=1))

        obstacle_logit = self.obstacle_head(fused).squeeze(1)
        distance_pred_m = self.distance_head(fused).squeeze(1)
        return obstacle_logit, distance_pred_m
