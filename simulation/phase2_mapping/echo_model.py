from __future__ import annotations

import torch
import torch.nn as nn


class AcousticEchoMapper(nn.Module):
    """Small 1D-CNN multi-head mapper for local patch prediction from echo vectors."""

    def __init__(
        self,
        in_channels: int,
        n_bins: int,
        patch_size: int,
        hidden_dim: int = 256,
        meta_dim: int = 8,
        dropout: float = 0.1,
        use_visibility_head: bool = True,
        use_pose_head: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size
        self.use_visibility_head = use_visibility_head
        self.use_pose_head = use_pose_head

        self.signal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
        )
        self.signal_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, max(32, hidden_dim // 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        fusion_dim = hidden_dim + max(32, hidden_dim // 4)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.occupancy_head = nn.Linear(hidden_dim, self.patch_dim)
        self.wall_head = nn.Linear(hidden_dim, self.patch_dim)
        self.doorway_head = nn.Linear(hidden_dim, self.patch_dim)
        self.free_head = nn.Linear(hidden_dim, self.patch_dim)
        self.visibility_head = nn.Linear(hidden_dim, self.patch_dim) if use_visibility_head else None
        self.pose_head = nn.Linear(hidden_dim, 3) if use_pose_head else None

    def forward(self, signal: torch.Tensor, meta: torch.Tensor) -> dict[str, torch.Tensor]:
        sig_feat = self.signal_proj(self.signal_encoder(signal))
        meta_feat = self.meta_encoder(meta)
        feat = self.fusion(torch.cat([sig_feat, meta_feat], dim=1))

        out = {
            "occupancy_logits": self.occupancy_head(feat),
            "wall_logits": self.wall_head(feat),
            "doorway_logits": self.doorway_head(feat),
            "free_logits": self.free_head(feat),
        }
        if self.visibility_head is not None:
            out["visibility_logits"] = self.visibility_head(feat)
        if self.pose_head is not None:
            out["pose_pred"] = self.pose_head(feat)
        return out
