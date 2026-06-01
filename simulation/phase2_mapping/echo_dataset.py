from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


RAY_MAX_RANGE = 3.0


@dataclass
class DatasetInfo:
    map_names: list[str]
    difficulty_names: list[str]
    action_names: list[str]
    patch_size: int
    n_bins: int
    in_channels: int


class EchoMappingNPZDataset(Dataset):
    """PyTorch dataset wrapper for Phase 2B train/val/test .npz files."""

    def __init__(
        self,
        npz_path: str | Path,
        patch_size: int = 32,
        use_extra_channels: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(npz_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.path}")

        d = np.load(self.path, allow_pickle=False)
        self.echo_multichannel = d["echo_multichannel"].astype(np.float32)
        self.echo_timing = d["echo_timing_vector"].astype(np.float32)
        self.echo_intensity = d["echo_intensity_vector"].astype(np.float32)
        self.scan_dirs = d["scan_directions_rad"].astype(np.float32)

        self.true_pose = d["true_pose"].astype(np.float32)
        self.estimated_pose = d["estimated_pose"].astype(np.float32)
        self.pose_correction = d["pose_correction_target"].astype(np.float32)

        self.map_index = d["map_index"].astype(np.int64)
        self.difficulty_index = d["difficulty_index"].astype(np.int64)
        self.timestep_index = d["timestep_index"].astype(np.int64)
        self.action_index = d["action_index"].astype(np.int64)

        self.occupancy = d["occupancy_patch"].astype(np.float32)
        self.wall = d["wall_patch"].astype(np.float32)
        self.doorway = d["doorway_patch"].astype(np.float32)
        self.free = d["free_space_patch"].astype(np.float32)
        self.visibility = d["visibility_mask_patch"].astype(np.float32)

        self.map_names = [str(x) for x in d["map_names"].tolist()]
        self.difficulty_names = [str(x) for x in d["difficulty_names"].tolist()]
        self.action_names = [str(x) for x in d["action_names"].tolist()]

        self.patch_size = int(d["patch_size"][0]) if "patch_size" in d.files else int(self.occupancy.shape[-1])
        if patch_size != self.patch_size:
            raise ValueError(
                f"Patch-size mismatch for {self.path.name}: requested {patch_size}, dataset has {self.patch_size}"
            )

        self.use_extra_channels = use_extra_channels
        self.n_bins = int(self.echo_multichannel.shape[-1])
        self.signal_channels = int(self.echo_multichannel.shape[1])
        self.in_channels = self.signal_channels * (4 if self.use_extra_channels else 1)

    def __len__(self) -> int:
        return int(self.echo_multichannel.shape[0])

    def _build_signal_input(self, idx: int) -> np.ndarray:
        signal = self.echo_multichannel[idx]  # [5, bins]
        if not self.use_extra_channels:
            return signal

        timing = np.repeat((self.echo_timing[idx][:, None] / RAY_MAX_RANGE), self.n_bins, axis=1)
        intensity = np.repeat(self.echo_intensity[idx][:, None], self.n_bins, axis=1)
        scan_norm = np.repeat(((self.scan_dirs[idx][:, None] + np.pi) / (2.0 * np.pi)), self.n_bins, axis=1)
        signal = np.concatenate([signal, timing.astype(np.float32), intensity.astype(np.float32), scan_norm.astype(np.float32)], axis=0)
        return signal

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        signal = self._build_signal_input(idx)

        meta = np.concatenate(
            [
                self.true_pose[idx],
                self.estimated_pose[idx],
                np.asarray([
                    float(self.timestep_index[idx]) / 1000.0,
                    float(self.action_index[idx]) / max(1.0, float(len(self.action_names) - 1)),
                ], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        sample = {
            "signal": torch.from_numpy(signal),
            "meta": torch.from_numpy(meta),
            "occupancy": torch.from_numpy(self.occupancy[idx].reshape(-1)),
            "wall": torch.from_numpy(self.wall[idx].reshape(-1)),
            "doorway": torch.from_numpy(self.doorway[idx].reshape(-1)),
            "free": torch.from_numpy(self.free[idx].reshape(-1)),
            "visibility": torch.from_numpy(self.visibility[idx].reshape(-1)),
            "pose_correction": torch.from_numpy(self.pose_correction[idx]),
            "map_index": torch.tensor(self.map_index[idx], dtype=torch.long),
            "difficulty_index": torch.tensor(self.difficulty_index[idx], dtype=torch.long),
            "timestep_index": torch.tensor(self.timestep_index[idx], dtype=torch.long),
            "action_index": torch.tensor(self.action_index[idx], dtype=torch.long),
            "echo_timing": torch.from_numpy(self.echo_timing[idx]),
            "echo_intensity": torch.from_numpy(self.echo_intensity[idx]),
        }
        return sample

    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            map_names=self.map_names,
            difficulty_names=self.difficulty_names,
            action_names=self.action_names,
            patch_size=self.patch_size,
            n_bins=self.n_bins,
            in_channels=self.in_channels,
        )


def build_weighted_sampling_weights(dataset: EchoMappingNPZDataset, doorway_boost: float = 4.0, wall_boost: float = 1.5) -> np.ndarray:
    """Higher sampling weights for doorway/wall-positive samples."""
    door_has = (dataset.doorway.reshape(len(dataset), -1).sum(axis=1) > 0).astype(np.float32)
    wall_has = (dataset.wall.reshape(len(dataset), -1).sum(axis=1) > 0).astype(np.float32)
    weights = np.ones(len(dataset), dtype=np.float32)
    weights += doorway_boost * door_has
    weights += wall_boost * wall_has
    return weights


def compute_pos_weight_from_ratio(ratio: float, min_w: float = 1.0, max_w: float = 200.0) -> float:
    ratio = float(np.clip(ratio, 1e-6, 1.0 - 1e-6))
    w = (1.0 - ratio) / ratio
    return float(np.clip(w, min_w, max_w))
