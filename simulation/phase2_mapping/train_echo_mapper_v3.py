from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from .echo_dataset import EchoMappingNPZDataset
except ImportError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.phase2_mapping.echo_dataset import EchoMappingNPZDataset  # type: ignore


NON_DOORWAY_MAPS = {"empty_room", "corridor", "single_block", "cluttered_room"}


class EchoMapperV2(nn.Module):
    """Sharper multi-head decoder for local mapping patches."""

    def __init__(
        self,
        in_channels: int,
        n_bins: int,
        patch_size: int,
        hidden_dim: int = 320,
        meta_dim: int = 8,
        dropout: float = 0.12,
        use_visibility_head: bool = True,
        use_pose_head: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.use_visibility_head = use_visibility_head
        self.use_pose_head = use_pose_head

        self.signal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(16),
        )
        self.signal_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160 * 16, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + 96, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.to_spatial = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        self.shared_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.occ_head = nn.Conv2d(32, 1, kernel_size=1)
        self.wall_head = nn.Conv2d(32, 1, kernel_size=1)
        self.door_head = nn.Conv2d(32, 1, kernel_size=1)
        self.free_head = nn.Conv2d(32, 1, kernel_size=1)
        self.vis_head = nn.Conv2d(32, 1, kernel_size=1) if use_visibility_head else None
        self.pose_head = nn.Linear(hidden_dim, 3) if use_pose_head else None

    def forward(self, signal: torch.Tensor, meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        sig = self.signal_proj(self.signal_encoder(signal))
        meta_f = self.meta_proj(meta)
        feat = self.fuse(torch.cat([sig, meta_f], dim=1))

        spatial = self.to_spatial(feat).view(signal.shape[0], 64, 8, 8)
        dec = self.shared_decoder(spatial)

        out: Dict[str, torch.Tensor] = {
            "occupancy_logits": self.occ_head(dec).flatten(1),
            "wall_logits": self.wall_head(dec).flatten(1),
            "doorway_logits": self.door_head(dec).flatten(1),
            "free_logits": self.free_head(dec).flatten(1),
        }
        if self.vis_head is not None:
            out["visibility_logits"] = self.vis_head(dec).flatten(1)
        if self.pose_head is not None:
            out["pose_pred"] = self.pose_head(feat)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2C.2 - doorway false-positive control (v3).")
    parser.add_argument("--data-dir", type=str, default="datasets/phase2_echo_mapping")
    parser.add_argument("--output-dir", type=str, default="runs/phase2_echo_mapper_v3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-plots", action="store_true")

    parser.add_argument("--doorway-focal-gamma", type=float, default=2.0)
    parser.add_argument("--doorway-hard-neg-weight", type=float, default=0.45)
    parser.add_argument("--doorway-negative-weight", type=float, default=2.0)
    parser.add_argument("--doorway-fp-penalty", type=float, default=1.0)
    parser.add_argument("--doorway-fp-lambda", type=float, default=0.5)
    parser.add_argument("--use-doorway-gating", type=int, default=1)
    parser.add_argument("--use-visibility-head", action="store_true", default=True)
    parser.add_argument("--use-pose-head", action="store_true", default=True)

    parser.add_argument("--oversample-doorway-positive-boost", type=float, default=4.5)
    parser.add_argument("--oversample-hard-negative-boost", type=float, default=1.8)
    parser.add_argument("--oversample-corridor-boost", type=float, default=1.25)
    parser.add_argument("--oversample-clutter-boost", type=float, default=1.25)

    parser.add_argument(
        "--threshold-candidates",
        type=str,
        default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--prediction-plot-count", type=int, default=28)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def compute_pos_weight(ratio: float, min_w: float, max_w: float) -> float:
    ratio = float(np.clip(ratio, 1e-6, 1.0 - 1e-6))
    w = (1.0 - ratio) / ratio
    return float(np.clip(w, min_w, max_w))


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = torch.pow(1.0 - pt, gamma)
    return (focal * bce).mean()


def doorway_hard_negative_penalty(logits: torch.Tensor, targets: torch.Tensor, map_index: torch.Tensor, doorway_map_idx: int) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    per_sample_prob = probs.mean(dim=1)
    no_door_target = (targets.sum(dim=1) <= 0.5)
    non_door_maps = (map_index != doorway_map_idx)
    mask = no_door_target & non_door_maps
    if mask.any():
        return per_sample_prob[mask].mean()
    return torch.zeros((), device=logits.device)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def binary_stats(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    tp = float(np.logical_and(pred, tgt).sum())
    fp = float(np.logical_and(pred, np.logical_not(tgt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), tgt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(tgt)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    iou = tp / max(1.0, tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "iou": iou,
    }


def _shift_no_wrap(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(a)
    y_src_start = max(0, -dy)
    y_src_end = a.shape[1] - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end = a.shape[2] - max(0, dx)

    y_dst_start = max(0, dy)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)

    if y_src_end > y_src_start and x_src_end > x_src_start:
        out[:, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = a[:, y_src_start:y_src_end, x_src_start:x_src_end]
    return out


def apply_doorway_structural_gating(
    door_prob_2d: np.ndarray,
    wall_prob_2d: np.ndarray,
    free_prob_2d: np.ndarray,
    occ_prob_2d: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    n = door_prob_2d.shape[0]
    d = door_prob_2d.reshape(n, patch_size, patch_size)
    w = wall_prob_2d.reshape(n, patch_size, patch_size)
    f = free_prob_2d.reshape(n, patch_size, patch_size)
    o = occ_prob_2d.reshape(n, patch_size, patch_size)

    # Wall-gap-wall evidence across both principal axes.
    h1 = np.minimum(_shift_no_wrap(w, 0, -1), _shift_no_wrap(w, 0, 1))
    h2 = np.minimum(_shift_no_wrap(w, 0, -2), _shift_no_wrap(w, 0, 2))
    v1 = np.minimum(_shift_no_wrap(w, -1, 0), _shift_no_wrap(w, 1, 0))
    v2 = np.minimum(_shift_no_wrap(w, -2, 0), _shift_no_wrap(w, 2, 0))
    wall_support = np.maximum.reduce([h1, h2, v1, v2])

    neigh = [
        _shift_no_wrap(w, -1, -1),
        _shift_no_wrap(w, -1, 0),
        _shift_no_wrap(w, -1, 1),
        _shift_no_wrap(w, 0, -1),
        _shift_no_wrap(w, 0, 1),
        _shift_no_wrap(w, 1, -1),
        _shift_no_wrap(w, 1, 0),
        _shift_no_wrap(w, 1, 1),
    ]
    local_wall = np.mean(np.stack(neigh, axis=0), axis=0)

    # Gate high in free passage with opposing side wall support; suppress open empty areas.
    support_gate = np.clip((wall_support - 0.25) / 0.45, 0.0, 1.0)
    free_gate = np.clip((f - 0.35) / 0.45, 0.0, 1.0)
    occ_gate = np.clip((0.65 - o) / 0.65, 0.0, 1.0)
    gated_strength = support_gate * free_gate * occ_gate

    gated = d * (0.10 + 0.90 * gated_strength)

    open_empty = (f > 0.78) & (o < 0.20) & (local_wall < 0.12)
    weak_structure = local_wall < 0.08
    isolated_object_edge = (o > 0.55) & (wall_support < 0.20)
    gated[open_empty] *= 0.08
    gated[weak_structure] *= 0.05
    gated[isolated_object_edge] *= 0.25

    return np.clip(gated, 0.0, 1.0).reshape(door_prob_2d.shape[0], -1)


def non_doorway_fp_rates(
    door_pred_flat: np.ndarray,
    door_gt_flat: np.ndarray,
    map_idx: np.ndarray,
    map_names: List[str],
) -> Tuple[Dict[str, float], float]:
    fpr_by_map: Dict[str, float] = {}
    vals = []
    for name in NON_DOORWAY_MAPS:
        if name not in map_names:
            continue
        idx = map_names.index(name)
        mask_map = map_idx == idx
        if not np.any(mask_map):
            fpr_by_map[name] = 0.0
            vals.append(0.0)
            continue
        gt_neg = door_gt_flat[mask_map].sum(axis=1) == 0
        if np.any(gt_neg):
            pred_pos = door_pred_flat[mask_map][gt_neg].any(axis=1)
            fpr = float(pred_pos.mean())
        else:
            fpr = 0.0
        fpr_by_map[name] = fpr
        vals.append(fpr)
    avg_fpr = float(np.mean(vals)) if vals else 0.0
    return fpr_by_map, avg_fpr


def scan_best_threshold(prob: np.ndarray, tgt: np.ndarray, thresholds: Iterable[float], metric: str) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    all_stats: Dict[str, Dict[str, float]] = {}
    best_t = 0.5
    best_val = -1.0
    best_stats: Dict[str, float] = {}

    for t in thresholds:
        pred = prob >= t
        stats = binary_stats(pred, tgt)
        all_stats[str(t)] = stats
        val = stats[metric]
        if val > best_val + 1e-12:
            best_val = val
            best_t = float(t)
            best_stats = stats
    return best_t, best_stats, all_stats


def scan_best_doorway_threshold(
    door_prob: np.ndarray,
    door_t: np.ndarray,
    map_idx: np.ndarray,
    map_names: List[str],
    thresholds: Iterable[float],
    fp_lambda: float,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    all_stats: Dict[str, Dict[str, float]] = {}
    best_t = 0.5
    best_score = -1e9
    best = {}

    for t in thresholds:
        pred = door_prob >= t
        stats = binary_stats(pred, door_t)
        fpr_by_map, avg_fpr = non_doorway_fp_rates(pred, door_t, map_idx, map_names)
        score = float(stats["f1"] - fp_lambda * avg_fpr)
        row = {
            **stats,
            "avg_non_doorway_fp_rate": avg_fpr,
            "combined_score": score,
            "fp_empty_room": float(fpr_by_map.get("empty_room", 0.0)),
            "fp_corridor": float(fpr_by_map.get("corridor", 0.0)),
            "fp_single_block": float(fpr_by_map.get("single_block", 0.0)),
            "fp_cluttered_room": float(fpr_by_map.get("cluttered_room", 0.0)),
        }
        all_stats[str(t)] = row
        if score > best_score:
            best_score = score
            best_t = float(t)
            best = row

    return best_t, best, all_stats


def evaluate_losses_and_collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    wall_pos_weight: torch.Tensor,
    door_pos_weight: torch.Tensor,
    doorway_focal_gamma: float,
    doorway_hard_neg_weight: float,
    doorway_map_idx: int,
    use_visibility_head: bool,
    use_pose_head: bool,
    use_doorway_gating: bool,
    patch_size: int,
) -> Tuple[float, Dict[str, np.ndarray], Dict[str, float]]:
    model.eval()

    sums = {
        "total": 0.0,
        "occupancy": 0.0,
        "wall": 0.0,
        "doorway": 0.0,
        "free": 0.0,
        "visibility": 0.0,
        "pose": 0.0,
        "door_hard_neg": 0.0,
    }
    nb = 0

    occ_probs = []
    wall_probs = []
    door_probs = []
    free_probs = []

    occ_t = []
    wall_t = []
    door_t = []
    free_t = []

    map_idx = []
    diff_idx = []

    pose_preds = []
    pose_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch["signal"], batch["meta"])

            loss_occ = F.binary_cross_entropy_with_logits(out["occupancy_logits"], batch["occupancy"])
            loss_wall_bce = F.binary_cross_entropy_with_logits(
                out["wall_logits"], batch["wall"], pos_weight=wall_pos_weight
            )
            loss_wall = 0.7 * loss_wall_bce + 0.3 * dice_loss_from_logits(out["wall_logits"], batch["wall"])

            loss_door_focal = focal_bce_with_logits(
                out["doorway_logits"], batch["doorway"], pos_weight=door_pos_weight, gamma=doorway_focal_gamma
            )
            loss_door_dice = dice_loss_from_logits(out["doorway_logits"], batch["doorway"])
            loss_door_hard_neg = doorway_hard_negative_penalty(
                out["doorway_logits"], batch["doorway"], batch["map_index"], doorway_map_idx
            )
            loss_door = 0.65 * loss_door_focal + 0.35 * loss_door_dice + doorway_hard_neg_weight * loss_door_hard_neg

            loss_free = F.binary_cross_entropy_with_logits(out["free_logits"], batch["free"])
            loss_occ_dice = dice_loss_from_logits(out["occupancy_logits"], batch["occupancy"])
            loss_occ = 0.7 * loss_occ + 0.3 * loss_occ_dice

            if use_visibility_head and "visibility_logits" in out:
                loss_vis = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["visibility"])
            else:
                loss_vis = torch.zeros((), device=device)

            if use_pose_head and "pose_pred" in out:
                loss_pose = F.mse_loss(out["pose_pred"], batch["pose_correction"])
            else:
                loss_pose = torch.zeros((), device=device)

            total = (
                1.0 * loss_occ
                + 1.0 * loss_wall
                + 1.55 * loss_door
                + 0.8 * loss_free
                + 0.2 * loss_vis
                + 0.1 * loss_pose
            )

            sums["total"] += float(total.item())
            sums["occupancy"] += float(loss_occ.item())
            sums["wall"] += float(loss_wall.item())
            sums["doorway"] += float(loss_door.item())
            sums["free"] += float(loss_free.item())
            sums["visibility"] += float(loss_vis.item())
            sums["pose"] += float(loss_pose.item())
            sums["door_hard_neg"] += float(loss_door_hard_neg.item())
            nb += 1

            occ_probs.append(torch.sigmoid(out["occupancy_logits"]).cpu().numpy())
            wall_probs.append(torch.sigmoid(out["wall_logits"]).cpu().numpy())
            door_probs.append(torch.sigmoid(out["doorway_logits"]).cpu().numpy())
            free_probs.append(torch.sigmoid(out["free_logits"]).cpu().numpy())

            occ_t.append(batch["occupancy"].cpu().numpy())
            wall_t.append(batch["wall"].cpu().numpy())
            door_t.append(batch["doorway"].cpu().numpy())
            free_t.append(batch["free"].cpu().numpy())

            map_idx.append(batch["map_index"].cpu().numpy())
            diff_idx.append(batch["difficulty_index"].cpu().numpy())

            if use_pose_head and "pose_pred" in out:
                pose_preds.append(out["pose_pred"].cpu().numpy())
                pose_targets.append(batch["pose_correction"].cpu().numpy())

    arrays = {
        "occ_prob": np.concatenate(occ_probs, axis=0),
        "wall_prob": np.concatenate(wall_probs, axis=0),
        "door_prob": np.concatenate(door_probs, axis=0),
        "free_prob": np.concatenate(free_probs, axis=0),
        "occ_t": np.concatenate(occ_t, axis=0) >= 0.5,
        "wall_t": np.concatenate(wall_t, axis=0) >= 0.5,
        "door_t": np.concatenate(door_t, axis=0) >= 0.5,
        "free_t": np.concatenate(free_t, axis=0) >= 0.5,
        "map_idx": np.concatenate(map_idx, axis=0),
        "diff_idx": np.concatenate(diff_idx, axis=0),
    }

    if use_doorway_gating:
        arrays["door_prob_gated"] = apply_doorway_structural_gating(
            door_prob_2d=arrays["door_prob"],
            wall_prob_2d=arrays["wall_prob"],
            free_prob_2d=arrays["free_prob"],
            occ_prob_2d=arrays["occ_prob"],
            patch_size=patch_size,
        )

    if use_pose_head and pose_preds:
        pp = np.concatenate(pose_preds, axis=0)
        pt = np.concatenate(pose_targets, axis=0)
        pose_rmse = float(np.sqrt(np.mean((pp - pt) ** 2)))
    else:
        pose_rmse = 0.0

    loss_metrics = {k + "_loss": float(v / max(1, nb)) for k, v in sums.items()}
    loss_metrics["pose_correction_rmse"] = pose_rmse
    return float(sums["total"] / max(1, nb)), arrays, loss_metrics


def metrics_at_thresholds(
    arrays: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    map_names: List[str],
    difficulty_names: List[str],
    use_doorway_gating: bool,
) -> Dict[str, object]:
    occ_pred = arrays["occ_prob"] >= thresholds["occupancy"]
    wall_pred = arrays["wall_prob"] >= thresholds["wall"]
    door_key = "door_prob_gated" if use_doorway_gating and ("door_prob_gated" in arrays) else "door_prob"
    door_pred = arrays[door_key] >= thresholds["doorway"]
    free_pred = arrays["free_prob"] >= thresholds["free"]

    overall_occ = binary_stats(occ_pred, arrays["occ_t"])
    overall_wall = binary_stats(wall_pred, arrays["wall_t"])
    overall_door = binary_stats(door_pred, arrays["door_t"])
    overall_free = binary_stats(free_pred, arrays["free_t"])

    out_overall = {
        "occupancy_accuracy": overall_occ["accuracy"],
        "occupancy_iou": overall_occ["iou"],
        "wall_precision": overall_wall["precision"],
        "wall_recall": overall_wall["recall"],
        "wall_f1": overall_wall["f1"],
        "doorway_precision": overall_door["precision"],
        "doorway_recall": overall_door["recall"],
        "doorway_f1": overall_door["f1"],
        "free_space_accuracy": overall_free["accuracy"],
        "mean_patch_accuracy": float(
            np.mean([overall_occ["accuracy"], overall_wall["accuracy"], overall_door["accuracy"], overall_free["accuracy"]])
        ),
    }

    by_map: Dict[str, Dict[str, float]] = {}
    by_diff: Dict[str, Dict[str, float]] = {}

    for i, name in enumerate(map_names):
        mask = arrays["map_idx"] == i
        if mask.any():
            occ = binary_stats(occ_pred[mask], arrays["occ_t"][mask])
            wall = binary_stats(wall_pred[mask], arrays["wall_t"][mask])
            door = binary_stats(door_pred[mask], arrays["door_t"][mask])
            by_map[name] = {
                "occupancy_accuracy": occ["accuracy"],
                "occupancy_iou": occ["iou"],
                "wall_precision": wall["precision"],
                "wall_recall": wall["recall"],
                "wall_f1": wall["f1"],
                "doorway_precision": door["precision"],
                "doorway_recall": door["recall"],
                "doorway_f1": door["f1"],
            }

    for i, name in enumerate(difficulty_names):
        mask = arrays["diff_idx"] == i
        if mask.any():
            occ = binary_stats(occ_pred[mask], arrays["occ_t"][mask])
            wall = binary_stats(wall_pred[mask], arrays["wall_t"][mask])
            door = binary_stats(door_pred[mask], arrays["door_t"][mask])
            by_diff[name] = {
                "occupancy_accuracy": occ["accuracy"],
                "occupancy_iou": occ["iou"],
                "wall_precision": wall["precision"],
                "wall_recall": wall["recall"],
                "wall_f1": wall["f1"],
                "doorway_precision": door["precision"],
                "doorway_recall": door["recall"],
                "doorway_f1": door["f1"],
            }

    # Doorway false-positive rates on non-doorway maps (patch-level activation)
    fpr, _ = non_doorway_fp_rates(door_pred, arrays["door_t"], arrays["map_idx"], map_names)

    if "doorway" in map_names:
        didx = map_names.index("doorway")
        door_mask = arrays["map_idx"] == didx
        gt_pos = arrays["door_t"][door_mask].sum(axis=1) > 0
        if gt_pos.any():
            pred_pos = door_pred[door_mask][gt_pos].any(axis=1)
            tpr_doorway_map = float(pred_pos.mean())
        else:
            tpr_doorway_map = 0.0
    else:
        tpr_doorway_map = 0.0

    out = {
        "overall": out_overall,
        "by_map": by_map,
        "by_difficulty": by_diff,
        "doorway_false_positive_rate_empty_room": fpr.get("empty_room", 0.0),
        "doorway_false_positive_rate_corridor": fpr.get("corridor", 0.0),
        "doorway_false_positive_rate_single_block": fpr.get("single_block", 0.0),
        "doorway_false_positive_rate_cluttered_room": fpr.get("cluttered_room", 0.0),
        "doorway_true_positive_rate_doorway_map": tpr_doorway_map,
    }

    if "corridor" in by_map:
        out["corridor_performance"] = by_map["corridor"]
    if "doorway" in by_map:
        out["doorway_map_performance"] = by_map["doorway"]
    if "hard_noise" in by_diff:
        out["hard_noise_performance"] = by_diff["hard_noise"]
    if "cluttered_room" in by_map:
        out["cluttered_room_performance"] = by_map["cluttered_room"]
    return out


def save_prediction_plots(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    out_dir: Path,
    thresholds: Dict[str, float],
    max_plots: int,
    use_doorway_gating: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch["signal"], batch["meta"])

            occ_prob = torch.sigmoid(out["occupancy_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)
            wall_prob = torch.sigmoid(out["wall_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)
            door_prob_raw = torch.sigmoid(out["doorway_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)
            free_prob = torch.sigmoid(out["free_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)
            door_prob_gated = apply_doorway_structural_gating(
                door_prob_2d=door_prob_raw.reshape(door_prob_raw.shape[0], -1),
                wall_prob_2d=wall_prob.reshape(wall_prob.shape[0], -1),
                free_prob_2d=free_prob.reshape(free_prob.shape[0], -1),
                occ_prob_2d=occ_prob.reshape(occ_prob.shape[0], -1),
                patch_size=patch_size,
            ).reshape(-1, patch_size, patch_size)
            door_prob = door_prob_gated if use_doorway_gating else door_prob_raw

            occ_bin = (occ_prob >= thresholds["occupancy"]).astype(np.float32)
            wall_bin = (wall_prob >= thresholds["wall"]).astype(np.float32)
            door_bin = (door_prob >= thresholds["doorway"]).astype(np.float32)
            free_bin = (free_prob >= thresholds["free"]).astype(np.float32)

            occ_gt = batch["occupancy"].cpu().numpy().reshape(-1, patch_size, patch_size)
            wall_gt = batch["wall"].cpu().numpy().reshape(-1, patch_size, patch_size)
            door_gt = batch["doorway"].cpu().numpy().reshape(-1, patch_size, patch_size)
            free_gt = batch["free"].cpu().numpy().reshape(-1, patch_size, patch_size)
            timing = batch["echo_timing"].cpu().numpy()
            inten = batch["echo_intensity"].cpu().numpy()

            for i in range(occ_prob.shape[0]):
                if count >= max_plots:
                    return

                fig, ax = plt.subplots(4, 4, figsize=(12, 11), dpi=120)
                axs = ax.flatten()

                x = np.arange(timing.shape[1])
                axs[0].plot(x, timing[i], marker="o", label="timing(m)")
                axs[0].plot(x, inten[i], marker="s", label="intensity")
                axs[0].set_title("Input Echo")
                axs[0].legend(fontsize=7)

                axs[1].imshow(occ_gt[i], cmap="gray_r", vmin=0, vmax=1)
                axs[1].set_title("GT Occupancy")
                axs[2].imshow(occ_prob[i], cmap="gray_r", vmin=0, vmax=1)
                axs[2].set_title("Pred Occ Prob")
                axs[3].imshow(occ_bin[i], cmap="gray_r", vmin=0, vmax=1)
                axs[3].set_title(f"Pred Occ Bin@{thresholds['occupancy']:.2f}")

                axs[4].imshow(wall_gt[i], cmap="magma", vmin=0, vmax=1)
                axs[4].set_title("GT Wall")
                axs[5].imshow(wall_prob[i], cmap="magma", vmin=0, vmax=1)
                axs[5].set_title("Pred Wall Prob")
                axs[6].imshow(wall_bin[i], cmap="magma", vmin=0, vmax=1)
                axs[6].set_title(f"Pred Wall Bin@{thresholds['wall']:.2f}")

                axs[7].imshow(door_gt[i], cmap="viridis", vmin=0, vmax=1)
                axs[7].set_title("GT Doorway")
                axs[8].imshow(door_prob_raw[i], cmap="viridis", vmin=0, vmax=1)
                axs[8].set_title("Pred Door Raw Prob")
                axs[9].imshow(door_bin[i], cmap="viridis", vmin=0, vmax=1)
                axs[9].set_title(f"Pred Door Bin@{thresholds['doorway']:.2f}")

                axs[10].imshow(free_gt[i], cmap="Blues", vmin=0, vmax=1)
                axs[10].set_title("GT Free")
                axs[11].imshow(free_prob[i], cmap="Blues", vmin=0, vmax=1)
                axs[11].set_title("Pred Free Prob")
                axs[12].imshow(free_bin[i], cmap="Blues", vmin=0, vmax=1)
                axs[12].set_title(f"Pred Free Bin@{thresholds['free']:.2f}")

                axs[13].imshow(door_prob_gated[i], cmap="viridis", vmin=0, vmax=1)
                axs[13].set_title("Pred Door Gated Prob")
                axs[14].axis("off")
                axs[14].text(0.0, 0.8, f"sample={count}", fontsize=10)
                axs[14].text(0.0, 0.6, f"door_gt_px={int(door_gt[i].sum())}", fontsize=10)
                axs[14].text(0.0, 0.4, f"door_pred_px={int(door_bin[i].sum())}", fontsize=10)
                axs[14].text(0.0, 0.2, f"gated={int(use_doorway_gating)}", fontsize=10)
                axs[15].axis("off")

                for a in axs:
                    a.set_xticks([])
                    a.set_yticks([])

                fig.tight_layout()
                fig.savefig(out_dir / f"prediction_sample_{count:04d}.png")
                plt.close(fig)
                count += 1


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    wall_pos_weight: torch.Tensor,
    door_pos_weight: torch.Tensor,
    doorway_focal_gamma: float,
    doorway_hard_neg_weight: float,
    doorway_negative_weight: float,
    doorway_fp_penalty: float,
    doorway_map_idx: int,
    use_visibility_head: bool,
    use_pose_head: bool,
) -> Dict[str, float]:
    model.train()

    sums = {
        "total": 0.0,
        "occupancy": 0.0,
        "wall": 0.0,
        "doorway": 0.0,
        "free": 0.0,
        "visibility": 0.0,
        "pose": 0.0,
        "door_hard_neg": 0.0,
    }
    nb = 0

    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["signal"], batch["meta"])

        loss_occ_bce = F.binary_cross_entropy_with_logits(out["occupancy_logits"], batch["occupancy"])
        loss_occ = 0.7 * loss_occ_bce + 0.3 * dice_loss_from_logits(out["occupancy_logits"], batch["occupancy"])

        loss_wall_bce = F.binary_cross_entropy_with_logits(out["wall_logits"], batch["wall"], pos_weight=wall_pos_weight)
        loss_wall = 0.7 * loss_wall_bce + 0.3 * dice_loss_from_logits(out["wall_logits"], batch["wall"])

        loss_door_focal = focal_bce_with_logits(
            out["doorway_logits"], batch["doorway"], pos_weight=door_pos_weight, gamma=doorway_focal_gamma
        )
        loss_door_dice = dice_loss_from_logits(out["doorway_logits"], batch["doorway"])
        loss_door_hn = doorway_hard_negative_penalty(out["doorway_logits"], batch["doorway"], batch["map_index"], doorway_map_idx)
        loss_door = (
            doorway_negative_weight * (0.65 * loss_door_focal + 0.35 * loss_door_dice)
            + doorway_hard_neg_weight * doorway_fp_penalty * loss_door_hn
        )

        loss_free = F.binary_cross_entropy_with_logits(out["free_logits"], batch["free"])

        if use_visibility_head and "visibility_logits" in out:
            loss_vis = F.binary_cross_entropy_with_logits(out["visibility_logits"], batch["visibility"])
        else:
            loss_vis = torch.zeros((), device=device)

        if use_pose_head and "pose_pred" in out:
            loss_pose = F.mse_loss(out["pose_pred"], batch["pose_correction"])
        else:
            loss_pose = torch.zeros((), device=device)

        total = 1.0 * loss_occ + 1.0 * loss_wall + 1.55 * loss_door + 0.8 * loss_free + 0.2 * loss_vis + 0.1 * loss_pose

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        sums["total"] += float(total.item())
        sums["occupancy"] += float(loss_occ.item())
        sums["wall"] += float(loss_wall.item())
        sums["doorway"] += float(loss_door.item())
        sums["free"] += float(loss_free.item())
        sums["visibility"] += float(loss_vis.item())
        sums["pose"] += float(loss_pose.item())
        sums["door_hard_neg"] += float(loss_door_hn.item())
        nb += 1

    return {k + "_loss": float(v / max(1, nb)) for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    use_doorway_gating = bool(args.use_doorway_gating)

    thresholds = [float(x.strip()) for x in args.threshold_candidates.split(",") if x.strip()]
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    thresholds = sorted(set(thresholds))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = EchoMappingNPZDataset(data_dir / "train.npz", patch_size=args.patch_size)
    val_ds = EchoMappingNPZDataset(data_dir / "val.npz", patch_size=args.patch_size)
    test_ds = EchoMappingNPZDataset(data_dir / "test.npz", patch_size=args.patch_size)

    map_names = train_ds.map_names
    difficulty_names = train_ds.difficulty_names
    if "doorway" in map_names:
        doorway_map_idx = map_names.index("doorway")
    else:
        doorway_map_idx = -1

    # Balanced training sampler
    door_pos = (train_ds.doorway.reshape(len(train_ds), -1).sum(axis=1) > 0).astype(np.float32)
    map_idx_np = train_ds.map_index
    weights = np.ones(len(train_ds), dtype=np.float32)
    weights += args.oversample_doorway_positive_boost * door_pos

    for i, map_name in enumerate(map_names):
        mask = map_idx_np == i
        if map_name in NON_DOORWAY_MAPS:
            neg_mask = mask & (door_pos == 0)
            weights[neg_mask] *= args.oversample_hard_negative_boost
        if map_name == "corridor":
            weights[mask] *= args.oversample_corridor_boost
        if map_name == "cluttered_room":
            weights[mask] *= args.oversample_clutter_boost

    sampler = WeightedRandomSampler(weights=torch.from_numpy(weights), num_samples=len(train_ds), replacement=True)

    device = resolve_device(args.device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    wall_ratio = float(train_ds.wall.mean())
    door_ratio = float(train_ds.doorway.mean())
    wall_pos_weight_val = compute_pos_weight(wall_ratio, min_w=1.0, max_w=100.0)
    door_pos_weight_val = compute_pos_weight(door_ratio, min_w=5.0, max_w=300.0)

    wall_pos_weight = torch.tensor([wall_pos_weight_val], dtype=torch.float32, device=device)
    door_pos_weight = torch.tensor([door_pos_weight_val], dtype=torch.float32, device=device)

    model = EchoMapperV2(
        in_channels=train_ds.in_channels,
        n_bins=train_ds.n_bins,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        meta_dim=8,
        use_visibility_head=args.use_visibility_head,
        use_pose_head=args.use_pose_head,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = {
        "phase": "Phase 2C.2",
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patch_size": args.patch_size,
        "hidden_dim": args.hidden_dim,
        "device": str(device),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "use_visibility_head": args.use_visibility_head,
        "use_pose_head": args.use_pose_head,
        "doorway_focal_gamma": args.doorway_focal_gamma,
        "doorway_hard_neg_weight": args.doorway_hard_neg_weight,
        "doorway_negative_weight": args.doorway_negative_weight,
        "doorway_fp_penalty": args.doorway_fp_penalty,
        "doorway_fp_lambda": args.doorway_fp_lambda,
        "use_doorway_gating": use_doorway_gating,
        "threshold_candidates": thresholds,
        "wall_positive_ratio": wall_ratio,
        "doorway_positive_ratio": door_ratio,
        "wall_pos_weight": wall_pos_weight_val,
        "door_pos_weight": door_pos_weight_val,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val_loss = float("inf")
    best_epoch = -1
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            wall_pos_weight=wall_pos_weight,
            door_pos_weight=door_pos_weight,
            doorway_focal_gamma=args.doorway_focal_gamma,
            doorway_hard_neg_weight=args.doorway_hard_neg_weight,
            doorway_negative_weight=args.doorway_negative_weight,
            doorway_fp_penalty=args.doorway_fp_penalty,
            doorway_map_idx=doorway_map_idx,
            use_visibility_head=args.use_visibility_head,
            use_pose_head=args.use_pose_head,
        )

        val_total, val_arrays, val_loss_metrics = evaluate_losses_and_collect(
            model=model,
            loader=val_loader,
            device=device,
            wall_pos_weight=wall_pos_weight,
            door_pos_weight=door_pos_weight,
            doorway_focal_gamma=args.doorway_focal_gamma,
            doorway_hard_neg_weight=args.doorway_hard_neg_weight,
            doorway_map_idx=doorway_map_idx,
            use_visibility_head=args.use_visibility_head,
            use_pose_head=args.use_pose_head,
            use_doorway_gating=use_doorway_gating,
            patch_size=args.patch_size,
        )

        door_prob_for_eval = val_arrays["door_prob_gated"] if use_doorway_gating and ("door_prob_gated" in val_arrays) else val_arrays["door_prob"]
        door_t05 = binary_stats(door_prob_for_eval >= 0.5, val_arrays["door_t"])
        occ_t05 = binary_stats(val_arrays["occ_prob"] >= 0.5, val_arrays["occ_t"])
        wall_t05 = binary_stats(val_arrays["wall_prob"] >= 0.5, val_arrays["wall_t"])

        best_occ_t, best_occ_stats, _ = scan_best_threshold(val_arrays["occ_prob"], val_arrays["occ_t"], thresholds, metric="iou")
        best_wall_t, best_wall_stats, _ = scan_best_threshold(val_arrays["wall_prob"], val_arrays["wall_t"], thresholds, metric="f1")
        best_door_t, best_door_stats, _ = scan_best_doorway_threshold(
            door_prob=door_prob_for_eval,
            door_t=val_arrays["door_t"],
            map_idx=val_arrays["map_idx"],
            map_names=map_names,
            thresholds=thresholds,
            fp_lambda=args.doorway_fp_lambda,
        )
        best_free_t, best_free_stats, _ = scan_best_threshold(val_arrays["free_prob"], val_arrays["free_t"], thresholds, metric="accuracy")

        rec = {
            "epoch": epoch,
            "train": train_loss,
            "val_losses": val_loss_metrics,
            "val_threshold_0p5": {
                "occupancy_iou": occ_t05["iou"],
                "wall_f1": wall_t05["f1"],
                "doorway_precision": door_t05["precision"],
                "doorway_recall": door_t05["recall"],
                "doorway_f1": door_t05["f1"],
            },
            "val_best_thresholds": {
                "occupancy": best_occ_t,
                "wall": best_wall_t,
                "doorway": best_door_t,
                "free": best_free_t,
            },
            "val_best_threshold_metrics": {
                "occupancy_iou": best_occ_stats["iou"],
                "wall_f1": best_wall_stats["f1"],
                "doorway_precision": best_door_stats["precision"],
                "doorway_recall": best_door_stats["recall"],
                "doorway_f1": best_door_stats["f1"],
                "free_space_accuracy": best_free_stats["accuracy"],
            },
            "elapsed_sec": float(time.time() - start),
        }
        history.append(rec)

        print(
            f"epoch={epoch:02d} "
            f"train_total={train_loss['total_loss']:.4f} "
            f"val_total={val_total:.4f} "
            f"val_occ_iou@0.5={occ_t05['iou']:.4f} "
            f"val_wall_f1@0.5={wall_t05['f1']:.4f} "
            f"val_door_f1@0.5={door_t05['f1']:.4f} "
            f"val_door_f1@best={best_door_stats['f1']:.4f}"
        )

        if val_total < best_val_loss:
            best_val_loss = float(val_total)
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Evaluate best model
    best_model = EchoMapperV2(
        in_channels=train_ds.in_channels,
        n_bins=train_ds.n_bins,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        meta_dim=8,
        use_visibility_head=args.use_visibility_head,
        use_pose_head=args.use_pose_head,
    ).to(device)
    best_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    val_total, val_arrays, val_loss_metrics = evaluate_losses_and_collect(
        model=best_model,
        loader=val_loader,
        device=device,
        wall_pos_weight=wall_pos_weight,
        door_pos_weight=door_pos_weight,
        doorway_focal_gamma=args.doorway_focal_gamma,
            doorway_hard_neg_weight=args.doorway_hard_neg_weight,
            doorway_map_idx=doorway_map_idx,
            use_visibility_head=args.use_visibility_head,
            use_pose_head=args.use_pose_head,
            use_doorway_gating=use_doorway_gating,
            patch_size=args.patch_size,
        )

    # Threshold search on validation set
    best_occ_t, best_occ_stats, occ_all = scan_best_threshold(val_arrays["occ_prob"], val_arrays["occ_t"], thresholds, metric="iou")
    best_wall_t, best_wall_stats, wall_all = scan_best_threshold(val_arrays["wall_prob"], val_arrays["wall_t"], thresholds, metric="f1")
    door_prob_for_eval = val_arrays["door_prob_gated"] if use_doorway_gating and ("door_prob_gated" in val_arrays) else val_arrays["door_prob"]
    best_door_t, best_door_stats, door_all = scan_best_doorway_threshold(
        door_prob=door_prob_for_eval,
        door_t=val_arrays["door_t"],
        map_idx=val_arrays["map_idx"],
        map_names=map_names,
        thresholds=thresholds,
        fp_lambda=args.doorway_fp_lambda,
    )
    best_free_t, best_free_stats, free_all = scan_best_threshold(val_arrays["free_prob"], val_arrays["free_t"], thresholds, metric="accuracy")

    val_thr_05 = {
        "occupancy": binary_stats(val_arrays["occ_prob"] >= 0.5, val_arrays["occ_t"]),
        "wall": binary_stats(val_arrays["wall_prob"] >= 0.5, val_arrays["wall_t"]),
        "doorway": binary_stats(door_prob_for_eval >= 0.5, val_arrays["door_t"]),
        "free": binary_stats(val_arrays["free_prob"] >= 0.5, val_arrays["free_t"]),
    }

    selected_thresholds = {
        "occupancy": float(best_occ_t),
        "wall": float(best_wall_t),
        "doorway": float(best_door_t),
        "free": float(best_free_t),
    }

    val_metrics_best = metrics_at_thresholds(
        arrays=val_arrays,
        thresholds=selected_thresholds,
        map_names=map_names,
        difficulty_names=difficulty_names,
        use_doorway_gating=use_doorway_gating,
    )
    val_metrics_no_gating = metrics_at_thresholds(
        arrays=val_arrays,
        thresholds=selected_thresholds,
        map_names=map_names,
        difficulty_names=difficulty_names,
        use_doorway_gating=False,
    )
    val_metrics_best["pose_correction_rmse"] = val_loss_metrics["pose_correction_rmse"]

    validation_payload = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "threshold_search": {
            "candidates": thresholds,
            "best_thresholds": selected_thresholds,
            "doorway_at_0.5": {
                "precision": val_thr_05["doorway"]["precision"],
                "recall": val_thr_05["doorway"]["recall"],
                "f1": val_thr_05["doorway"]["f1"],
            },
            "doorway_at_best": {
                "precision": best_door_stats["precision"],
                "recall": best_door_stats["recall"],
                "f1": best_door_stats["f1"],
                "avg_non_doorway_fp_rate": best_door_stats.get("avg_non_doorway_fp_rate", 0.0),
                "combined_score": best_door_stats.get("combined_score", 0.0),
            },
            "occupancy_at_best_iou": best_occ_stats,
            "wall_at_best_f1": best_wall_stats,
            "free_at_best_acc": best_free_stats,
            "all_threshold_stats": {
                "occupancy": occ_all,
                "wall": wall_all,
                "doorway": door_all,
                "free": free_all,
            },
        },
        "val_losses": val_loss_metrics,
        "val_metrics_without_gating": val_metrics_no_gating,
        "val_metrics_at_best_thresholds": val_metrics_best,
    }
    (out_dir / "validation_metrics.json").write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")

    # Test evaluation using selected thresholds from validation
    _, test_arrays, test_loss_metrics = evaluate_losses_and_collect(
        model=best_model,
        loader=test_loader,
        device=device,
        wall_pos_weight=wall_pos_weight,
        door_pos_weight=door_pos_weight,
        doorway_focal_gamma=args.doorway_focal_gamma,
        doorway_hard_neg_weight=args.doorway_hard_neg_weight,
        doorway_map_idx=doorway_map_idx,
        use_visibility_head=args.use_visibility_head,
        use_pose_head=args.use_pose_head,
        use_doorway_gating=use_doorway_gating,
        patch_size=args.patch_size,
    )

    test_thr_05 = {
        "occupancy": binary_stats(test_arrays["occ_prob"] >= 0.5, test_arrays["occ_t"]),
        "wall": binary_stats(test_arrays["wall_prob"] >= 0.5, test_arrays["wall_t"]),
        "doorway": binary_stats(
            (test_arrays["door_prob_gated"] if use_doorway_gating and ("door_prob_gated" in test_arrays) else test_arrays["door_prob"]) >= 0.5,
            test_arrays["door_t"],
        ),
        "free": binary_stats(test_arrays["free_prob"] >= 0.5, test_arrays["free_t"]),
    }

    test_metrics_best = metrics_at_thresholds(
        arrays=test_arrays,
        thresholds=selected_thresholds,
        map_names=map_names,
        difficulty_names=difficulty_names,
        use_doorway_gating=use_doorway_gating,
    )
    test_metrics_no_gating = metrics_at_thresholds(
        arrays=test_arrays,
        thresholds=selected_thresholds,
        map_names=map_names,
        difficulty_names=difficulty_names,
        use_doorway_gating=False,
    )
    test_metrics_best["pose_correction_rmse"] = test_loss_metrics["pose_correction_rmse"]

    test_payload = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "selected_thresholds_from_validation": selected_thresholds,
        "test_losses": test_loss_metrics,
        "test_metrics_without_gating": test_metrics_no_gating,
        "test_metrics_at_threshold_0.5": {
            "occupancy_iou": test_thr_05["occupancy"]["iou"],
            "wall_f1": test_thr_05["wall"]["f1"],
            "doorway_precision": test_thr_05["doorway"]["precision"],
            "doorway_recall": test_thr_05["doorway"]["recall"],
            "doorway_f1": test_thr_05["doorway"]["f1"],
            "free_space_accuracy": test_thr_05["free"]["accuracy"],
        },
        "test_metrics_at_best_thresholds": test_metrics_best,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")

    if args.save_plots:
        save_prediction_plots(
            model=best_model,
            loader=test_loader,
            device=device,
            patch_size=args.patch_size,
            out_dir=out_dir / "prediction_samples",
            thresholds=selected_thresholds,
            max_plots=args.prediction_plot_count,
            use_doorway_gating=use_doorway_gating,
        )

    print("Training complete (v3)")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(
        f"Best thresholds occ/wall/door/free: "
        f"{selected_thresholds['occupancy']:.2f}/"
        f"{selected_thresholds['wall']:.2f}/"
        f"{selected_thresholds['doorway']:.2f}/"
        f"{selected_thresholds['free']:.2f}"
    )
    m = test_metrics_best["overall"]
    print(
        f"Test@best-thr occ_iou={m['occupancy_iou']:.4f}, wall_f1={m['wall_f1']:.4f}, "
        f"door_p/r/f1={m['doorway_precision']:.4f}/{m['doorway_recall']:.4f}/{m['doorway_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
