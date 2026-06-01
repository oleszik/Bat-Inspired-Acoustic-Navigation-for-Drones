"""
Train peak-feature hybrid regressor (MLP with two heads) on hard_v1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
FEATURES_CSV = DATASET_ROOT / "peak_features.csv"
SPLIT_JSON = Path("neural_network/checkpoints/hard_v1/regression_split_indices.json")

CHECKPOINT_DIR = Path("neural_network/checkpoints/hard_v1_peak")
BEST_MODEL_PATH = CHECKPOINT_DIR / "peak_feature_regressor_best.pt"
NORM_STATS_PATH = CHECKPOINT_DIR / "feature_normalization.json"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
DISTANCE_MIN_M = 0.0
DISTANCE_MAX_M = 5.0

FEATURE_COLS = [
    "strongest_peak_delay_ms",
    "strongest_peak_distance_m",
    "strongest_peak_value",
    "first_relative_peak_delay_ms",
    "first_relative_peak_distance_m",
    "first_relative_peak_value",
    "first_noise_floor_peak_delay_ms",
    "first_noise_floor_peak_distance_m",
    "first_noise_floor_peak_value",
    "noise_floor",
    "peak_snr",
    "num_peaks",
    "top1_delay_ms",
    "top1_value",
    "top2_delay_ms",
    "top2_value",
    "top3_delay_ms",
    "top3_value",
    "top2_minus_top1_delay_ms",
    "top3_minus_top1_delay_ms",
    "top2_over_top1_value",
    "top3_over_top1_value",
]


class PeakFeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y_obs: np.ndarray, y_dist: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y_obs = torch.from_numpy(y_obs.astype(np.float32))
        self.y_dist = torch.from_numpy(y_dist.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_obs[idx], self.y_dist[idx]


class PeakFeatureRegressor(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.obstacle_head = nn.Linear(64, 1)
        self.distance_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        obs_logit = self.obstacle_head(h).squeeze(1)
        dist_pred = self.distance_head(h).squeeze(1)
        return obs_logit, dist_pred


def compute_distance_mse(pred: torch.Tensor, true: torch.Tensor, has_obstacle: torch.Tensor) -> torch.Tensor:
    mask = has_obstacle > 0.5
    if mask.any():
        diff = pred[mask] - true[mask]
        return torch.mean(diff * diff)
    return pred.new_tensor(0.0)


def evaluate(model: nn.Module, loader: DataLoader, bce: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    tp = 0
    fn = 0
    abs_err_cm: list[float] = []
    sq_err_cm: list[float] = []

    with torch.no_grad():
        for x, y_obs, y_dist in loader:
            x = x.to(device)
            y_obs = y_obs.to(device)
            y_dist = y_dist.to(device)

            obs_logit, dist_pred = model(x)
            obs_loss = bce(obs_logit, y_obs)
            dist_loss = compute_distance_mse(dist_pred, y_dist, y_obs)
            loss = obs_loss + dist_loss

            running_loss += loss.item() * x.size(0)
            total += x.size(0)

            prob = torch.sigmoid(obs_logit)
            pred_obs = (prob >= 0.5).float()
            correct += int((pred_obs == y_obs).sum().item())
            tp += int(((pred_obs == 1) & (y_obs == 1)).sum().item())
            fn += int(((pred_obs == 0) & (y_obs == 1)).sum().item())

            mask = y_obs > 0.5
            if mask.any():
                err = (dist_pred[mask] - y_dist[mask]) * 100.0
                abs_err_cm.extend(torch.abs(err).cpu().tolist())
                sq_err_cm.extend((err * err).cpu().tolist())

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    val_recall = tp / max(tp + fn, 1)
    val_mae = float(sum(abs_err_cm) / len(abs_err_cm)) if abs_err_cm else float("nan")
    val_rmse = float(math.sqrt(sum(sq_err_cm) / len(sq_err_cm))) if sq_err_cm else float("nan")
    return val_loss, val_acc, val_recall, val_mae, val_rmse


def main() -> None:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing feature CSV: {FEATURES_CSV}")
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(f"Missing split JSON: {SPLIT_JSON}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    with SPLIT_JSON.open("r", encoding="utf-8") as f:
        split = json.load(f)
    train_idx = split["train_indices"]
    val_idx = split["val_indices"]
    test_idx = split["test_indices"]

    x_all = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    y_obs_all = pd.to_numeric(df["has_obstacle"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    y_dist_all = pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(dtype=np.float32)

    y_dist_all = np.where(np.isnan(y_dist_all), 0.0, y_dist_all)
    y_dist_all = np.clip(y_dist_all, DISTANCE_MIN_M, DISTANCE_MAX_M).astype(np.float32)

    x_train = x_all.iloc[train_idx].copy()
    train_mean = x_train.mean(axis=0, skipna=True)
    train_std = x_train.std(axis=0, skipna=True, ddof=0)
    train_std = train_std.mask(train_std < 1e-6, 1.0)

    # Impute missing values with training mean, then z-normalize with train stats.
    x_all_filled = x_all.fillna(train_mean)
    x_all_norm = (x_all_filled - train_mean) / train_std
    x_all_np = x_all_norm.to_numpy(dtype=np.float32)
    x_all_np = np.nan_to_num(x_all_np, nan=0.0, posinf=0.0, neginf=0.0)

    norm_stats = {
        "feature_columns": FEATURE_COLS,
        "mean": {k: float(train_mean[k]) for k in FEATURE_COLS},
        "std": {k: float(train_std[k]) for k in FEATURE_COLS},
    }
    with NORM_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    train_ds = PeakFeatureDataset(x_all_np[train_idx], y_obs_all[train_idx], y_dist_all[train_idx])
    val_ds = PeakFeatureDataset(x_all_np[val_idx], y_obs_all[val_idx], y_dist_all[val_idx])
    _test_ds = PeakFeatureDataset(x_all_np[test_idx], y_obs_all[test_idx], y_dist_all[test_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset size: total={len(df)}, train={len(train_ds)}, val={len(val_ds)}, test={len(_test_ds)}")

    model = PeakFeatureRegressor(in_dim=len(FEATURE_COLS)).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0

        for x, y_obs, y_dist in train_loader:
            x = x.to(device)
            y_obs = y_obs.to(device)
            y_dist = y_dist.to(device)

            opt.zero_grad()
            obs_logit, dist_pred = model(x)
            obs_loss = bce(obs_logit, y_obs)
            dist_loss = compute_distance_mse(dist_pred, y_dist, y_obs)
            loss = obs_loss + dist_loss
            loss.backward()
            opt.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(seen, 1)
        val_loss, val_acc, val_recall, val_mae, val_rmse = evaluate(model, val_loader, bce, device)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_obstacle_acc={val_acc * 100:.2f}% | val_obstacle_recall={val_recall * 100:.2f}% | "
            f"val_distance_mae={val_mae:.2f} cm | val_distance_rmse={val_rmse:.2f} cm"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": FEATURE_COLS,
                },
                BEST_MODEL_PATH,
            )
            print(f"  Saved best model -> {BEST_MODEL_PATH}")

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
