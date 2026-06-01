"""
Evaluate peak-feature regressor on hard_v1 test split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
FEATURES_CSV = DATASET_ROOT / "peak_features.csv"
SPLIT_JSON = Path("neural_network/checkpoints/hard_v1/regression_split_indices.json")

CHECKPOINT_DIR = Path("neural_network/checkpoints/hard_v1_peak")
MODEL_PATH = CHECKPOINT_DIR / "peak_feature_regressor_best.pt"
NORM_STATS_PATH = CHECKPOINT_DIR / "feature_normalization.json"

RESULTS_DIR = Path("neural_network/results/hard_v1_peak")
RESULTS_JSON = RESULTS_DIR / "peak_regressor_test_results.json"
SCATTER_PNG = RESULTS_DIR / "peak_regressor_true_vs_predicted.png"
HIST_PNG = RESULTS_DIR / "peak_regressor_error_histogram.png"

BATCH_SIZE = 64
DISTANCE_MIN_M = 0.0
DISTANCE_MAX_M = 5.0


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
        return self.obstacle_head(h).squeeze(1), self.distance_head(h).squeeze(1)


def main() -> None:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing feature CSV: {FEATURES_CSV}")
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(f"Missing split JSON: {SPLIT_JSON}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")
    if not NORM_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing normalization stats: {NORM_STATS_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    with SPLIT_JSON.open("r", encoding="utf-8") as f:
        split = json.load(f)
    test_idx = split["test_indices"]

    with NORM_STATS_PATH.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    feature_cols: list[str] = norm["feature_columns"]
    mean = pd.Series(norm["mean"], dtype=float)
    std = pd.Series(norm["std"], dtype=float)
    std = std.mask(std < 1e-6, 1.0)

    x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_all = x_all.fillna(mean)
    x_all = (x_all - mean) / std
    x_all = np.nan_to_num(x_all.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    y_obs_all = pd.to_numeric(df["has_obstacle"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    y_dist_all = pd.to_numeric(df["distance_m"], errors="coerce").to_numpy(dtype=np.float32)
    y_dist_all = np.where(np.isnan(y_dist_all), 0.0, y_dist_all)
    y_dist_all = np.clip(y_dist_all, DISTANCE_MIN_M, DISTANCE_MAX_M).astype(np.float32)

    test_ds = PeakFeatureDataset(x_all[test_idx], y_obs_all[test_idx], y_dist_all[test_idx])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = PeakFeatureRegressor(in_dim=len(feature_cols)).to(device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()

    tp = fp = tn = fn = 0
    abs_err_cm: list[float] = []
    sq_err_cm: list[float] = []
    true_dist: list[float] = []
    pred_dist: list[float] = []

    with torch.no_grad():
        for x, y_obs, y_dist in test_loader:
            x = x.to(device)
            y_obs = y_obs.to(device)
            y_dist = y_dist.to(device)

            obs_logit, dist_pred = model(x)
            prob = torch.sigmoid(obs_logit)
            pred_obs = (prob >= 0.5).float()

            tp += int(((pred_obs == 1) & (y_obs == 1)).sum().item())
            fp += int(((pred_obs == 1) & (y_obs == 0)).sum().item())
            tn += int(((pred_obs == 0) & (y_obs == 0)).sum().item())
            fn += int(((pred_obs == 0) & (y_obs == 1)).sum().item())

            wall_mask = y_obs > 0.5
            if wall_mask.any():
                y_t = y_dist[wall_mask]
                y_p = dist_pred[wall_mask]
                err = (y_p - y_t) * 100.0
                abs_err_cm.extend(torch.abs(err).cpu().tolist())
                sq_err_cm.extend((err * err).cpu().tolist())
                true_dist.extend(y_t.cpu().tolist())
                pred_dist.extend(y_p.cpu().tolist())

    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    err_arr = np.asarray(abs_err_cm, dtype=float)
    mae = float(np.mean(err_arr)) if err_arr.size else float("nan")
    rmse = float(math.sqrt(np.mean(np.asarray(sq_err_cm, dtype=float)))) if sq_err_cm else float("nan")
    med = float(np.median(err_arr)) if err_arr.size else float("nan")
    p90 = float(np.percentile(err_arr, 90)) if err_arr.size else float("nan")
    p95 = float(np.percentile(err_arr, 95)) if err_arr.size else float("nan")
    mx = float(np.max(err_arr)) if err_arr.size else float("nan")

    print(f"Obstacle accuracy: {acc * 100:.2f}%")
    print(f"Obstacle precision: {prec * 100:.2f}%")
    print(f"Obstacle recall: {rec * 100:.2f}%")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Distance MAE (cm): {mae:.2f}")
    print(f"Distance RMSE (cm): {rmse:.2f}")
    print(f"Distance median abs error (cm): {med:.2f}")
    print(f"Distance p90 abs error (cm): {p90:.2f}")
    print(f"Distance p95 abs error (cm): {p95:.2f}")
    print(f"Distance max abs error (cm): {mx:.2f}")

    plt.figure(figsize=(6, 6), dpi=140)
    plt.scatter(true_dist, pred_dist, s=12, alpha=0.65)
    if true_dist:
        lo = min(min(true_dist), min(pred_dist))
        hi = max(max(true_dist), max(pred_dist))
        plt.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.2, label="Ideal y=x")
        plt.legend()
    plt.xlabel("True distance (m)")
    plt.ylabel("Predicted distance (m)")
    plt.title("Peak-Feature Regressor: True vs Predicted")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG)
    plt.close()

    plt.figure(figsize=(7, 4), dpi=140)
    plt.hist(err_arr, bins=40, color="tab:blue", alpha=0.85)
    plt.xlabel("Absolute distance error (cm)")
    plt.ylabel("Count")
    plt.title("Peak-Feature Regressor Error Histogram")
    plt.tight_layout()
    plt.savefig(HIST_PNG)
    plt.close()

    results = {
        "num_test_samples": total,
        "num_wall_samples_for_distance_metrics": int(err_arr.size),
        "obstacle_accuracy": acc,
        "obstacle_precision": prec,
        "obstacle_recall": rec,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "distance_mae_cm": mae,
        "distance_rmse_cm": rmse,
        "distance_median_abs_error_cm": med,
        "distance_p90_abs_error_cm": p90,
        "distance_p95_abs_error_cm": p95,
        "distance_max_abs_error_cm": mx,
        "confusion_counts": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results JSON: {RESULTS_JSON}")
    print(f"Saved scatter plot: {SCATTER_PNG}")
    print(f"Saved error histogram: {HIST_PNG}")


if __name__ == "__main__":
    main()
