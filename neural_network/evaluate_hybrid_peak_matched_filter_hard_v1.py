"""
Evaluate hybrid safety-gated estimators that combine:
1) peak-feature neural regressor
2) matched-filter noise-floor baseline

Selection priority for best rule:
1) lowest max error
2) lowest RMSE
3) lowest MAE
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


DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
PEAK_FEATURES_CSV = DATASET_ROOT / "peak_features.csv"
SPLIT_JSON = Path("neural_network/checkpoints/hard_v1/regression_split_indices.json")
PEAK_MODEL_PATH = Path("neural_network/checkpoints/hard_v1_peak/peak_feature_regressor_best.pt")
NORM_STATS_PATH = Path("neural_network/checkpoints/hard_v1_peak/feature_normalization.json")
MATCHED_FILTER_PER_SAMPLE_CSV = Path("neural_network/results/hard_v1/matched_filter_baseline_per_sample.csv")
PEAK_RESULTS_JSON = Path("neural_network/results/hard_v1_peak/peak_regressor_test_results.json")
MATCHED_FILTER_SUMMARY_JSON = Path("neural_network/results/hard_v1/matched_filter_baseline_summary.json")

OUT_DIR = Path("neural_network/results/hard_v1_hybrid")
OUT_JSON = OUT_DIR / "hybrid_comparison_results.json"
OUT_CSV = OUT_DIR / "hybrid_per_sample_predictions.csv"
OUT_HIST = OUT_DIR / "best_hybrid_error_histogram.png"
OUT_SCATTER = OUT_DIR / "best_hybrid_true_vs_predicted.png"
OUT_DISAGREE = OUT_DIR / "disagreement_vs_neural_error.png"
OUT_BAR = OUT_DIR / "comparison_mae_rmse_max.png"

DISTANCE_MIN_M = 0.0
DISTANCE_MAX_M = 5.0


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


def compute_metrics(df: pd.DataFrame, pred_obs_col: str, pred_dist_col: str) -> dict[str, float]:
    true_obs = df["has_obstacle"].astype(int)
    pred_obs = df[pred_obs_col].astype(int)

    tp = int(((pred_obs == 1) & (true_obs == 1)).sum())
    fp = int(((pred_obs == 1) & (true_obs == 0)).sum())
    tn = int(((pred_obs == 0) & (true_obs == 0)).sum())
    fn = int(((pred_obs == 0) & (true_obs == 1)).sum())

    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    wall = df[df["has_obstacle"] == 1].copy()
    wall = wall[wall[pred_dist_col].notna()].copy()

    if wall.empty:
        mae = rmse = med = p90 = p95 = mx = float("nan")
    else:
        err_cm = np.abs((wall[pred_dist_col].astype(float) - wall["distance_m"].astype(float)) * 100.0)
        mae = float(np.mean(err_cm))
        rmse = float(np.sqrt(np.mean(err_cm**2)))
        med = float(np.median(err_cm))
        p90 = float(np.percentile(err_cm, 90))
        p95 = float(np.percentile(err_cm, 95))
        mx = float(np.max(err_cm))

    return {
        "obstacle_accuracy": acc,
        "obstacle_precision": precision,
        "obstacle_recall": recall,
        "false_positives": fp,
        "false_negatives": fn,
        "distance_mae_cm": mae,
        "distance_rmse_cm": rmse,
        "distance_median_abs_error_cm": med,
        "distance_p90_abs_error_cm": p90,
        "distance_p95_abs_error_cm": p95,
        "distance_max_abs_error_cm": mx,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [
        PEAK_FEATURES_CSV,
        SPLIT_JSON,
        PEAK_MODEL_PATH,
        NORM_STATS_PATH,
        MATCHED_FILTER_PER_SAMPLE_CSV,
        PEAK_RESULTS_JSON,
        MATCHED_FILTER_SUMMARY_JSON,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    with SPLIT_JSON.open("r", encoding="utf-8") as f:
        split = json.load(f)
    test_idx = split["test_indices"]

    df_peak = pd.read_csv(PEAK_FEATURES_CSV)
    df_test = df_peak.iloc[test_idx].copy().reset_index(drop=True)

    with NORM_STATS_PATH.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    feature_cols = norm["feature_columns"]
    mean = pd.Series(norm["mean"], dtype=float)
    std = pd.Series(norm["std"], dtype=float).mask(lambda s: s < 1e-6, 1.0)

    x = df_test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(mean)
    x = (x - mean) / std
    x_np = np.nan_to_num(x.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    y_obs = pd.to_numeric(df_test["has_obstacle"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_dist = pd.to_numeric(df_test["distance_m"], errors="coerce").to_numpy(dtype=np.float32)
    y_dist = np.where(np.isnan(y_dist), 0.0, y_dist)
    y_dist = np.clip(y_dist, DISTANCE_MIN_M, DISTANCE_MAX_M)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(PEAK_MODEL_PATH, map_location=device)
    model = PeakFeatureRegressor(in_dim=len(feature_cols)).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        xt = torch.from_numpy(x_np).to(device)
        obs_logit, dist_pred = model(xt)
        neural_prob = torch.sigmoid(obs_logit).cpu().numpy()
        neural_dist = dist_pred.cpu().numpy()

    df_test["distance_m"] = pd.to_numeric(df_test["distance_m"], errors="coerce")
    df_test["has_obstacle"] = y_obs
    df_test["neural_obstacle_probability"] = neural_prob
    df_test["neural_pred_has_obstacle"] = (df_test["neural_obstacle_probability"] >= 0.5).astype(int)
    df_test["neural_pred_distance_m"] = neural_dist

    df_mf = pd.read_csv(MATCHED_FILTER_PER_SAMPLE_CSV)
    df_mf_sub = df_mf[["filename", "pred_distance_first_noise_m"]].copy()
    df_mf_sub = df_mf_sub.rename(columns={"pred_distance_first_noise_m": "matched_filter_pred_distance_m"})

    df = df_test.merge(df_mf_sub, on="filename", how="left")
    df["matched_filter_exists"] = df["matched_filter_pred_distance_m"].notna().astype(int)

    df["disagreement_cm"] = np.abs(
        (df["neural_pred_distance_m"].astype(float) - df["matched_filter_pred_distance_m"].astype(float)) * 100.0
    )
    df.loc[df["matched_filter_exists"] == 0, "disagreement_cm"] = np.nan

    # Rule A/B/C: neural obstacle decision, swap distance to MF when disagreement threshold exceeded.
    for rule_name, thr_cm in [("A", 10.0), ("B", 15.0), ("C", 20.0)]:
        pred_obs_col = f"rule_{rule_name}_pred_has_obstacle"
        pred_dist_col = f"rule_{rule_name}_pred_distance_m"
        df[pred_obs_col] = df["neural_pred_has_obstacle"].astype(int)
        df[pred_dist_col] = df["neural_pred_distance_m"].astype(float)
        use_mf = (df["matched_filter_exists"] == 1) & (df["disagreement_cm"] > thr_cm)
        df.loc[use_mf, pred_dist_col] = df.loc[use_mf, "matched_filter_pred_distance_m"].astype(float)

    # Rule D:
    # if obstacle prob < 0.5 => no obstacle
    # if obstacle prob >= 0.5 and MF exists => use MF distance
    # else use neural distance
    df["rule_D_pred_has_obstacle"] = (df["neural_obstacle_probability"] >= 0.5).astype(int)
    df["rule_D_pred_distance_m"] = np.where(
        (df["rule_D_pred_has_obstacle"] == 1) & (df["matched_filter_exists"] == 1),
        df["matched_filter_pred_distance_m"],
        df["neural_pred_distance_m"],
    )

    # Rule E: weighted average 0.5/0.5 when both exist, otherwise neural.
    df["rule_E_pred_has_obstacle"] = df["neural_pred_has_obstacle"].astype(int)
    df["rule_E_pred_distance_m"] = df["neural_pred_distance_m"].astype(float)
    both = df["matched_filter_exists"] == 1
    df.loc[both, "rule_E_pred_distance_m"] = 0.5 * df.loc[both, "neural_pred_distance_m"] + 0.5 * df.loc[
        both, "matched_filter_pred_distance_m"
    ]

    # Add per-rule error columns (wall only).
    for rn in ["A", "B", "C", "D", "E"]:
        dcol = f"rule_{rn}_pred_distance_m"
        ecol = f"rule_{rn}_abs_error_cm"
        df[ecol] = np.nan
        wall_mask = (df["has_obstacle"] == 1) & (df[dcol].notna())
        df.loc[wall_mask, ecol] = np.abs((df.loc[wall_mask, dcol] - df.loc[wall_mask, "distance_m"]) * 100.0)

    # Metrics for each rule.
    metrics = {}
    for rn in ["A", "B", "C", "D", "E"]:
        metrics[rn] = compute_metrics(df, f"rule_{rn}_pred_has_obstacle", f"rule_{rn}_pred_distance_m")

    # Best rule: lowest max error, then lowest RMSE, then lowest MAE.
    def key_fn(item: tuple[str, dict[str, float]]):
        _, m = item
        return (m["distance_max_abs_error_cm"], m["distance_rmse_cm"], m["distance_mae_cm"])

    best_rule, best_metrics = min(metrics.items(), key=key_fn)

    # Comparison references.
    with PEAK_RESULTS_JSON.open("r", encoding="utf-8") as f:
        peak_ref = json.load(f)
    with MATCHED_FILTER_SUMMARY_JSON.open("r", encoding="utf-8") as f:
        mf_ref = json.load(f)
    mf_noise = mf_ref["matched_filter_first_noise_floor_peak"]

    # Plots for best rule.
    best_err_col = f"rule_{best_rule}_abs_error_cm"
    best_dist_col = f"rule_{best_rule}_pred_distance_m"
    wall = df[df["has_obstacle"] == 1].copy()

    plt.figure(figsize=(7, 4), dpi=140)
    plt.hist(wall[best_err_col].dropna().to_numpy(dtype=float), bins=40, color="tab:blue", alpha=0.85)
    plt.xlabel("Absolute distance error (cm)")
    plt.ylabel("Count")
    plt.title(f"Best Hybrid Rule {best_rule}: Error Histogram")
    plt.tight_layout()
    plt.savefig(OUT_HIST)
    plt.close()

    plt.figure(figsize=(6, 6), dpi=140)
    sub = wall[wall[best_dist_col].notna()]
    plt.scatter(sub["distance_m"], sub[best_dist_col], s=12, alpha=0.65)
    if not sub.empty:
        lo = float(min(sub["distance_m"].min(), sub[best_dist_col].min()))
        hi = float(max(sub["distance_m"].max(), sub[best_dist_col].max()))
        plt.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.2, label="Ideal y=x")
        plt.legend()
    plt.xlabel("True distance (m)")
    plt.ylabel("Predicted distance (m)")
    plt.title(f"Best Hybrid Rule {best_rule}: True vs Predicted")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER)
    plt.close()

    # Disagreement vs neural error (wall samples where MF exists).
    wall["neural_error_cm"] = np.abs((wall["neural_pred_distance_m"] - wall["distance_m"]) * 100.0)
    sub2 = wall[(wall["matched_filter_exists"] == 1) & wall["disagreement_cm"].notna()]
    plt.figure(figsize=(6.8, 4.8), dpi=140)
    plt.scatter(sub2["disagreement_cm"], sub2["neural_error_cm"], s=12, alpha=0.55)
    plt.xlabel("Disagreement between neural and matched-filter (cm)")
    plt.ylabel("Neural absolute error (cm)")
    plt.title("Disagreement vs Neural Error (Wall Samples)")
    plt.tight_layout()
    plt.savefig(OUT_DISAGREE)
    plt.close()

    # Comparison bar chart MAE/RMSE/max for peak, MF noise-floor, and rules A-E.
    names = [
        "peak_regressor",
        "matched_filter_noise_floor",
        "hybrid_A",
        "hybrid_B",
        "hybrid_C",
        "hybrid_D",
        "hybrid_E",
    ]
    mae_vals = [
        peak_ref["distance_mae_cm"],
        mf_noise["mae_cm"],
        metrics["A"]["distance_mae_cm"],
        metrics["B"]["distance_mae_cm"],
        metrics["C"]["distance_mae_cm"],
        metrics["D"]["distance_mae_cm"],
        metrics["E"]["distance_mae_cm"],
    ]
    rmse_vals = [
        peak_ref["distance_rmse_cm"],
        mf_noise["rmse_cm"],
        metrics["A"]["distance_rmse_cm"],
        metrics["B"]["distance_rmse_cm"],
        metrics["C"]["distance_rmse_cm"],
        metrics["D"]["distance_rmse_cm"],
        metrics["E"]["distance_rmse_cm"],
    ]
    max_vals = [
        peak_ref["distance_max_abs_error_cm"],
        mf_noise["max_abs_error_cm"],
        metrics["A"]["distance_max_abs_error_cm"],
        metrics["B"]["distance_max_abs_error_cm"],
        metrics["C"]["distance_max_abs_error_cm"],
        metrics["D"]["distance_max_abs_error_cm"],
        metrics["E"]["distance_max_abs_error_cm"],
    ]

    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(12.5, 5.2), dpi=140)
    plt.bar(x - w, mae_vals, width=w, label="MAE")
    plt.bar(x, rmse_vals, width=w, label="RMSE")
    plt.bar(x + w, max_vals, width=w, label="Max Error")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Error (cm)")
    plt.title("MAE / RMSE / Max Error Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_BAR)
    plt.close()

    # Save per-sample and summary.
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "selection_priority": "lowest max error, then lowest RMSE, then lowest MAE",
        "best_rule": best_rule,
        "best_rule_metrics": best_metrics,
        "rule_metrics": metrics,
        "reference_models": {
            "peak_feature_regressor": {
                "mae_cm": peak_ref["distance_mae_cm"],
                "rmse_cm": peak_ref["distance_rmse_cm"],
                "max_abs_error_cm": peak_ref["distance_max_abs_error_cm"],
                "obstacle_accuracy": peak_ref["obstacle_accuracy"],
                "obstacle_precision": peak_ref["obstacle_precision"],
                "obstacle_recall": peak_ref["obstacle_recall"],
                "false_positives": peak_ref["false_positives"],
                "false_negatives": peak_ref["false_negatives"],
            },
            "matched_filter_noise_floor": {
                "mae_cm": mf_noise["mae_cm"],
                "rmse_cm": mf_noise["rmse_cm"],
                "max_abs_error_cm": mf_noise["max_abs_error_cm"],
                "false_positive_rate_no_obstacle": mf_noise["false_positive_rate"],
            },
        },
        "output_files": {
            "per_sample_csv": str(OUT_CSV),
            "results_json": str(OUT_JSON),
            "best_hist_png": str(OUT_HIST),
            "best_true_vs_pred_png": str(OUT_SCATTER),
            "disagreement_vs_neural_error_png": str(OUT_DISAGREE),
            "comparison_bar_png": str(OUT_BAR),
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best hybrid rule: {best_rule}")
    print(
        f"Obstacle acc={best_metrics['obstacle_accuracy'] * 100:.2f}% | "
        f"precision={best_metrics['obstacle_precision'] * 100:.2f}% | "
        f"recall={best_metrics['obstacle_recall'] * 100:.2f}% | "
        f"FP={best_metrics['false_positives']} | FN={best_metrics['false_negatives']}"
    )
    print(
        f"Distance MAE={best_metrics['distance_mae_cm']:.2f} cm | "
        f"RMSE={best_metrics['distance_rmse_cm']:.2f} cm | "
        f"P95={best_metrics['distance_p95_abs_error_cm']:.2f} cm | "
        f"Max={best_metrics['distance_max_abs_error_cm']:.2f} cm"
    )
    print(f"Saved summary: {OUT_JSON}")
    print(f"Saved per-sample predictions: {OUT_CSV}")


if __name__ == "__main__":
    main()
