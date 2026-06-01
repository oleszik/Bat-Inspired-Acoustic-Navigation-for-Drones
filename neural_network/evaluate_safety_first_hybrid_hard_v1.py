"""
Evaluate safety-first hybrid rules for hard_v1 acoustic distance estimation.

This script combines:
- peak-feature neural predictions
- matched-filter noise-floor predictions

and prioritizes worst-case safety:
1) lowest max error
2) lowest p95 error
3) lowest RMSE
4) lowest MAE
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HYBRID_INPUT_CSV = Path("neural_network/results/hard_v1_hybrid/hybrid_per_sample_predictions.csv")
MATCHED_FILTER_CSV = Path("neural_network/results/hard_v1/matched_filter_baseline_per_sample.csv")
PEAK_FEATURES_CSV = Path("datasets/synthetic_echoes_regression_hard_v1/peak_features.csv")

PEAK_RESULTS_JSON = Path("neural_network/results/hard_v1_peak/peak_regressor_test_results.json")
MATCHED_SUMMARY_JSON = Path("neural_network/results/hard_v1/matched_filter_baseline_summary.json")
HYBRID_RESULTS_JSON = Path("neural_network/results/hard_v1_hybrid/hybrid_comparison_results.json")

OUT_DIR = Path("neural_network/results/hard_v1_safety_hybrid")
OUT_JSON = OUT_DIR / "safety_hybrid_results.json"
OUT_CSV = OUT_DIR / "safety_hybrid_per_sample_predictions.csv"
OUT_HIST = OUT_DIR / "best_safety_hybrid_error_histogram.png"
OUT_SCATTER = OUT_DIR / "best_safety_hybrid_true_vs_predicted.png"
OUT_BAR = OUT_DIR / "safety_hybrid_comparison_bar.png"


def compute_metrics(df: pd.DataFrame, pred_obs_col: str, pred_dist_col: str) -> dict[str, float]:
    y_true = df["has_obstacle"].astype(int)
    y_pred = df[pred_obs_col].astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

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
        "obstacle_precision": prec,
        "obstacle_recall": rec,
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
    for p in [HYBRID_INPUT_CSV, MATCHED_FILTER_CSV, PEAK_FEATURES_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_h = pd.read_csv(HYBRID_INPUT_CSV)
    df_mf = pd.read_csv(MATCHED_FILTER_CSV)
    df_peak = pd.read_csv(PEAK_FEATURES_CSV)

    # Base is hybrid input CSV (already hard_v1 test samples).
    base_cols = [
        "filename",
        "has_obstacle",
        "distance_m",
        "neural_obstacle_probability",
        "neural_pred_distance_m",
        "neural_pred_has_obstacle",
        "matched_filter_pred_distance_m",
        "disagreement_cm",
        "peak_snr",
        "reflection_strength",
        "noise_level",
        "has_secondary_reflection",
    ]
    for c in base_cols:
        if c not in df_h.columns:
            df_h[c] = np.nan
    df = df_h[base_cols].copy()

    # Merge matched-filter noise-floor prediction explicitly from baseline CSV.
    df_mf2 = df_mf[["filename", "pred_distance_first_noise_m"]].copy()
    df_mf2 = df_mf2.rename(columns={"pred_distance_first_noise_m": "mf_noise_pred_distance_m"})
    df = df.merge(df_mf2, on="filename", how="left")

    # Merge metadata/peak features from peak_features.csv when available.
    df_peak2 = df_peak[
        [
            "filename",
            "peak_snr",
            "reflection_strength",
            "noise_level",
            "has_secondary_reflection",
            "has_obstacle",
            "distance_m",
        ]
    ].copy()
    df = df.merge(df_peak2, on="filename", how="left", suffixes=("", "_peak"))

    # Fill from merged sources where base values were missing.
    for col in ["peak_snr", "reflection_strength", "noise_level", "has_secondary_reflection", "has_obstacle", "distance_m"]:
        peak_col = f"{col}_peak"
        if peak_col in df.columns:
            df[col] = df[col].where(df[col].notna(), df[peak_col])
            df = df.drop(columns=[peak_col])

    # Prefer matched-filter prediction loaded from baseline CSV.
    df["matched_filter_pred_distance_m"] = df["mf_noise_pred_distance_m"].where(
        df["mf_noise_pred_distance_m"].notna(), df["matched_filter_pred_distance_m"]
    )
    df = df.drop(columns=["mf_noise_pred_distance_m"])

    df["has_obstacle"] = pd.to_numeric(df["has_obstacle"], errors="coerce").fillna(0).astype(int)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["neural_obstacle_probability"] = pd.to_numeric(df["neural_obstacle_probability"], errors="coerce").fillna(0.0)
    df["neural_pred_has_obstacle"] = (df["neural_obstacle_probability"] >= 0.5).astype(int)
    df["neural_pred_distance_m"] = pd.to_numeric(df["neural_pred_distance_m"], errors="coerce")
    df["matched_filter_pred_distance_m"] = pd.to_numeric(df["matched_filter_pred_distance_m"], errors="coerce")

    # Recompute disagreement robustly.
    both = df["neural_pred_distance_m"].notna() & df["matched_filter_pred_distance_m"].notna()
    df["disagreement_cm"] = np.nan
    df.loc[both, "disagreement_cm"] = np.abs(
        (df.loc[both, "neural_pred_distance_m"] - df.loc[both, "matched_filter_pred_distance_m"]) * 100.0
    )

    # Dynamic thresholds for S4 (safety-focused).
    wall = df[df["has_obstacle"] == 1]
    snr_low_thr = float(wall["peak_snr"].quantile(0.25)) if wall["peak_snr"].notna().any() else 2.0
    refl_low_thr = float(wall["reflection_strength"].quantile(0.25)) if wall["reflection_strength"].notna().any() else 0.15

    # Shared obstacle decision from neural probability.
    for rn in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        df[f"{rn}_pred_has_obstacle"] = df["neural_pred_has_obstacle"].astype(int)

    # Rule S1: always matched-filter when available.
    df["S1_pred_distance_m"] = df["matched_filter_pred_distance_m"].where(
        df["matched_filter_pred_distance_m"].notna(), df["neural_pred_distance_m"]
    )

    # Rule S2: if disagreement > 5cm => MF, else average.
    df["S2_pred_distance_m"] = df["neural_pred_distance_m"]
    both = df["matched_filter_pred_distance_m"].notna() & df["neural_pred_distance_m"].notna()
    use_mf_s2 = both & (df["disagreement_cm"] > 5.0)
    use_avg_s2 = both & (~use_mf_s2)
    df.loc[use_mf_s2, "S2_pred_distance_m"] = df.loc[use_mf_s2, "matched_filter_pred_distance_m"]
    df.loc[use_avg_s2, "S2_pred_distance_m"] = 0.5 * df.loc[use_avg_s2, "neural_pred_distance_m"] + 0.5 * df.loc[
        use_avg_s2, "matched_filter_pred_distance_m"
    ]

    # Rule S3: if disagreement > 3cm => MF, else average.
    df["S3_pred_distance_m"] = df["neural_pred_distance_m"]
    use_mf_s3 = both & (df["disagreement_cm"] > 3.0)
    use_avg_s3 = both & (~use_mf_s3)
    df.loc[use_mf_s3, "S3_pred_distance_m"] = df.loc[use_mf_s3, "matched_filter_pred_distance_m"]
    df.loc[use_avg_s3, "S3_pred_distance_m"] = 0.5 * df.loc[use_avg_s3, "neural_pred_distance_m"] + 0.5 * df.loc[
        use_avg_s3, "matched_filter_pred_distance_m"
    ]

    # Rule S4: low SNR or low reflection => MF, else average.
    df["S4_pred_distance_m"] = df["neural_pred_distance_m"]
    low_quality = (pd.to_numeric(df["peak_snr"], errors="coerce") < snr_low_thr) | (
        pd.to_numeric(df["reflection_strength"], errors="coerce") < refl_low_thr
    )
    use_mf_s4 = both & low_quality
    use_avg_s4 = both & (~low_quality)
    df.loc[use_mf_s4, "S4_pred_distance_m"] = df.loc[use_mf_s4, "matched_filter_pred_distance_m"]
    df.loc[use_avg_s4, "S4_pred_distance_m"] = 0.5 * df.loc[use_avg_s4, "neural_pred_distance_m"] + 0.5 * df.loc[
        use_avg_s4, "matched_filter_pred_distance_m"
    ]

    # Rule S5: if secondary reflection and disagreement > 3cm => MF, else average.
    df["S5_pred_distance_m"] = df["neural_pred_distance_m"]
    has_sec = pd.to_numeric(df["has_secondary_reflection"], errors="coerce").fillna(0).astype(int) == 1
    use_mf_s5 = both & has_sec & (df["disagreement_cm"] > 3.0)
    use_avg_s5 = both & (~use_mf_s5)
    df.loc[use_mf_s5, "S5_pred_distance_m"] = df.loc[use_mf_s5, "matched_filter_pred_distance_m"]
    df.loc[use_avg_s5, "S5_pred_distance_m"] = 0.5 * df.loc[use_avg_s5, "neural_pred_distance_m"] + 0.5 * df.loc[
        use_avg_s5, "matched_filter_pred_distance_m"
    ]

    # Rule S6: conservative smaller distance.
    df["S6_pred_distance_m"] = df["neural_pred_distance_m"]
    df.loc[both, "S6_pred_distance_m"] = np.minimum(
        df.loc[both, "neural_pred_distance_m"], df.loc[both, "matched_filter_pred_distance_m"]
    )
    only_mf = df["matched_filter_pred_distance_m"].notna() & df["neural_pred_distance_m"].isna()
    df.loc[only_mf, "S6_pred_distance_m"] = df.loc[only_mf, "matched_filter_pred_distance_m"]

    # Add per-rule error columns (wall samples only).
    for rn in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        dcol = f"{rn}_pred_distance_m"
        ecol = f"{rn}_abs_error_cm"
        df[ecol] = np.nan
        wall_mask = (df["has_obstacle"] == 1) & df[dcol].notna()
        df.loc[wall_mask, ecol] = np.abs((df.loc[wall_mask, dcol] - df.loc[wall_mask, "distance_m"]) * 100.0)

    metrics = {}
    for rn in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        metrics[rn] = compute_metrics(df, f"{rn}_pred_has_obstacle", f"{rn}_pred_distance_m")

    # Best safety-first rule:
    # lowest max -> lowest p95 -> lowest RMSE -> lowest MAE
    def rank_key(item: tuple[str, dict[str, float]]):
        _, m = item
        return (
            m["distance_max_abs_error_cm"],
            m["distance_p95_abs_error_cm"],
            m["distance_rmse_cm"],
            m["distance_mae_cm"],
        )

    best_rule, best_metrics = min(metrics.items(), key=rank_key)

    # Load references for comparison printout.
    with PEAK_RESULTS_JSON.open("r", encoding="utf-8") as f:
        peak_ref = json.load(f)
    with MATCHED_SUMMARY_JSON.open("r", encoding="utf-8") as f:
        mf_ref = json.load(f)["matched_filter_first_noise_floor_peak"]
    with HYBRID_RESULTS_JSON.open("r", encoding="utf-8") as f:
        prev_hybrid = json.load(f)
    prev_rule_e = prev_hybrid["rule_metrics"]["E"] if "rule_metrics" in prev_hybrid else prev_hybrid["best_rule_metrics"]

    # Save per-sample CSV.
    df.to_csv(OUT_CSV, index=False)

    # Plot 1: best rule histogram.
    wall = df[df["has_obstacle"] == 1].copy()
    best_err_col = f"{best_rule}_abs_error_cm"
    best_dist_col = f"{best_rule}_pred_distance_m"
    plt.figure(figsize=(7, 4), dpi=140)
    plt.hist(wall[best_err_col].dropna().to_numpy(dtype=float), bins=40, color="tab:blue", alpha=0.85)
    plt.xlabel("Absolute distance error (cm)")
    plt.ylabel("Count")
    plt.title(f"Best Safety Rule {best_rule}: Error Histogram")
    plt.tight_layout()
    plt.savefig(OUT_HIST)
    plt.close()

    # Plot 2: best rule true vs predicted.
    sub = wall[wall[best_dist_col].notna()]
    plt.figure(figsize=(6, 6), dpi=140)
    plt.scatter(sub["distance_m"], sub[best_dist_col], s=12, alpha=0.65)
    if not sub.empty:
        lo = float(min(sub["distance_m"].min(), sub[best_dist_col].min()))
        hi = float(max(sub["distance_m"].max(), sub[best_dist_col].max()))
        plt.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.2, label="Ideal y=x")
        plt.legend()
    plt.xlabel("True distance (m)")
    plt.ylabel("Predicted distance (m)")
    plt.title(f"Best Safety Rule {best_rule}: True vs Predicted")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER)
    plt.close()

    # Plot 3: bar chart comparing peak, MF noise-floor, prev Rule E, and S1-S6.
    names = ["peak_regressor", "matched_filter_noise", "hybrid_E_prev", "S1", "S2", "S3", "S4", "S5", "S6"]
    mae_vals = [
        peak_ref["distance_mae_cm"],
        mf_ref["mae_cm"],
        prev_rule_e["distance_mae_cm"],
        metrics["S1"]["distance_mae_cm"],
        metrics["S2"]["distance_mae_cm"],
        metrics["S3"]["distance_mae_cm"],
        metrics["S4"]["distance_mae_cm"],
        metrics["S5"]["distance_mae_cm"],
        metrics["S6"]["distance_mae_cm"],
    ]
    rmse_vals = [
        peak_ref["distance_rmse_cm"],
        mf_ref["rmse_cm"],
        prev_rule_e["distance_rmse_cm"],
        metrics["S1"]["distance_rmse_cm"],
        metrics["S2"]["distance_rmse_cm"],
        metrics["S3"]["distance_rmse_cm"],
        metrics["S4"]["distance_rmse_cm"],
        metrics["S5"]["distance_rmse_cm"],
        metrics["S6"]["distance_rmse_cm"],
    ]
    max_vals = [
        peak_ref["distance_max_abs_error_cm"],
        mf_ref["max_abs_error_cm"],
        prev_rule_e["distance_max_abs_error_cm"],
        metrics["S1"]["distance_max_abs_error_cm"],
        metrics["S2"]["distance_max_abs_error_cm"],
        metrics["S3"]["distance_max_abs_error_cm"],
        metrics["S4"]["distance_max_abs_error_cm"],
        metrics["S5"]["distance_max_abs_error_cm"],
        metrics["S6"]["distance_max_abs_error_cm"],
    ]
    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(13.2, 5.2), dpi=140)
    plt.bar(x - w, mae_vals, width=w, label="MAE")
    plt.bar(x, rmse_vals, width=w, label="RMSE")
    plt.bar(x + w, max_vals, width=w, label="Max Error")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Error (cm)")
    plt.title("Safety Hybrid Comparison: MAE / RMSE / Max Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_BAR)
    plt.close()

    results = {
        "selection_priority": "lowest max error -> lowest p95 -> lowest RMSE -> lowest MAE",
        "s4_thresholds": {"peak_snr_low_threshold": snr_low_thr, "reflection_strength_low_threshold": refl_low_thr},
        "best_rule": best_rule,
        "best_rule_metrics": best_metrics,
        "rule_metrics": metrics,
        "comparison_references": {
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
            "matched_filter_noise_floor_baseline": {
                "mae_cm": mf_ref["mae_cm"],
                "rmse_cm": mf_ref["rmse_cm"],
                "max_abs_error_cm": mf_ref["max_abs_error_cm"],
                "false_positive_rate_no_obstacle": mf_ref["false_positive_rate"],
            },
            "previous_best_hybrid_rule_E": prev_rule_e,
        },
        "output_files": {
            "results_json": str(OUT_JSON),
            "per_sample_csv": str(OUT_CSV),
            "best_histogram_png": str(OUT_HIST),
            "best_true_vs_pred_png": str(OUT_SCATTER),
            "comparison_bar_png": str(OUT_BAR),
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Best safety-first rule: {best_rule}")
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
    print("\nComparison references:")
    print(
        f"Peak-feature regressor -> MAE {peak_ref['distance_mae_cm']:.2f}, "
        f"RMSE {peak_ref['distance_rmse_cm']:.2f}, max {peak_ref['distance_max_abs_error_cm']:.2f} cm"
    )
    print(
        f"Matched-filter noise-floor -> MAE {mf_ref['mae_cm']:.2f}, "
        f"RMSE {mf_ref['rmse_cm']:.2f}, max {mf_ref['max_abs_error_cm']:.2f} cm"
    )
    print(
        f"Previous best hybrid Rule E -> MAE {prev_rule_e['distance_mae_cm']:.2f}, "
        f"RMSE {prev_rule_e['distance_rmse_cm']:.2f}, max {prev_rule_e['distance_max_abs_error_cm']:.2f} cm"
    )
    print(f"Saved results JSON: {OUT_JSON}")
    print(f"Saved per-sample CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
