"""
Debug worst-case errors for safety hybrid estimator on hard_v1.

This script:
1) loads per-sample safety-hybrid predictions,
2) sorts wall samples by absolute error,
3) prints/saves top-20 worst cases,
4) prints a focused diagnosis for the single worst sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SAFETY_CSV = Path("neural_network/results/hard_v1_safety_hybrid/safety_hybrid_per_sample_predictions.csv")
SAFETY_JSON = Path("neural_network/results/hard_v1_safety_hybrid/safety_hybrid_results.json")
MATCHED_FILTER_CSV = Path("neural_network/results/hard_v1/matched_filter_baseline_per_sample.csv")
PEAK_FEATURES_CSV = Path("datasets/synthetic_echoes_regression_hard_v1/peak_features.csv")

OUT_TOP20 = Path("neural_network/results/hard_v1_safety_hybrid/top20_worst_safety_hybrid_cases.csv")


def get_best_rule() -> str:
    if SAFETY_JSON.exists():
        with SAFETY_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "best_rule" in data:
            return str(data["best_rule"])
    return "S4"


def col_or_nan(row: pd.Series, col: str):
    return row[col] if col in row.index else np.nan


def main() -> None:
    if not SAFETY_CSV.exists():
        raise FileNotFoundError(f"Missing file: {SAFETY_CSV}")

    best_rule = get_best_rule()
    pred_dist_col = f"{best_rule}_pred_distance_m"
    pred_obs_col = f"{best_rule}_pred_has_obstacle"
    err_col = f"{best_rule}_abs_error_cm"

    df = pd.read_csv(SAFETY_CSV)
    if pred_dist_col not in df.columns or err_col not in df.columns:
        raise ValueError(
            f"Expected columns for best rule '{best_rule}' not found. "
            f"Need: {pred_dist_col}, {err_col}"
        )

    # Enrich with optional metadata if available.
    if PEAK_FEATURES_CSV.exists():
        peak = pd.read_csv(PEAK_FEATURES_CSV)
        extra_cols = ["filename", "secondary_delay_ms", "secondary_strength"]
        have = [c for c in extra_cols if c in peak.columns]
        if "filename" in have:
            df = df.merge(peak[have], on="filename", how="left")

    if MATCHED_FILTER_CSV.exists():
        mf = pd.read_csv(MATCHED_FILTER_CSV)
        mf_cols = ["filename", "pred_distance_first_noise_m"]
        have = [c for c in mf_cols if c in mf.columns]
        if "filename" in have and "pred_distance_first_noise_m" in have:
            df = df.merge(mf[have], on="filename", how="left", suffixes=("", "_mfsrc"))
            # Prefer explicit baseline column if safety CSV missing MF prediction.
            if "matched_filter_pred_distance_m" in df.columns:
                df["matched_filter_pred_distance_m"] = df["matched_filter_pred_distance_m"].where(
                    df["matched_filter_pred_distance_m"].notna(), df["pred_distance_first_noise_m"]
                )
            else:
                df["matched_filter_pred_distance_m"] = df["pred_distance_first_noise_m"]
            df = df.drop(columns=["pred_distance_first_noise_m"])

    wall = df[df["has_obstacle"] == 1].copy()
    wall = wall.sort_values(err_col, ascending=False)

    top20 = wall.head(20).copy()
    OUT_TOP20.parent.mkdir(parents=True, exist_ok=True)
    top20.to_csv(OUT_TOP20, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"Top 20 worst wall samples by {err_col}:\n")
    print(top20)
    print(f"\nSaved top-20 CSV: {OUT_TOP20}")

    if top20.empty:
        print("\nNo wall samples found for diagnosis.")
        return

    worst = top20.iloc[0]
    true_has_obs = int(worst["has_obstacle"])
    true_dist = float(worst["distance_m"]) if pd.notna(worst["distance_m"]) else np.nan
    neural_prob = float(col_or_nan(worst, "neural_obstacle_probability"))
    neural_pred_obs = int(col_or_nan(worst, "neural_pred_has_obstacle")) if pd.notna(col_or_nan(worst, "neural_pred_has_obstacle")) else -1
    neural_dist = float(col_or_nan(worst, "neural_pred_distance_m")) if pd.notna(col_or_nan(worst, "neural_pred_distance_m")) else np.nan
    mf_dist = float(col_or_nan(worst, "matched_filter_pred_distance_m")) if pd.notna(col_or_nan(worst, "matched_filter_pred_distance_m")) else np.nan
    final_dist = float(col_or_nan(worst, pred_dist_col)) if pd.notna(col_or_nan(worst, pred_dist_col)) else np.nan
    final_err = float(col_or_nan(worst, err_col)) if pd.notna(col_or_nan(worst, err_col)) else np.nan
    disagreement = float(col_or_nan(worst, "disagreement_cm")) if pd.notna(col_or_nan(worst, "disagreement_cm")) else np.nan

    print("\nWorst max-error sample details:")
    print(f"filename: {worst['filename']}")
    print(f"true has_obstacle: {true_has_obs}")
    print(f"true distance_m: {true_dist}")
    print(f"neural obstacle probability: {neural_prob}")
    print(f"neural predicted has_obstacle: {neural_pred_obs}")
    print(f"neural predicted distance_m: {neural_dist}")
    print(f"matched-filter noise-floor predicted distance_m: {mf_dist}")
    print(f"final safety-hybrid predicted distance_m: {final_dist}")
    print(f"final rule used: {best_rule}")
    print(f"absolute error cm: {final_err}")
    print(f"peak_snr: {col_or_nan(worst, 'peak_snr')}")
    print(f"reflection_strength: {col_or_nan(worst, 'reflection_strength')}")
    print(f"noise_level: {col_or_nan(worst, 'noise_level')}")
    print(f"has_secondary_reflection: {col_or_nan(worst, 'has_secondary_reflection')}")
    print(f"secondary_delay_ms: {col_or_nan(worst, 'secondary_delay_ms')}")
    print(f"secondary_strength: {col_or_nan(worst, 'secondary_strength')}")

    # Diagnosis logic.
    print("\nDiagnosis:")
    mf_exists = not np.isnan(mf_dist)
    if mf_exists and not np.isnan(true_dist):
        mf_err = abs(mf_dist - true_dist) * 100.0
    else:
        mf_err = np.nan

    if mf_exists and not np.isnan(mf_err) and (mf_err <= 10.0) and (final_err >= 20.0):
        print("Hybrid rule selection bug or gating issue.")

    if true_has_obs == 1 and pd.notna(col_or_nan(worst, pred_obs_col)) and int(col_or_nan(worst, pred_obs_col)) == 0:
        print("False negative caused distance tail.")

    if not mf_exists:
        print("Matched-filter missing for this sample.")

    if mf_exists and not np.isnan(disagreement) and disagreement > 10.0 and not np.isnan(mf_err) and mf_err <= 10.0:
        print("Need stricter matched-filter fallback.")


if __name__ == "__main__":
    main()
