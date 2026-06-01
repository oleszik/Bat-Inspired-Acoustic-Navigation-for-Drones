"""
Temporal confirmation evaluator for hard_v1 three-state acoustic decisions.

This script simulates repeated measurements using fixed windows of 3 samples
from the existing per-sample three-state predictions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


THREE_STATE_CSV = Path("neural_network/results/hard_v1_three_state/three_state_per_sample_predictions.csv")
PEAK_FEATURES_CSV = Path("datasets/synthetic_echoes_regression_hard_v1/peak_features.csv")

OUT_DIR = Path("neural_network/results/hard_v1_temporal")
OUT_JSON = OUT_DIR / "temporal_confirmation_results.json"
OUT_CSV = OUT_DIR / "temporal_confirmation_predictions.csv"

WINDOW_SIZE = 3


def temporal_decision(window: pd.DataFrame) -> tuple[str, float, str]:
    """
    Decision logic for one 3-sample window.
    """
    obstacle_rows = window[(window["final_state"] == "OBSTACLE") & window["final_distance_m"].notna()]
    if not obstacle_rows.empty:
        dist = float(np.median(obstacle_rows["final_distance_m"].astype(float)))
        return "OBSTACLE", dist, "any_obstacle_with_valid_distance"

    uncertain_count = int((window["final_state"] == "UNCERTAIN").sum())
    if uncertain_count >= 2:
        return "UNCERTAIN", float("nan"), "at_least_two_uncertain"

    return "CLEAR", float("nan"), "default_clear"


def build_windows(df: pd.DataFrame, class_value: int, prefix: str) -> tuple[list[pd.DataFrame], int]:
    """
    Build deterministic windows of 3 from one true class.

    We sort by filename for reproducibility and use only full windows.
    Returns windows and number of leftover samples not used.
    """
    sub = df[df["has_obstacle"] == class_value].copy().sort_values("filename").reset_index(drop=True)
    n = len(sub)
    num_full = n // WINDOW_SIZE
    leftover = n - num_full * WINDOW_SIZE

    windows: list[pd.DataFrame] = []
    for i in range(num_full):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        w = sub.iloc[start:end].copy()
        w["window_id"] = f"{prefix}_{i:04d}"
        windows.append(w)
    return windows, leftover


def main() -> None:
    if not THREE_STATE_CSV.exists():
        raise FileNotFoundError(f"Missing file: {THREE_STATE_CSV}")
    if not PEAK_FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing file: {PEAK_FEATURES_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(THREE_STATE_CSV)
    peak = pd.read_csv(PEAK_FEATURES_CSV)[["filename", "peak_snr", "reflection_strength", "noise_level"]]

    # Merge metadata if missing in three-state CSV.
    if "peak_snr" not in df.columns or "reflection_strength" not in df.columns or "noise_level" not in df.columns:
        df = df.merge(peak, on="filename", how="left", suffixes=("", "_peak"))

    df["has_obstacle"] = pd.to_numeric(df["has_obstacle"], errors="coerce").fillna(0).astype(int)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["final_distance_m"] = pd.to_numeric(df["final_distance_m"], errors="coerce")

    wall_windows, wall_leftover = build_windows(df, class_value=1, prefix="wall")
    noobs_windows, noobs_leftover = build_windows(df, class_value=0, prefix="noobs")
    all_windows = wall_windows + noobs_windows

    rows: list[dict[str, object]] = []
    for w in all_windows:
        true_has_obstacle = int(w["has_obstacle"].iloc[0])
        true_distance = float(np.median(w["distance_m"].dropna())) if true_has_obstacle == 1 else float("nan")
        final_state, final_distance, reason = temporal_decision(w)

        # Distance error only when true wall + final obstacle with valid distance.
        if true_has_obstacle == 1 and final_state == "OBSTACLE" and not np.isnan(final_distance) and not np.isnan(true_distance):
            abs_err_cm = abs(final_distance - true_distance) * 100.0
        else:
            abs_err_cm = float("nan")

        rows.append(
            {
                "window_id": w["window_id"].iloc[0],
                "sample_filenames": ";".join(w["filename"].tolist()),
                "true_has_obstacle": true_has_obstacle,
                "true_distance_m": true_distance,
                "num_obstacle_states": int((w["final_state"] == "OBSTACLE").sum()),
                "num_uncertain_states": int((w["final_state"] == "UNCERTAIN").sum()),
                "num_clear_states": int((w["final_state"] == "CLEAR").sum()),
                "window_final_state": final_state,
                "window_final_distance_m": final_distance,
                "decision_reason": reason,
                "window_abs_error_cm": abs_err_cm,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    wall_out = out[out["true_has_obstacle"] == 1]
    noobs_out = out[out["true_has_obstacle"] == 0]

    wall_final_obstacle = int((wall_out["window_final_state"] == "OBSTACLE").sum())
    wall_final_uncertain = int((wall_out["window_final_state"] == "UNCERTAIN").sum())
    wall_final_clear = int((wall_out["window_final_state"] == "CLEAR").sum())

    noobs_final_clear = int((noobs_out["window_final_state"] == "CLEAR").sum())
    noobs_final_uncertain = int((noobs_out["window_final_state"] == "UNCERTAIN").sum())
    noobs_final_obstacle = int((noobs_out["window_final_state"] == "OBSTACLE").sum())

    wall_obstacle_with_dist = wall_out[
        (wall_out["window_final_state"] == "OBSTACLE") & (wall_out["window_final_distance_m"].notna())
    ].copy()
    if wall_obstacle_with_dist.empty:
        mae = float("nan")
        rmse = float("nan")
    else:
        err = np.abs(
            (wall_obstacle_with_dist["window_final_distance_m"] - wall_obstacle_with_dist["true_distance_m"]) * 100.0
        ).to_numpy(dtype=float)
        mae = float(np.mean(err))
        rmse = float(math.sqrt(np.mean(err**2)))

    coverage = float(len(wall_obstacle_with_dist) / max(len(wall_out), 1))
    safety_miss_rate = float(wall_final_clear / max(len(wall_out), 1))

    summary = {
        "window_size": WINDOW_SIZE,
        "num_windows_total": int(len(out)),
        "num_wall_windows": int(len(wall_out)),
        "num_no_obstacle_windows": int(len(noobs_out)),
        "leftover_samples_not_used": {"wall": int(wall_leftover), "no_obstacle": int(noobs_leftover)},
        "final_counts": {
            "wall_final_obstacle": wall_final_obstacle,
            "wall_final_uncertain": wall_final_uncertain,
            "wall_final_clear": wall_final_clear,
            "no_obstacle_final_clear": noobs_final_clear,
            "no_obstacle_final_uncertain": noobs_final_uncertain,
            "no_obstacle_final_obstacle": noobs_final_obstacle,
        },
        "distance_metrics_for_final_obstacle_windows": {
            "mae_cm": mae,
            "rmse_cm": rmse,
            "num_windows": int(len(wall_obstacle_with_dist)),
        },
        "coverage_wall_windows_with_valid_distance": coverage,
        "safety_miss_rate": safety_miss_rate,
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Temporal confirmation counts:")
    print(f"  wall final OBSTACLE: {wall_final_obstacle}")
    print(f"  wall final UNCERTAIN: {wall_final_uncertain}")
    print(f"  wall final CLEAR: {wall_final_clear}")
    print(f"  no-obstacle final CLEAR: {noobs_final_clear}")
    print(f"  no-obstacle final UNCERTAIN: {noobs_final_uncertain}")
    print(f"  no-obstacle final OBSTACLE: {noobs_final_obstacle}")
    print("\nDistance metrics (final OBSTACLE windows):")
    print(f"  MAE: {mae:.2f} cm")
    print(f"  RMSE: {rmse:.2f} cm")
    print(f"  coverage: {coverage * 100:.2f}%")
    print(f"  safety miss rate: {safety_miss_rate * 100:.2f}%")
    print(f"\nSaved JSON: {OUT_JSON}")
    print(f"Saved per-window CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
