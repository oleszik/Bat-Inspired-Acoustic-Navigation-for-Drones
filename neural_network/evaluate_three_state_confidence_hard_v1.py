"""
Three-state acoustic confidence evaluator for hard_v1:
- CLEAR
- OBSTACLE
- UNCERTAIN

No retraining is performed. Decisions use transparent fixed thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAFETY_CSV = Path("neural_network/results/hard_v1_safety_hybrid/safety_hybrid_per_sample_predictions.csv")
MATCHED_CSV = Path("neural_network/results/hard_v1/matched_filter_baseline_per_sample.csv")
WEAK_CSV = Path("neural_network/results/hard_v1_weak_echo/weak_echo_fallback_per_sample.csv")
PEAK_CSV = Path("datasets/synthetic_echoes_regression_hard_v1/peak_features.csv")

OUT_DIR = Path("neural_network/results/hard_v1_three_state")
OUT_JSON = OUT_DIR / "three_state_results.json"
OUT_CSV = OUT_DIR / "three_state_per_sample_predictions.csv"
OUT_CONF_PNG = OUT_DIR / "three_state_confusion_table.png"


# Fixed transparent thresholds (no training).
NEURAL_OBS_PROB_HIGH = 0.90
PEAK_SNR_LOW = 1.5
REFLECTION_LOW = 0.03
NOISE_HIGH = 0.02


def main() -> None:
    for p in [SAFETY_CSV, MATCHED_CSV, WEAK_CSV, PEAK_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Base test samples (hard_v1 test split) come from safety-hybrid per-sample CSV.
    s = pd.read_csv(SAFETY_CSV)
    s_keep = [
        "filename",
        "has_obstacle",
        "distance_m",
        "neural_obstacle_probability",
        "neural_pred_distance_m",
        "S4_pred_distance_m",
    ]
    for c in s_keep:
        if c not in s.columns:
            s[c] = np.nan
    df = s[s_keep].copy()

    # Matched-filter noise-floor predictions.
    mf = pd.read_csv(MATCHED_CSV)[["filename", "pred_distance_first_noise_m"]].rename(
        columns={"pred_distance_first_noise_m": "mf_noise_floor_distance_m"}
    )
    df = df.merge(mf, on="filename", how="left")

    # Weak-fallback info.
    w = pd.read_csv(WEAK_CSV)[
        ["filename", "weak_fallback_detected", "weak_fallback_distance_m", "weak_fallback_confidence", "selected_mode"]
    ].rename(columns={"selected_mode": "weak_selected_mode"})
    df = df.merge(w, on="filename", how="left")

    # Metadata.
    p = pd.read_csv(PEAK_CSV)[
        ["filename", "peak_snr", "reflection_strength", "noise_level", "has_secondary_reflection"]
    ]
    df = df.merge(p, on="filename", how="left")

    # Type safety.
    df["has_obstacle"] = pd.to_numeric(df["has_obstacle"], errors="coerce").fillna(0).astype(int)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["neural_obstacle_probability"] = pd.to_numeric(df["neural_obstacle_probability"], errors="coerce").fillna(0.0)
    df["neural_pred_distance_m"] = pd.to_numeric(df["neural_pred_distance_m"], errors="coerce")
    df["S4_pred_distance_m"] = pd.to_numeric(df["S4_pred_distance_m"], errors="coerce")
    df["mf_noise_floor_distance_m"] = pd.to_numeric(df["mf_noise_floor_distance_m"], errors="coerce")
    df["weak_fallback_detected"] = pd.to_numeric(df["weak_fallback_detected"], errors="coerce").fillna(0).astype(int)
    df["weak_fallback_distance_m"] = pd.to_numeric(df["weak_fallback_distance_m"], errors="coerce")
    df["weak_fallback_confidence"] = pd.to_numeric(df["weak_fallback_confidence"], errors="coerce").fillna(0.0)
    df["peak_snr"] = pd.to_numeric(df["peak_snr"], errors="coerce")
    df["reflection_strength"] = pd.to_numeric(df["reflection_strength"], errors="coerce")
    df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")
    df["has_secondary_reflection"] = pd.to_numeric(df["has_secondary_reflection"], errors="coerce").fillna(0).astype(int)

    # Three-state decision.
    states: list[str] = []
    confs: list[str] = []
    final_dist: list[float] = []
    reasons: list[str] = []

    for _, r in df.iterrows():
        mf_dist = r["mf_noise_floor_distance_m"]
        neural_prob = float(r["neural_obstacle_probability"])
        neural_dist = r["S4_pred_distance_m"] if pd.notna(r["S4_pred_distance_m"]) else r["neural_pred_distance_m"]
        weak_det = int(r["weak_fallback_detected"])
        weak_dist = r["weak_fallback_distance_m"]
        low_quality = (
            (pd.notna(r["noise_level"]) and float(r["noise_level"]) >= NOISE_HIGH)
            or (pd.notna(r["reflection_strength"]) and float(r["reflection_strength"]) <= REFLECTION_LOW)
            or (pd.notna(r["peak_snr"]) and float(r["peak_snr"]) <= PEAK_SNR_LOW)
        )

        if pd.notna(mf_dist):
            state = "OBSTACLE"
            dist = float(mf_dist)
            conf = "high"
            reason = "matched_filter_noise_floor_detected"
        elif neural_prob >= NEURAL_OBS_PROB_HIGH:
            state = "OBSTACLE"
            dist = float(neural_dist) if pd.notna(neural_dist) else np.nan
            conf = "medium"
            reason = "high_neural_obstacle_probability"
        elif weak_det == 1:
            state = "UNCERTAIN"
            dist = np.nan
            conf = "low"
            reason = "weak_fallback_only"
        elif low_quality:
            state = "UNCERTAIN"
            dist = np.nan
            conf = "low"
            reason = "low_signal_quality"
        else:
            state = "CLEAR"
            dist = np.nan
            conf = "high"
            reason = "no_echo_and_no_risk_flags"

        states.append(state)
        confs.append(conf)
        final_dist.append(dist)
        reasons.append(reason)

    df["final_state"] = states
    df["final_confidence"] = confs
    df["final_distance_m"] = final_dist
    df["decision_reason"] = reasons

    # Error only when true wall and distance provided.
    df["absolute_error_cm"] = np.nan
    wall_with_dist = (df["has_obstacle"] == 1) & df["final_distance_m"].notna() & df["distance_m"].notna()
    df.loc[wall_with_dist, "absolute_error_cm"] = np.abs(
        (df.loc[wall_with_dist, "final_distance_m"] - df.loc[wall_with_dist, "distance_m"]) * 100.0
    )

    # Confusion counts (three-state).
    wall = df[df["has_obstacle"] == 1]
    noobs = df[df["has_obstacle"] == 0]

    wall_obstacle = int((wall["final_state"] == "OBSTACLE").sum())
    wall_uncertain = int((wall["final_state"] == "UNCERTAIN").sum())
    wall_clear = int((wall["final_state"] == "CLEAR").sum())

    noobs_clear = int((noobs["final_state"] == "CLEAR").sum())
    noobs_uncertain = int((noobs["final_state"] == "UNCERTAIN").sum())
    noobs_obstacle = int((noobs["final_state"] == "OBSTACLE").sum())

    # Distance metrics for OBSTACLE with valid distance.
    valid_obstacle_dist = wall[(wall["final_state"] == "OBSTACLE") & wall["final_distance_m"].notna() & wall["distance_m"].notna()]
    if valid_obstacle_dist.empty:
        mae = rmse = float("nan")
    else:
        err_cm = np.abs((valid_obstacle_dist["final_distance_m"] - valid_obstacle_dist["distance_m"]) * 100.0).to_numpy(
            dtype=float
        )
        mae = float(np.mean(err_cm))
        rmse = float(np.sqrt(np.mean(err_cm**2)))

    wall_total = len(wall)
    coverage = float(len(valid_obstacle_dist) / max(wall_total, 1))
    safety_miss_rate = float(wall_clear / max(wall_total, 1))

    # Check wall_04940.
    t = df[df["filename"] == "wall_04940"]
    wall_04940_state = t["final_state"].iloc[0] if not t.empty else "missing"

    # Save outputs.
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "thresholds": {
            "neural_obstacle_probability_high": NEURAL_OBS_PROB_HIGH,
            "peak_snr_low": PEAK_SNR_LOW,
            "reflection_strength_low": REFLECTION_LOW,
            "noise_level_high": NOISE_HIGH,
        },
        "three_state_counts": {
            "wall_to_obstacle": wall_obstacle,
            "wall_to_uncertain": wall_uncertain,
            "wall_to_clear": wall_clear,
            "no_obstacle_to_clear": noobs_clear,
            "no_obstacle_to_uncertain": noobs_uncertain,
            "no_obstacle_to_obstacle": noobs_obstacle,
        },
        "distance_metrics_for_obstacle_with_valid_distance": {
            "mae_cm": mae,
            "rmse_cm": rmse,
            "num_samples": int(len(valid_obstacle_dist)),
        },
        "coverage_wall_with_valid_distance": coverage,
        "safety_miss_rate_wall_to_clear": safety_miss_rate,
        "num_uncertain_samples_total": int((df["final_state"] == "UNCERTAIN").sum()),
        "wall_04940_final_state": wall_04940_state,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Confusion table image (2x3): rows true class [wall, no_obstacle], cols [OBSTACLE, UNCERTAIN, CLEAR]
    table = np.array(
        [
            [wall_obstacle, wall_uncertain, wall_clear],
            [noobs_obstacle, noobs_uncertain, noobs_clear],
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    im = ax.imshow(table, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks([0, 1, 2], labels=["OBSTACLE", "UNCERTAIN", "CLEAR"])
    ax.set_yticks([0, 1], labels=["True WALL", "True NO_OBS"])
    ax.set_title("Three-State Confidence Table")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, str(table[i, j]), ha="center", va="center", fontsize=11, color="black")
    fig.tight_layout()
    fig.savefig(OUT_CONF_PNG)
    plt.close(fig)

    print("Three-state confusion counts:")
    print(f"  wall -> OBSTACLE: {wall_obstacle}")
    print(f"  wall -> UNCERTAIN: {wall_uncertain}")
    print(f"  wall -> CLEAR: {wall_clear}")
    print(f"  no_obstacle -> CLEAR: {noobs_clear}")
    print(f"  no_obstacle -> UNCERTAIN: {noobs_uncertain}")
    print(f"  no_obstacle -> OBSTACLE: {noobs_obstacle}")

    print("\nDistance metrics (OBSTACLE with valid distance):")
    print(f"  MAE: {mae:.2f} cm")
    print(f"  RMSE: {rmse:.2f} cm")
    print(f"  coverage (wall with valid distance): {coverage * 100:.2f}%")

    print("\nSafety:")
    print(f"  UNCERTAIN samples: {int((df['final_state'] == 'UNCERTAIN').sum())}")
    print(f"  safety miss rate (true wall -> CLEAR): {safety_miss_rate * 100:.2f}%")
    print(f"  wall_04940 final state: {wall_04940_state}")

    print(f"\nSaved JSON: {OUT_JSON}")
    print(f"Saved per-sample CSV: {OUT_CSV}")
    print(f"Saved confusion table image: {OUT_CONF_PNG}")


if __name__ == "__main__":
    main()
