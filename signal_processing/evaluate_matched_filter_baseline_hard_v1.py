"""
Evaluate classical matched-filter baselines on hard_v1 regression test split.

This script compares three peak-picking strategies:
A) strongest peak
B) first peak above a relative threshold
C) first peak above a noise-floor threshold

No model retraining is performed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import chirp, correlate, find_peaks


# -----------------------------
# Paths
# -----------------------------
DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
LABELS_CSV = DATASET_ROOT / "labels.csv"
SPLIT_JSON = Path("neural_network/checkpoints/hard_v1/regression_split_indices.json")
RESULTS_DIR = Path("neural_network/results/hard_v1")

PER_SAMPLE_CSV = RESULTS_DIR / "matched_filter_baseline_per_sample.csv"
SUMMARY_JSON = RESULTS_DIR / "matched_filter_baseline_summary.json"

PLOT_TRUE_VS_PRED = RESULTS_DIR / "matched_filter_true_vs_predicted.png"
PLOT_HIST = RESULTS_DIR / "matched_filter_error_histograms.png"
PLOT_ERR_VS_REFL = RESULTS_DIR / "matched_filter_error_vs_reflection_strength.png"
PLOT_ERR_VS_SEC = RESULTS_DIR / "matched_filter_error_vs_secondary_reflection.png"


# -----------------------------
# Acoustic settings
# -----------------------------
SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0

DIRECT_IGNORE_S = 0.00055
DIST_MIN_M = 0.05
DIST_MAX_M = 2.80


# -----------------------------
# Peak-picking thresholds
# -----------------------------
RELATIVE_THRESHOLD = 0.35
NOISE_FLOOR_K = 6.0
NOISE_MIN_RELATIVE = 0.08


def make_transmit_chirp() -> np.ndarray:
    """Recreate transmit chirp used by the dataset."""
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


def load_wav_float(path: Path) -> np.ndarray:
    """Load WAV and convert to mono float."""
    sr, data = wavfile.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate in {path}: {sr}")
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    return data


def compute_search_region(signal: np.ndarray, tx_chirp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute matched-filter magnitude and keep only valid lag window.

    Valid window:
    - corresponding to distance range [0.05 m, 2.80 m]
    - ignore direct transmit region before 0.55 ms
    """
    corr = correlate(signal, tx_chirp, mode="full", method="fft")
    corr_mag = np.abs(corr).astype(np.float32)

    lags = np.arange(-len(tx_chirp) + 1, len(signal), dtype=np.int64)
    delays_s = lags.astype(np.float64) / SAMPLE_RATE

    min_delay_s = 2.0 * DIST_MIN_M / SPEED_OF_SOUND
    max_delay_s = 2.0 * DIST_MAX_M / SPEED_OF_SOUND

    valid = (delays_s >= min_delay_s) & (delays_s <= max_delay_s)
    delays_valid = delays_s[valid]
    corr_valid = corr_mag[valid].copy()

    # Remove direct pulse contamination.
    corr_valid[delays_valid < DIRECT_IGNORE_S] = 0.0

    return delays_valid, corr_valid


def distance_from_delay(delay_s: float) -> float:
    return delay_s * SPEED_OF_SOUND / 2.0


def pick_strongest_peak(delays_s: np.ndarray, corr_vals: np.ndarray) -> float | None:
    if corr_vals.size == 0:
        return None
    peak_idx = int(np.argmax(corr_vals))
    if corr_vals[peak_idx] <= 0.0:
        return None
    return float(delays_s[peak_idx])


def pick_first_relative_peak(delays_s: np.ndarray, corr_vals: np.ndarray) -> float | None:
    if corr_vals.size == 0:
        return None
    peaks, _ = find_peaks(corr_vals)
    if peaks.size == 0:
        return None
    threshold = RELATIVE_THRESHOLD * float(np.max(corr_vals))
    for p in peaks:
        if corr_vals[p] >= threshold:
            return float(delays_s[int(p)])
    return None


def pick_first_noise_floor_peak(delays_s: np.ndarray, corr_vals: np.ndarray) -> float | None:
    if corr_vals.size == 0:
        return None
    peaks, _ = find_peaks(corr_vals)
    if peaks.size == 0:
        return None

    median = float(np.median(corr_vals))
    mad = float(np.median(np.abs(corr_vals - median)))
    robust_sigma = 1.4826 * mad
    threshold = median + NOISE_FLOOR_K * robust_sigma
    threshold = max(threshold, NOISE_MIN_RELATIVE * float(np.max(corr_vals)))

    for p in peaks:
        if corr_vals[p] >= threshold:
            return float(delays_s[int(p)])
    return None


def summarize_wall_errors(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    wall = df[df["has_obstacle"] == 1].copy()
    detected = wall[wall[pred_col].notna()].copy()
    if detected.empty:
        return {
            "wall_detection_rate": 0.0,
            "mae_cm": float("nan"),
            "rmse_cm": float("nan"),
            "median_abs_error_cm": float("nan"),
            "max_abs_error_cm": float("nan"),
            "p90_abs_error_cm": float("nan"),
            "p95_abs_error_cm": float("nan"),
        }

    err_cm = np.abs((detected[pred_col] - detected["distance_m"]).to_numpy(dtype=float) * 100.0)
    return {
        "wall_detection_rate": float(len(detected) / max(len(wall), 1)),
        "mae_cm": float(np.mean(err_cm)),
        "rmse_cm": float(np.sqrt(np.mean(err_cm**2))),
        "median_abs_error_cm": float(np.median(err_cm)),
        "max_abs_error_cm": float(np.max(err_cm)),
        "p90_abs_error_cm": float(np.percentile(err_cm, 90)),
        "p95_abs_error_cm": float(np.percentile(err_cm, 95)),
    }


def false_positive_rate_no_obstacle(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    no_obs = df[df["has_obstacle"] == 0]
    n = len(no_obs)
    detections = int(no_obs[pred_col].notna().sum())
    return {
        "num_no_obstacle": n,
        "false_echo_detections": detections,
        "false_positive_rate": float(detections / max(n, 1)),
    }


def add_error_column(df: pd.DataFrame, pred_col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = np.nan
    wall = df["has_obstacle"] == 1
    detected = wall & df[pred_col].notna()
    df.loc[detected, out_col] = (
        np.abs(df.loc[detected, pred_col].astype(float) - df.loc[detected, "distance_m"].astype(float)) * 100.0
    )
    return df


def plot_true_vs_pred(df: pd.DataFrame) -> None:
    methods = [
        ("pred_distance_strongest_m", "Strongest Peak"),
        ("pred_distance_first_relative_m", "First Relative Peak"),
        ("pred_distance_first_noise_m", "First Noise-Floor Peak"),
    ]
    wall = df[df["has_obstacle"] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)
    for ax, (col, title) in zip(axes, methods):
        sub = wall[wall[col].notna()]
        ax.scatter(sub["distance_m"], sub[col], s=10, alpha=0.55)
        if not sub.empty:
            lo = float(min(sub["distance_m"].min(), sub[col].min()))
            hi = float(max(sub["distance_m"].max(), sub[col].max()))
            ax.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("True distance (m)")
        ax.set_ylabel("Predicted distance (m)")
    fig.suptitle("Matched-Filter Baseline: True vs Predicted Distance")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_TRUE_VS_PRED)
    plt.close(fig)


def plot_error_hist(df: pd.DataFrame) -> None:
    methods = [
        ("abs_error_strongest_cm", "Strongest Peak"),
        ("abs_error_first_relative_cm", "First Relative Peak"),
        ("abs_error_first_noise_cm", "First Noise-Floor Peak"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)
    for ax, (col, title) in zip(axes, methods):
        vals = df[col].dropna().to_numpy(dtype=float)
        ax.hist(vals, bins=40, color="tab:blue", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Absolute error (cm)")
        ax.set_ylabel("Count")
    fig.suptitle("Matched-Filter Baseline: Error Histograms")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_HIST)
    plt.close(fig)


def plot_error_vs_reflection(df: pd.DataFrame) -> None:
    methods = [
        ("abs_error_strongest_cm", "Strongest Peak"),
        ("abs_error_first_relative_cm", "First Relative Peak"),
        ("abs_error_first_noise_cm", "First Noise-Floor Peak"),
    ]
    wall = df[df["has_obstacle"] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)
    for ax, (col, title) in zip(axes, methods):
        sub = wall[wall[col].notna()]
        ax.scatter(sub["reflection_strength"], sub[col], s=10, alpha=0.55)
        ax.set_title(title)
        ax.set_xlabel("Reflection strength")
        ax.set_ylabel("Absolute error (cm)")
    fig.suptitle("Matched-Filter Baseline: Error vs Reflection Strength")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_ERR_VS_REFL)
    plt.close(fig)


def plot_error_vs_secondary(df: pd.DataFrame) -> None:
    methods = [
        ("abs_error_strongest_cm", "Strongest Peak"),
        ("abs_error_first_relative_cm", "First Relative Peak"),
        ("abs_error_first_noise_cm", "First Noise-Floor Peak"),
    ]
    wall = df[df["has_obstacle"] == 1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)
    for ax, (col, title) in zip(axes, methods):
        sub = wall[wall[col].notna()].copy()
        with_sec = sub[sub["has_secondary_reflection"] == 1][col].to_numpy(dtype=float)
        without_sec = sub[sub["has_secondary_reflection"] == 0][col].to_numpy(dtype=float)
        ax.boxplot([without_sec, with_sec], tick_labels=["No secondary", "Secondary"])
        ax.set_title(title)
        ax.set_ylabel("Absolute error (cm)")
    fig.suptitle("Matched-Filter Baseline: Error vs Secondary Reflection")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_ERR_VS_SEC)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(f"Missing split JSON: {SPLIT_JSON}")

    df_labels = pd.read_csv(LABELS_CSV)
    with SPLIT_JSON.open("r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_indices = split_data["test_indices"]
    df_test = df_labels.iloc[test_indices].copy().reset_index(drop=True)

    tx_chirp = make_transmit_chirp()

    per_sample_rows: list[dict[str, object]] = []
    for i, row in df_test.iterrows():
        wav_path = DATASET_ROOT / str(row["wav_path"])
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")

        signal = load_wav_float(wav_path)
        delays_s, corr_vals = compute_search_region(signal, tx_chirp)

        d_strong = pick_strongest_peak(delays_s, corr_vals)
        d_rel = pick_first_relative_peak(delays_s, corr_vals)
        d_noise = pick_first_noise_floor_peak(delays_s, corr_vals)

        dist_strong = np.nan if d_strong is None else distance_from_delay(d_strong)
        dist_rel = np.nan if d_rel is None else distance_from_delay(d_rel)
        dist_noise = np.nan if d_noise is None else distance_from_delay(d_noise)

        true_dist = row["distance_m"] if int(row["has_obstacle"]) == 1 else np.nan
        if pd.isna(true_dist):
            true_dist = np.nan
        else:
            true_dist = float(true_dist)

        per_sample_rows.append(
            {
                "filename": row["filename"],
                "has_obstacle": int(row["has_obstacle"]),
                "distance_m": true_dist,
                "reflection_strength": float(row["reflection_strength"]),
                "noise_level": float(row["noise_level"]),
                "surface_absorption": float(row["surface_absorption"]),
                "has_secondary_reflection": int(row["has_secondary_reflection"]),
                "secondary_delay_ms": row["secondary_delay_ms"],
                "secondary_strength": row["secondary_strength"],
                "pred_distance_strongest_m": dist_strong,
                "pred_distance_first_relative_m": dist_rel,
                "pred_distance_first_noise_m": dist_noise,
            }
        )

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(df_test)} test samples...")

    df = pd.DataFrame(per_sample_rows)
    df = add_error_column(df, "pred_distance_strongest_m", "abs_error_strongest_cm")
    df = add_error_column(df, "pred_distance_first_relative_m", "abs_error_first_relative_cm")
    df = add_error_column(df, "pred_distance_first_noise_m", "abs_error_first_noise_cm")

    df.to_csv(PER_SAMPLE_CSV, index=False)

    summary = {
        "matched_filter_strongest_peak": {
            **summarize_wall_errors(df, "pred_distance_strongest_m"),
            **false_positive_rate_no_obstacle(df, "pred_distance_strongest_m"),
        },
        "matched_filter_first_relative_peak": {
            **summarize_wall_errors(df, "pred_distance_first_relative_m"),
            **false_positive_rate_no_obstacle(df, "pred_distance_first_relative_m"),
        },
        "matched_filter_first_noise_floor_peak": {
            **summarize_wall_errors(df, "pred_distance_first_noise_m"),
            **false_positive_rate_no_obstacle(df, "pred_distance_first_noise_m"),
        },
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_true_vs_pred(df)
    plot_error_hist(df)
    plot_error_vs_reflection(df)
    plot_error_vs_secondary(df)

    print("\nComparison Table")
    print("Spectrogram-only CNN: MAE 7.10 cm, RMSE 11.43 cm, max 108.10 cm")
    print("Dual-input CNN:       MAE 6.76 cm, RMSE 10.64 cm, max 88.15 cm")

    s = summary["matched_filter_strongest_peak"]
    r = summary["matched_filter_first_relative_peak"]
    n = summary["matched_filter_first_noise_floor_peak"]

    print(
        "Matched-filter strongest peak: "
        f"MAE {s['mae_cm']:.2f} cm, RMSE {s['rmse_cm']:.2f} cm, max {s['max_abs_error_cm']:.2f} cm"
    )
    print(
        "Matched-filter first relative peak: "
        f"MAE {r['mae_cm']:.2f} cm, RMSE {r['rmse_cm']:.2f} cm, max {r['max_abs_error_cm']:.2f} cm"
    )
    print(
        "Matched-filter noise-floor peak: "
        f"MAE {n['mae_cm']:.2f} cm, RMSE {n['rmse_cm']:.2f} cm, max {n['max_abs_error_cm']:.2f} cm"
    )

    print(f"\nSaved per-sample results: {PER_SAMPLE_CSV}")
    print(f"Saved summary JSON: {SUMMARY_JSON}")
    print(f"Saved plots: {PLOT_TRUE_VS_PRED}, {PLOT_HIST}, {PLOT_ERR_VS_REFL}, {PLOT_ERR_VS_SEC}")


if __name__ == "__main__":
    main()
