"""
Evaluate weak-echo fallback matched-filter detector on hard_v1 test split.

Purpose:
- Keep strict normal noise-floor detection.
- When normal detection fails, run a weaker but conservative fallback that searches
  for plausible early local maxima with minimum prominence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import chirp, correlate, find_peaks


DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
LABELS_CSV = DATASET_ROOT / "labels.csv"
WAV_ROOT = DATASET_ROOT / "wav"

SPLIT_JSON = Path("neural_network/checkpoints/hard_v1/regression_split_indices.json")
MATCHED_BASELINE_CSV = Path("neural_network/results/hard_v1/matched_filter_baseline_per_sample.csv")
SAFETY_HYBRID_CSV = Path("neural_network/results/hard_v1_safety_hybrid/safety_hybrid_per_sample_predictions.csv")

OUT_DIR = Path("neural_network/results/hard_v1_weak_echo")
OUT_PER_SAMPLE = OUT_DIR / "weak_echo_fallback_per_sample.csv"
OUT_SUMMARY = OUT_DIR / "weak_echo_fallback_summary.json"
OUT_HIST = OUT_DIR / "weak_echo_error_histogram.png"
OUT_SCATTER = OUT_DIR / "weak_echo_true_vs_predicted.png"


SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0

DIRECT_IGNORE_S = 0.00055
DIST_MIN_M = 0.05
DIST_MAX_M = 2.80

# Normal noise-floor detector (same spirit as previous baseline).
NOISE_FLOOR_K = 6.0
NOISE_MIN_REL = 0.08

# Weak fallback detector thresholds.
FALLBACK_K = 3.0
FALLBACK_MIN_REL = 0.03
FALLBACK_MIN_PROM_REL = 0.015
FALLBACK_MIN_PROM_ABS = 1e-6


def make_transmit_chirp() -> np.ndarray:
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear").astype(np.float32)


def load_wav_float(path: Path) -> np.ndarray:
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
    corr = correlate(signal, tx_chirp, mode="full", method="fft")
    corr_mag = np.abs(corr).astype(np.float32)

    lags = np.arange(-len(tx_chirp) + 1, len(signal), dtype=np.int64)
    delays_s = lags.astype(np.float64) / SAMPLE_RATE

    min_delay_s = 2.0 * DIST_MIN_M / SPEED_OF_SOUND
    max_delay_s = 2.0 * DIST_MAX_M / SPEED_OF_SOUND
    keep = (delays_s >= min_delay_s) & (delays_s <= max_delay_s)

    d = delays_s[keep]
    v = corr_mag[keep].copy()
    v[d < DIRECT_IGNORE_S] = 0.0
    return d, v


def delay_to_distance(delay_s: float) -> float:
    return delay_s * SPEED_OF_SOUND / 2.0


def robust_noise_threshold(vals: np.ndarray, k: float, min_rel: float) -> tuple[float, float, float, float]:
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    robust_sigma = 1.4826 * mad
    vmax = float(np.max(vals))
    thr = max(median + k * robust_sigma, min_rel * vmax)
    return thr, median, mad, vmax


def detect_normal_noise_floor(delays_s: np.ndarray, vals: np.ndarray) -> tuple[float | None, float | None, dict]:
    peaks, _ = find_peaks(vals)
    thr, median, mad, vmax = robust_noise_threshold(vals, NOISE_FLOOR_K, NOISE_MIN_REL)
    for p in peaks:
        if vals[p] >= thr:
            return float(delays_s[p]), float(vals[p]), {"threshold": thr, "median": median, "mad": mad, "max": vmax}
    return None, None, {"threshold": thr, "median": median, "mad": mad, "max": vmax}


def detect_weak_fallback(delays_s: np.ndarray, vals: np.ndarray) -> tuple[float | None, float | None, float, dict]:
    """
    Weak fallback:
    - lower threshold than normal noise-floor
    - requires local prominence (reject random spikes)
    - choose earliest plausible peak (safety-oriented)
    """
    fallback_thr, median, mad, vmax = robust_noise_threshold(vals, FALLBACK_K, FALLBACK_MIN_REL)
    min_prom = max(FALLBACK_MIN_PROM_ABS, FALLBACK_MIN_PROM_REL * vmax)

    peaks, props = find_peaks(vals, prominence=min_prom)
    if peaks.size == 0:
        return None, None, 0.0, {"threshold": fallback_thr, "min_prom": min_prom, "max": vmax}

    prominences = props.get("prominences", np.zeros_like(peaks, dtype=np.float64))
    # Earliest plausible above fallback threshold.
    for p, prom in sorted(zip(peaks, prominences), key=lambda x: x[0]):
        pv = float(vals[int(p)])
        if pv >= fallback_thr:
            delay = float(delays_s[int(p)])
            # Confidence from peak height and prominence relative to max correlation.
            conf_h = pv / max(vmax, 1e-12)
            conf_p = float(prom) / max(vmax, 1e-12)
            conf = float(min(1.0, 0.6 * conf_h + 0.4 * conf_p))
            return delay, pv, conf, {"threshold": fallback_thr, "min_prom": min_prom, "max": vmax}

    return None, None, 0.0, {"threshold": fallback_thr, "min_prom": min_prom, "max": vmax}


def main() -> None:
    for p in [LABELS_CSV, SPLIT_JSON, MATCHED_BASELINE_CSV, SAFETY_HYBRID_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    if not WAV_ROOT.exists():
        raise FileNotFoundError(f"Missing WAV directory: {WAV_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(LABELS_CSV)
    with SPLIT_JSON.open("r", encoding="utf-8") as f:
        split = json.load(f)
    test_idx = split["test_indices"]
    df_test = labels.iloc[test_idx].copy().reset_index(drop=True)

    # Baseline/safety CSVs are loaded for comparative counters and cross-checks.
    df_mf = pd.read_csv(MATCHED_BASELINE_CSV)
    df_mf = df_mf[["filename", "pred_distance_first_noise_m"]].rename(
        columns={"pred_distance_first_noise_m": "baseline_noise_floor_distance_m"}
    )
    df_safety = pd.read_csv(SAFETY_HYBRID_CSV)
    if "S4_pred_has_obstacle" not in df_safety.columns:
        df_safety["S4_pred_has_obstacle"] = np.nan
    df_safety = df_safety[["filename", "S4_pred_has_obstacle"]]

    tx_chirp = make_transmit_chirp()
    rows: list[dict[str, object]] = []

    for i, row in df_test.iterrows():
        wav_path = DATASET_ROOT / str(row["wav_path"])
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")

        signal = load_wav_float(wav_path)
        delays_s, vals = compute_search_region(signal, tx_chirp)

        normal_delay_s, normal_val, normal_dbg = detect_normal_noise_floor(delays_s, vals)
        if normal_delay_s is not None:
            weak_delay_s = None
            weak_val = None
            weak_conf = 0.0
            mode = "normal_noise_floor"
            selected_dist = delay_to_distance(normal_delay_s)
            weak_detected = 0
        else:
            weak_delay_s, weak_val, weak_conf, weak_dbg = detect_weak_fallback(delays_s, vals)
            if weak_delay_s is not None:
                mode = "weak_echo_fallback"
                selected_dist = delay_to_distance(weak_delay_s)
                weak_detected = 1
            else:
                mode = "no_echo_detected"
                selected_dist = np.nan
                weak_detected = 0

        has_obs = int(row["has_obstacle"])
        true_dist = pd.to_numeric(row["distance_m"], errors="coerce")
        if has_obs == 1 and pd.notna(true_dist) and pd.notna(selected_dist):
            abs_err_cm = abs(float(selected_dist) - float(true_dist)) * 100.0
        else:
            abs_err_cm = np.nan

        rows.append(
            {
                "filename": row["filename"],
                "true_has_obstacle": has_obs,
                "true_distance_m": true_dist,
                "normal_noise_floor_distance_m": (delay_to_distance(normal_delay_s) if normal_delay_s is not None else np.nan),
                "weak_fallback_distance_m": (delay_to_distance(weak_delay_s) if weak_delay_s is not None else np.nan),
                "weak_fallback_detected": weak_detected,
                "weak_fallback_confidence": weak_conf,
                "selected_distance_m": selected_dist,
                "selected_mode": mode,
                "absolute_error_cm": abs_err_cm,
            }
        )

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(df_test)} test WAVs...")

    df = pd.DataFrame(rows)
    df = df.merge(df_mf, on="filename", how="left")
    df = df.merge(df_safety, on="filename", how="left")

    # Detailed print for wall_04940.
    target = df[df["filename"] == "wall_04940"]
    if not target.empty:
        t = target.iloc[0]
        print("\nDetailed detection for wall_04940:")
        print(f"filename: {t['filename']}")
        print(f"true_has_obstacle: {int(t['true_has_obstacle'])}")
        print(f"true_distance_m: {t['true_distance_m']}")
        print(f"normal_noise_floor_distance_m: {t['normal_noise_floor_distance_m']}")
        print(f"weak_fallback_distance_m: {t['weak_fallback_distance_m']}")
        print(f"weak_fallback_detected: {int(t['weak_fallback_detected'])}")
        print(f"weak_fallback_confidence: {t['weak_fallback_confidence']}")
        print(f"selected_distance_m: {t['selected_distance_m']}")
        print(f"selected_mode: {t['selected_mode']}")
        print(f"absolute_error_cm: {t['absolute_error_cm']}")

    wall = df[df["true_has_obstacle"] == 1].copy()
    wall_detected = wall[wall["selected_distance_m"].notna()].copy()
    no_obs = df[df["true_has_obstacle"] == 0].copy()

    if wall_detected.empty:
        mae = rmse = p90 = p95 = mx = float("nan")
    else:
        err = np.abs((wall_detected["selected_distance_m"] - wall_detected["true_distance_m"]) * 100.0).to_numpy(dtype=float)
        mae = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err**2)))
        p90 = float(np.percentile(err, 90))
        p95 = float(np.percentile(err, 95))
        mx = float(np.max(err))

    fpr = float((no_obs["selected_distance_m"].notna()).sum() / max(len(no_obs), 1))

    # Recovery counters.
    normal_missing_wall = wall["normal_noise_floor_distance_m"].isna()
    recovered_fn = int((normal_missing_wall & (wall["weak_fallback_detected"] == 1)).sum())

    normal_missing_noobs = no_obs["normal_noise_floor_distance_m"].isna()
    new_fp = int((normal_missing_noobs & (no_obs["weak_fallback_detected"] == 1)).sum())

    summary = {
        "wall_metrics": {
            "mae_cm": mae,
            "rmse_cm": rmse,
            "p90_cm": p90,
            "p95_cm": p95,
            "max_cm": mx,
            "num_wall_samples": int(len(wall)),
            "num_wall_detected": int(len(wall_detected)),
        },
        "no_obstacle_false_positive_rate": fpr,
        "num_no_obstacle_samples": int(len(no_obs)),
        "false_positive_count": int(no_obs["selected_distance_m"].notna().sum()),
        "recovered_false_negatives": recovered_fn,
        "new_false_positives": new_fp,
    }

    df.to_csv(OUT_PER_SAMPLE, index=False)
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    plt.figure(figsize=(7, 4), dpi=140)
    if not wall_detected.empty:
        plt.hist(
            np.abs((wall_detected["selected_distance_m"] - wall_detected["true_distance_m"]) * 100.0).to_numpy(dtype=float),
            bins=40,
            color="tab:blue",
            alpha=0.85,
        )
    plt.xlabel("Absolute distance error (cm)")
    plt.ylabel("Count")
    plt.title("Weak-Echo Fallback Error Histogram (Wall Samples)")
    plt.tight_layout()
    plt.savefig(OUT_HIST)
    plt.close()

    plt.figure(figsize=(6, 6), dpi=140)
    if not wall_detected.empty:
        plt.scatter(wall_detected["true_distance_m"], wall_detected["selected_distance_m"], s=12, alpha=0.65)
        lo = float(min(wall_detected["true_distance_m"].min(), wall_detected["selected_distance_m"].min()))
        hi = float(max(wall_detected["true_distance_m"].max(), wall_detected["selected_distance_m"].max()))
        plt.plot([lo, hi], [lo, hi], "--", color="red", linewidth=1.2, label="Ideal y=x")
        plt.legend()
    plt.xlabel("True distance (m)")
    plt.ylabel("Predicted distance (m)")
    plt.title("Weak-Echo Fallback: True vs Predicted")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER)
    plt.close()

    print("\nWeak-echo fallback summary:")
    print(
        f"Wall MAE={mae:.2f} cm | RMSE={rmse:.2f} cm | P90={p90:.2f} cm | "
        f"P95={p95:.2f} cm | Max={mx:.2f} cm"
    )
    print(f"No-obstacle FPR={fpr:.4f}")
    print(f"Recovered false negatives={recovered_fn}")
    print(f"New false positives={new_fp}")
    print(f"Saved per-sample CSV: {OUT_PER_SAMPLE}")
    print(f"Saved summary JSON: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
