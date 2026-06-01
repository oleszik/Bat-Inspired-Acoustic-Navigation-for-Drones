"""
Extract matched-filter peak features for hard_v1 regression experiments.

This script computes correlation-based descriptors from each WAV file and saves:
  datasets/synthetic_echoes_regression_hard_v1/peak_features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import chirp, correlate, find_peaks


DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
LABELS_CSV = DATASET_ROOT / "labels.csv"
OUT_CSV = DATASET_ROOT / "peak_features.csv"

SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0

DIRECT_IGNORE_S = 0.00055
DIST_MIN_M = 0.05
DIST_MAX_M = 2.80

RELATIVE_THRESHOLD = 0.35
NOISE_FLOOR_K = 6.0
NOISE_MIN_RELATIVE = 0.08


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


def delay_to_distance(delay_s: float) -> float:
    return delay_s * SPEED_OF_SOUND / 2.0


def compute_search_region(signal: np.ndarray, tx_chirp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corr = correlate(signal, tx_chirp, mode="full", method="fft")
    corr_mag = np.abs(corr).astype(np.float32)

    lags = np.arange(-len(tx_chirp) + 1, len(signal), dtype=np.int64)
    delays_s = lags.astype(np.float64) / SAMPLE_RATE

    min_delay_s = 2.0 * DIST_MIN_M / SPEED_OF_SOUND
    max_delay_s = 2.0 * DIST_MAX_M / SPEED_OF_SOUND

    mask = (delays_s >= min_delay_s) & (delays_s <= max_delay_s)
    delays_s = delays_s[mask]
    corr_mag = corr_mag[mask].copy()

    corr_mag[delays_s < DIRECT_IGNORE_S] = 0.0
    return delays_s, corr_mag


def first_peak_above_threshold(delays_s: np.ndarray, vals: np.ndarray, threshold: float) -> tuple[float, float]:
    peaks, _ = find_peaks(vals)
    for p in peaks:
        if vals[p] >= threshold:
            return float(delays_s[p]), float(vals[p])
    return np.nan, np.nan


def extract_peak_features_for_signal(signal: np.ndarray, tx_chirp: np.ndarray) -> dict[str, float]:
    delays_s, vals = compute_search_region(signal, tx_chirp)

    if vals.size == 0:
        return {k: np.nan for k in FEATURE_COLUMNS}

    peaks, _ = find_peaks(vals)
    num_peaks = int(peaks.size)

    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    robust_sigma = 1.4826 * mad
    noise_floor_threshold = median + NOISE_FLOOR_K * robust_sigma
    noise_floor_threshold = max(noise_floor_threshold, NOISE_MIN_RELATIVE * float(np.max(vals)))

    max_val = float(np.max(vals))
    rel_threshold = RELATIVE_THRESHOLD * max_val

    if num_peaks > 0:
        peak_vals = vals[peaks]
        strongest_local_idx = int(np.argmax(peak_vals))
        strongest_peak_idx = int(peaks[strongest_local_idx])
        strongest_delay_s = float(delays_s[strongest_peak_idx])
        strongest_value = float(vals[strongest_peak_idx])
    else:
        strongest_delay_s = np.nan
        strongest_value = np.nan

    first_rel_delay_s, first_rel_val = first_peak_above_threshold(delays_s, vals, rel_threshold)
    first_noise_delay_s, first_noise_val = first_peak_above_threshold(delays_s, vals, noise_floor_threshold)

    # Top-3 peaks by value.
    top_delays_ms = [np.nan, np.nan, np.nan]
    top_values = [np.nan, np.nan, np.nan]
    if num_peaks > 0:
        peak_vals = vals[peaks]
        order = np.argsort(peak_vals)[::-1]
        top_count = min(3, num_peaks)
        for i in range(top_count):
            idx = int(peaks[int(order[i])])
            top_delays_ms[i] = float(delays_s[idx] * 1000.0)
            top_values[i] = float(vals[idx])

    top1_val = top_values[0]
    if np.isnan(top1_val):
        peak_snr = np.nan
        top2_over_top1 = np.nan
        top3_over_top1 = np.nan
    else:
        peak_snr = float(top1_val / max(noise_floor_threshold, 1e-12))
        top2_over_top1 = float(top_values[1] / top1_val) if not np.isnan(top_values[1]) else np.nan
        top3_over_top1 = float(top_values[2] / top1_val) if not np.isnan(top_values[2]) else np.nan

    top2_minus_top1 = (
        float(top_delays_ms[1] - top_delays_ms[0])
        if (not np.isnan(top_delays_ms[1]) and not np.isnan(top_delays_ms[0]))
        else np.nan
    )
    top3_minus_top1 = (
        float(top_delays_ms[2] - top_delays_ms[0])
        if (not np.isnan(top_delays_ms[2]) and not np.isnan(top_delays_ms[0]))
        else np.nan
    )

    return {
        "strongest_peak_delay_ms": float(strongest_delay_s * 1000.0) if not np.isnan(strongest_delay_s) else np.nan,
        "strongest_peak_distance_m": delay_to_distance(strongest_delay_s) if not np.isnan(strongest_delay_s) else np.nan,
        "strongest_peak_value": strongest_value,
        "first_relative_peak_delay_ms": float(first_rel_delay_s * 1000.0) if not np.isnan(first_rel_delay_s) else np.nan,
        "first_relative_peak_distance_m": delay_to_distance(first_rel_delay_s) if not np.isnan(first_rel_delay_s) else np.nan,
        "first_relative_peak_value": first_rel_val,
        "first_noise_floor_peak_delay_ms": (
            float(first_noise_delay_s * 1000.0) if not np.isnan(first_noise_delay_s) else np.nan
        ),
        "first_noise_floor_peak_distance_m": (
            delay_to_distance(first_noise_delay_s) if not np.isnan(first_noise_delay_s) else np.nan
        ),
        "first_noise_floor_peak_value": first_noise_val,
        "noise_floor": float(noise_floor_threshold),
        "peak_snr": peak_snr,
        "num_peaks": float(num_peaks),
        "top1_delay_ms": top_delays_ms[0],
        "top1_value": top_values[0],
        "top2_delay_ms": top_delays_ms[1],
        "top2_value": top_values[1],
        "top3_delay_ms": top_delays_ms[2],
        "top3_value": top_values[2],
        "top2_minus_top1_delay_ms": top2_minus_top1,
        "top3_minus_top1_delay_ms": top3_minus_top1,
        "top2_over_top1_value": top2_over_top1,
        "top3_over_top1_value": top3_over_top1,
    }


FEATURE_COLUMNS = [
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


def main() -> None:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")

    labels = pd.read_csv(LABELS_CSV)
    tx_chirp = make_transmit_chirp()
    rows: list[dict[str, object]] = []

    for i, row in labels.iterrows():
        wav_path = DATASET_ROOT / str(row["wav_path"])
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")

        signal = load_wav_float(wav_path)
        feat = extract_peak_features_for_signal(signal, tx_chirp)

        out = {k: feat[k] for k in FEATURE_COLUMNS}
        out.update(
            {
                "filename": row["filename"],
                "has_obstacle": int(row["has_obstacle"]),
                "distance_m": row["distance_m"],
                "reflection_strength": row["reflection_strength"],
                "noise_level": row["noise_level"],
                "surface_absorption": row["surface_absorption"],
                "has_secondary_reflection": row["has_secondary_reflection"],
                "secondary_delay_ms": row["secondary_delay_ms"],
                "secondary_strength": row["secondary_strength"],
            }
        )
        rows.append(out)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(labels)} files...")

    df = pd.DataFrame(rows)
    ordered_cols = [
        "filename",
        *FEATURE_COLUMNS,
        "has_obstacle",
        "distance_m",
        "reflection_strength",
        "noise_level",
        "surface_absorption",
        "has_secondary_reflection",
        "secondary_delay_ms",
        "secondary_strength",
    ]
    df = df[ordered_cols]
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved peak features: {OUT_CSV}")


if __name__ == "__main__":
    main()
