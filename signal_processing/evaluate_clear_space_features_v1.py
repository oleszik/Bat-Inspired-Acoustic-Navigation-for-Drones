"""
Evaluate matched-filter feature separability on clear-space diagnostic dataset v1.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import chirp, correlate, find_peaks, peak_widths


DATASET_ROOT = Path("datasets/clear_space_diagnostic_v1")
LABELS_CSV = DATASET_ROOT / "labels.csv"
SUMMARY_JSON = DATASET_ROOT / "clear_space_feature_summary.json"
FEATURES_CSV = DATASET_ROOT / "clear_space_features.csv"

PLOT_SNR = DATASET_ROOT / "clear_space_peak_snr_by_class.png"
PLOT_PROM = DATASET_ROOT / "clear_space_peak_prominence_by_class.png"
PLOT_NPEAK = DATASET_ROOT / "clear_space_num_peaks_by_class.png"
PLOT_WIDTH = DATASET_ROOT / "clear_space_peak_width_by_class.png"


SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0
DIRECT_IGNORE_S = 0.00055
DIST_MIN_M = 0.05
DIST_MAX_M = 2.80

NOISE_FLOOR_K = 6.0
NOISE_MIN_REL = 0.08


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


def search_region(signal: np.ndarray, tx_chirp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def robust_threshold(vals: np.ndarray, k: float, min_rel: float) -> tuple[float, float, float, float]:
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    sigma = 1.4826 * mad
    vmax = float(np.max(vals))
    thr = max(median + k * sigma, min_rel * vmax)
    return thr, median, mad, vmax


def extract_features(signal: np.ndarray, tx_chirp: np.ndarray) -> dict[str, float]:
    d, v = search_region(signal, tx_chirp)
    peaks, props = find_peaks(v, prominence=0.0)
    num_peaks = int(peaks.size)

    nf_thr, _, _, vmax = robust_threshold(v, NOISE_FLOOR_K, NOISE_MIN_REL)

    # Strongest peak.
    if num_peaks > 0:
        peak_vals = v[peaks]
        sidx_local = int(np.argmax(peak_vals))
        sidx = int(peaks[sidx_local])
        strongest_peak_value = float(v[sidx])
        strongest_peak_delay_ms = float(d[sidx] * 1000.0)
        strongest_prom = float(props["prominences"][sidx_local]) if "prominences" in props else 0.0
        widths = peak_widths(v, [sidx], rel_height=0.5)[0]
        peak_width_ms = float(widths[0] / SAMPLE_RATE * 1000.0) if len(widths) > 0 else np.nan
    else:
        strongest_peak_value = np.nan
        strongest_peak_delay_ms = np.nan
        strongest_prom = np.nan
        peak_width_ms = np.nan

    # First noise-floor peak.
    first_nf_val = np.nan
    first_nf_delay_ms = np.nan
    if num_peaks > 0:
        for p in peaks:
            if v[p] >= nf_thr:
                first_nf_val = float(v[p])
                first_nf_delay_ms = float(d[p] * 1000.0)
                break

    peak_snr = float(strongest_peak_value / max(nf_thr, 1e-12)) if not np.isnan(strongest_peak_value) else np.nan

    return {
        "strongest_peak_value": strongest_peak_value,
        "strongest_peak_delay_ms": strongest_peak_delay_ms,
        "first_noise_floor_peak_value": first_nf_val,
        "first_noise_floor_peak_delay_ms": first_nf_delay_ms,
        "peak_snr": peak_snr,
        "peak_prominence": strongest_prom,
        "peak_width": peak_width_ms,
        "noise_floor": float(nf_thr),
        "num_peaks": float(num_peaks),
    }


def numeric_summary(df: pd.DataFrame, col: str) -> dict[str, float]:
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "q10": np.nan, "q50": np.nan, "q90": np.nan}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q10": float(np.percentile(x, 10)),
        "q50": float(np.percentile(x, 50)),
        "q90": float(np.percentile(x, 90)),
    }


def separation_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return {"cohen_d": np.nan, "overlap_fraction": np.nan}

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a)), float(np.var(b))
    pooled = np.sqrt((var_a + var_b) / 2.0)
    cohen_d = (mean_a - mean_b) / pooled if pooled > 1e-12 else np.nan

    a_min, a_max = float(np.min(a)), float(np.max(a))
    b_min, b_max = float(np.min(b)), float(np.max(b))
    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    span = max(a_max, b_max) - min(a_min, b_min)
    overlap_fraction = overlap / span if span > 1e-12 else 0.0
    return {"cohen_d": float(cohen_d), "overlap_fraction": float(overlap_fraction)}


def boxplot_by_class(df: pd.DataFrame, col: str, out_path: Path, title: str) -> None:
    classes = [
        "obstacle_easy",
        "obstacle_weak",
        "clear_clean",
        "clear_noisy",
        "clear_motor_noise",
        "clear_random_spikes",
    ]
    data = [pd.to_numeric(df[df["class_name"] == c][col], errors="coerce").dropna().to_numpy(dtype=float) for c in classes]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=140)
    ax.boxplot(data, tick_labels=classes, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(col)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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
        sig = load_wav_float(wav_path)
        feat = extract_features(sig, tx_chirp)

        out = {
            "filename": row["filename"],
            "class_name": row["class_name"],
            **feat,
        }
        rows.append(out)

        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{len(labels)} files...")

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(FEATURES_CSV, index=False)

    # Pairwise distribution comparisons.
    pairs = [
        ("obstacle_easy", "clear_clean"),
        ("obstacle_weak", "clear_noisy"),
        ("obstacle_weak", "clear_motor_noise"),
        ("obstacle_weak", "clear_random_spikes"),
    ]
    feature_cols = [
        "strongest_peak_value",
        "strongest_peak_delay_ms",
        "first_noise_floor_peak_value",
        "first_noise_floor_peak_delay_ms",
        "peak_snr",
        "peak_prominence",
        "peak_width",
        "noise_floor",
        "num_peaks",
    ]

    per_class_summary = {}
    for cls in feat_df["class_name"].unique():
        cdf = feat_df[feat_df["class_name"] == cls]
        per_class_summary[cls] = {f: numeric_summary(cdf, f) for f in feature_cols}

    pair_summary = {}
    for a, b in pairs:
        adf = feat_df[feat_df["class_name"] == a]
        bdf = feat_df[feat_df["class_name"] == b]
        key = f"{a}_vs_{b}"
        pair_summary[key] = {}
        for f in feature_cols:
            av = pd.to_numeric(adf[f], errors="coerce").to_numpy(dtype=float)
            bv = pd.to_numeric(bdf[f], errors="coerce").to_numpy(dtype=float)
            pair_summary[key][f] = separation_metrics(av, bv)

    # Plots requested.
    boxplot_by_class(feat_df, "peak_snr", PLOT_SNR, "Peak SNR by Class")
    boxplot_by_class(feat_df, "peak_prominence", PLOT_PROM, "Peak Prominence by Class")
    boxplot_by_class(feat_df, "num_peaks", PLOT_NPEAK, "Number of Peaks by Class")
    boxplot_by_class(feat_df, "peak_width", PLOT_WIDTH, "Peak Width by Class")

    summary = {
        "dataset_size": int(len(feat_df)),
        "classes": sorted(feat_df["class_name"].unique().tolist()),
        "features": feature_cols,
        "per_class_summary": per_class_summary,
        "pairwise_comparisons": pair_summary,
        "plot_files": {
            "peak_snr_by_class": str(PLOT_SNR),
            "peak_prominence_by_class": str(PLOT_PROM),
            "num_peaks_by_class": str(PLOT_NPEAK),
            "peak_width_by_class": str(PLOT_WIDTH),
        },
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON: {SUMMARY_JSON}")
    print(f"Saved feature CSV: {FEATURES_CSV}")
    print(f"Saved plots: {PLOT_SNR}, {PLOT_PROM}, {PLOT_NPEAK}, {PLOT_WIDTH}")


if __name__ == "__main__":
    main()
