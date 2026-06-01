"""
Calibrate rule-based CLEAR/UNCERTAIN thresholds for multisector evaluator v1.

This script:
1) loads sector predictions + labels,
2) recomputes missing diagnostic matched-filter features from WAV (if needed),
3) sweeps transparent CLEAR thresholds,
4) selects the best configuration with safety-first constraints,
5) saves sweep + summary + calibration plots.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import chirp, correlate, find_peaks


DATASET_ROOT = Path("datasets/synthetic_echoes_multisector_v1")
LABELS_CSV = DATASET_ROOT / "labels.csv"
PRED_CSV = DATASET_ROOT / "multisector_three_state_predictions.csv"

OUT_JSON = DATASET_ROOT / "multisector_clear_threshold_calibration.json"
OUT_SWEEP_CSV = DATASET_ROOT / "multisector_clear_threshold_sweep.csv"
OUT_PLOT_RECOVERY = DATASET_ROOT / "multisector_calibration_clear_recovery_vs_safety_miss.png"
OUT_PLOT_FALSE_OBS = DATASET_ROOT / "multisector_calibration_false_obstacle_vs_safety_miss.png"
OUT_PLOT_TOP = DATASET_ROOT / "multisector_calibration_top_thresholds.png"

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

WEAK_K = 3.0
WEAK_MIN_REL = 0.03
WEAK_MIN_PROM_REL = 0.015
WEAK_MIN_PROM_ABS = 1e-6


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

    min_delay = 2.0 * DIST_MIN_M / SPEED_OF_SOUND
    max_delay = 2.0 * DIST_MAX_M / SPEED_OF_SOUND
    keep = (delays_s >= min_delay) & (delays_s <= max_delay)
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


def compute_diagnostics(signal: np.ndarray, tx_chirp: np.ndarray) -> dict[str, float | int]:
    d, v = search_region(signal, tx_chirp)
    peaks, _ = find_peaks(v)

    nf_thr, _, _, vmax = robust_threshold(v, NOISE_FLOOR_K, NOISE_MIN_REL)

    matched_exists = 0
    matched_peak_value = np.nan
    matched_delay_ms = np.nan
    for p in peaks:
        if v[p] >= nf_thr:
            matched_exists = 1
            matched_peak_value = float(v[p])
            matched_delay_ms = float(d[p] * 1000.0)
            break

    weak_thr, _, _, _ = robust_threshold(v, WEAK_K, WEAK_MIN_REL)
    min_prom = max(WEAK_MIN_PROM_ABS, WEAK_MIN_PROM_REL * vmax)
    weak_peaks, weak_props = find_peaks(v, prominence=min_prom)
    weak_exists = 0
    weak_peak_value = np.nan
    weak_conf = 0.0
    if weak_peaks.size > 0:
        prominences = weak_props.get("prominences", np.zeros_like(weak_peaks, dtype=np.float64))
        for p, prom in sorted(zip(weak_peaks, prominences), key=lambda x: x[0]):
            pv = float(v[int(p)])
            if pv >= weak_thr:
                weak_exists = 1
                weak_peak_value = pv
                conf_h = pv / max(vmax, 1e-12)
                conf_p = float(prom) / max(vmax, 1e-12)
                weak_conf = float(min(1.0, 0.6 * conf_h + 0.4 * conf_p))
                break

    peak_value = float(np.max(v))
    peak_snr = float(peak_value / max(nf_thr, 1e-12))

    return {
        "noise_floor": float(nf_thr),
        "peak_value": peak_value,
        "peak_snr": peak_snr,
        "matched_peak_exists": int(matched_exists),
        "matched_peak_value": matched_peak_value,
        "matched_peak_delay_ms": matched_delay_ms,
        "weak_evidence_exists": int(weak_exists),
        "weak_peak_value": weak_peak_value,
        "weak_confidence": weak_conf,
    }


def evaluate_thresholds(
    df: pd.DataFrame,
    snr_clear_max: float,
    peak_value_clear_max: float,
    weak_conf_max: float,
    noise_level_clear_max: float,
) -> dict[str, float]:
    pred = np.full(len(df), "UNCERTAIN", dtype=object)

    matched = df["matched_peak_exists"] == 1
    pred[matched.to_numpy()] = "OBSTACLE"

    clear_mask = (
        (~matched)
        & (df["peak_snr"] <= snr_clear_max)
        & (df["peak_value"] <= peak_value_clear_max)
        & (df["weak_confidence"] <= weak_conf_max)
        & (df["noise_level"] <= noise_level_clear_max)
    )
    pred[clear_mask.to_numpy()] = "CLEAR"

    true_obs = df["has_obstacle"].astype(int)
    pred_series = pd.Series(pred, index=df.index)

    obstacle_to_obstacle = int(((true_obs == 1) & (pred_series == "OBSTACLE")).sum())
    obstacle_to_uncertain = int(((true_obs == 1) & (pred_series == "UNCERTAIN")).sum())
    obstacle_to_clear = int(((true_obs == 1) & (pred_series == "CLEAR")).sum())
    clear_to_clear = int(((true_obs == 0) & (pred_series == "CLEAR")).sum())
    clear_to_uncertain = int(((true_obs == 0) & (pred_series == "UNCERTAIN")).sum())
    clear_to_obstacle = int(((true_obs == 0) & (pred_series == "OBSTACLE")).sum())

    total_obs = int((true_obs == 1).sum())
    total_clear = int((true_obs == 0).sum())

    safety_miss_rate = obstacle_to_clear / max(total_obs, 1)
    clear_recovery_rate = clear_to_clear / max(total_clear, 1)
    false_obstacle_rate = clear_to_obstacle / max(total_clear, 1)

    return {
        "snr_clear_max": float(snr_clear_max),
        "peak_value_clear_max": float(peak_value_clear_max),
        "weak_conf_max": float(weak_conf_max),
        "noise_level_clear_max": float(noise_level_clear_max),
        "obstacle_to_obstacle": obstacle_to_obstacle,
        "obstacle_to_uncertain": obstacle_to_uncertain,
        "obstacle_to_clear": obstacle_to_clear,
        "clear_to_clear": clear_to_clear,
        "clear_to_uncertain": clear_to_uncertain,
        "clear_to_obstacle": clear_to_obstacle,
        "safety_miss_rate": float(safety_miss_rate),
        "clear_recovery_rate": float(clear_recovery_rate),
        "false_obstacle_rate": float(false_obstacle_rate),
    }


def select_best(df_sweep: pd.DataFrame) -> pd.Series:
    # Priority:
    # 1) obstacle_to_clear must be 0 if possible
    # 2) else safety_miss_rate < 0.1%
    # 3) maximize clear_recovery_rate
    # 4) minimize false_obstacle_rate
    safe_zero = df_sweep[df_sweep["obstacle_to_clear"] == 0].copy()
    if not safe_zero.empty:
        ranked = safe_zero.sort_values(
            by=["clear_recovery_rate", "false_obstacle_rate"],
            ascending=[False, True],
        )
        return ranked.iloc[0]

    safe_small = df_sweep[df_sweep["safety_miss_rate"] < 0.001].copy()
    if not safe_small.empty:
        ranked = safe_small.sort_values(
            by=["clear_recovery_rate", "false_obstacle_rate", "safety_miss_rate"],
            ascending=[False, True, True],
        )
        return ranked.iloc[0]

    ranked = df_sweep.sort_values(
        by=["safety_miss_rate", "clear_recovery_rate", "false_obstacle_rate"],
        ascending=[True, False, True],
    )
    return ranked.iloc[0]


def main() -> None:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels CSV: {LABELS_CSV}")
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {PRED_CSV}")

    labels = pd.read_csv(LABELS_CSV)
    preds = pd.read_csv(PRED_CSV)

    # Merge and normalize key fields.
    df = labels.merge(preds, on=["sample_id", "sector"], how="left", suffixes=("_label", "_pred"))
    df["has_obstacle"] = pd.to_numeric(df["has_obstacle"], errors="coerce").fillna(0).astype(int)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")
    df["reflection_strength"] = pd.to_numeric(df["reflection_strength"], errors="coerce")
    df["surface_absorption"] = pd.to_numeric(df["surface_absorption"], errors="coerce")
    df["has_secondary_reflection"] = pd.to_numeric(df["has_secondary_reflection"], errors="coerce").fillna(0).astype(int)
    df["predicted_distance_m"] = pd.to_numeric(df.get("predicted_distance_m"), errors="coerce")

    # Recompute diagnostics if not already present in predictions.
    needed = ["peak_snr", "noise_floor", "peak_value", "weak_confidence", "matched_peak_exists"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing diagnostic columns in prediction CSV. Recomputing from WAV: {missing}")
        tx = make_transmit_chirp()
        diag_rows: list[dict[str, float | int]] = []
        for i, row in df.iterrows():
            wav_path = DATASET_ROOT / str(row["wav_path"])
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing WAV file: {wav_path}")
            sig = load_wav_float(wav_path)
            diag_rows.append(compute_diagnostics(sig, tx))
            if (i + 1) % 2500 == 0:
                print(f"Computed diagnostics {i + 1}/{len(df)}")
        diag_df = pd.DataFrame(diag_rows)
        for c in diag_df.columns:
            df[c] = diag_df[c].to_numpy()
    else:
        print("Diagnostic columns already present; no recomputation needed.")

    # Distribution analysis for clear vs obstacle sectors.
    obs_df = df[df["has_obstacle"] == 1]
    clr_df = df[df["has_obstacle"] == 0]

    feature_stats = {}
    for col in [
        "peak_snr",
        "noise_floor",
        "peak_value",
        "selected_mode",
        "confidence",
        "predicted_distance_m",
        "reflection_strength",
        "noise_level",
        "surface_absorption",
        "has_secondary_reflection",
    ]:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_stats[col] = {
                "obstacle_mean": float(obs_df[col].mean()),
                "clear_mean": float(clr_df[col].mean()),
                "obstacle_q10": float(obs_df[col].quantile(0.10)),
                "obstacle_q50": float(obs_df[col].quantile(0.50)),
                "obstacle_q90": float(obs_df[col].quantile(0.90)),
                "clear_q10": float(clr_df[col].quantile(0.10)),
                "clear_q50": float(clr_df[col].quantile(0.50)),
                "clear_q90": float(clr_df[col].quantile(0.90)),
            }
        else:
            feature_stats[col] = {
                "obstacle_counts": obs_df[col].value_counts(dropna=False).to_dict(),
                "clear_counts": clr_df[col].value_counts(dropna=False).to_dict(),
            }

    # Build candidate threshold grids from observed distributions.
    snr_grid = sorted(
        set(
            [
                0.6,
                0.8,
                1.0,
                1.2,
                1.5,
                float(df["peak_snr"].quantile(0.25)),
                float(df["peak_snr"].quantile(0.35)),
            ]
        )
    )
    peak_value_grid = sorted(
        set(
            [
                float(df["peak_value"].quantile(0.10)),
                float(df["peak_value"].quantile(0.20)),
                float(df["peak_value"].quantile(0.30)),
                float(df["peak_value"].quantile(0.40)),
                float(df["peak_value"].quantile(0.50)),
            ]
        )
    )
    weak_conf_grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    noise_clear_grid = [0.006, 0.008, 0.010, 0.012, 0.015, 0.020]

    sweep_rows = []
    for snr_thr, pv_thr, conf_thr, nz_thr in itertools.product(
        snr_grid, peak_value_grid, weak_conf_grid, noise_clear_grid
    ):
        sweep_rows.append(evaluate_thresholds(df, snr_thr, pv_thr, conf_thr, nz_thr))

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_SWEEP_CSV, index=False)

    best = select_best(sweep)

    # Top configurations chart.
    # Show top 10 from selection-compatible ordering.
    safe_zero = sweep[sweep["obstacle_to_clear"] == 0]
    if not safe_zero.empty:
        top = safe_zero.sort_values(
            by=["clear_recovery_rate", "false_obstacle_rate"], ascending=[False, True]
        ).head(10)
    else:
        top = sweep.sort_values(
            by=["safety_miss_rate", "clear_recovery_rate", "false_obstacle_rate"],
            ascending=[True, False, True],
        ).head(10)
    top = top.reset_index(drop=True)

    # Plot: clear_recovery vs safety_miss
    plt.figure(figsize=(6.8, 4.8), dpi=140)
    sc = plt.scatter(
        sweep["safety_miss_rate"] * 100.0,
        sweep["clear_recovery_rate"] * 100.0,
        c=sweep["false_obstacle_rate"] * 100.0,
        s=18,
        alpha=0.65,
        cmap="viridis",
    )
    plt.colorbar(sc, label="False obstacle rate (%)")
    plt.xlabel("Safety miss rate (%)")
    plt.ylabel("Clear recovery rate (%)")
    plt.title("Clear Recovery vs Safety Miss")
    plt.tight_layout()
    plt.savefig(OUT_PLOT_RECOVERY)
    plt.close()

    # Plot: false_obstacle vs safety_miss
    plt.figure(figsize=(6.8, 4.8), dpi=140)
    sc = plt.scatter(
        sweep["safety_miss_rate"] * 100.0,
        sweep["false_obstacle_rate"] * 100.0,
        c=sweep["clear_recovery_rate"] * 100.0,
        s=18,
        alpha=0.65,
        cmap="plasma",
    )
    plt.colorbar(sc, label="Clear recovery rate (%)")
    plt.xlabel("Safety miss rate (%)")
    plt.ylabel("False obstacle rate (%)")
    plt.title("False Obstacle vs Safety Miss")
    plt.tight_layout()
    plt.savefig(OUT_PLOT_FALSE_OBS)
    plt.close()

    # Plot: top-threshold comparison
    labels = [f"C{i+1}" for i in range(len(top))]
    x = np.arange(len(top))
    w = 0.25
    plt.figure(figsize=(10.8, 5.0), dpi=140)
    plt.bar(x - w, top["clear_recovery_rate"] * 100.0, width=w, label="Clear recovery %")
    plt.bar(x, top["false_obstacle_rate"] * 100.0, width=w, label="False obstacle %")
    plt.bar(x + w, top["safety_miss_rate"] * 100.0, width=w, label="Safety miss %")
    plt.xticks(x, labels)
    plt.ylabel("Rate (%)")
    plt.title("Top Threshold Configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT_TOP)
    plt.close()

    # Build expected counts from best row.
    best_counts = {
        "obstacle_to_obstacle": int(best["obstacle_to_obstacle"]),
        "obstacle_to_uncertain": int(best["obstacle_to_uncertain"]),
        "obstacle_to_clear": int(best["obstacle_to_clear"]),
        "clear_to_clear": int(best["clear_to_clear"]),
        "clear_to_uncertain": int(best["clear_to_uncertain"]),
        "clear_to_obstacle": int(best["clear_to_obstacle"]),
    }

    result = {
        "selection_priority": [
            "obstacle_to_clear must be 0 if possible",
            "otherwise safety_miss_rate < 0.1%",
            "maximize clear_recovery_rate",
            "minimize false_obstacle_rate",
        ],
        "dataset_counts": {
            "total_rows": int(len(df)),
            "total_obstacle": int((df["has_obstacle"] == 1).sum()),
            "total_clear": int((df["has_obstacle"] == 0).sum()),
        },
        "best_configuration": {
            "snr_clear_max": float(best["snr_clear_max"]),
            "peak_value_clear_max": float(best["peak_value_clear_max"]),
            "weak_conf_max": float(best["weak_conf_max"]),
            "noise_level_clear_max": float(best["noise_level_clear_max"]),
            "safety_miss_rate": float(best["safety_miss_rate"]),
            "clear_recovery_rate": float(best["clear_recovery_rate"]),
            "false_obstacle_rate": float(best["false_obstacle_rate"]),
            "expected_confusion_counts": best_counts,
        },
        "feature_distribution_summary": feature_stats,
        "output_files": {
            "sweep_csv": str(OUT_SWEEP_CSV),
            "calibration_json": str(OUT_JSON),
            "plot_clear_recovery_vs_safety_miss": str(OUT_PLOT_RECOVERY),
            "plot_false_obstacle_vs_safety_miss": str(OUT_PLOT_FALSE_OBS),
            "plot_top_thresholds": str(OUT_PLOT_TOP),
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Best threshold configuration:")
    print(
        f"  snr_clear_max={best['snr_clear_max']:.6f}, "
        f"peak_value_clear_max={best['peak_value_clear_max']:.6f}, "
        f"weak_conf_max={best['weak_conf_max']:.6f}, "
        f"noise_level_clear_max={best['noise_level_clear_max']:.6f}"
    )
    print(
        f"  safety_miss_rate={best['safety_miss_rate']:.6f}, "
        f"clear_recovery_rate={best['clear_recovery_rate']:.6f}, "
        f"false_obstacle_rate={best['false_obstacle_rate']:.6f}"
    )
    print("Expected new confusion counts:")
    for k, v in best_counts.items():
        print(f"  {k}: {v}")
    print(f"\nSaved sweep CSV: {OUT_SWEEP_CSV}")
    print(f"Saved calibration JSON: {OUT_JSON}")
    print(f"Saved plots: {OUT_PLOT_RECOVERY}, {OUT_PLOT_FALSE_OBS}, {OUT_PLOT_TOP}")


if __name__ == "__main__":
    main()
