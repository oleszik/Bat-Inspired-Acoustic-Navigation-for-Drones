"""
Evaluate rule-based multi-sector three-state acoustic perception (v1).

States per sector:
- OBSTACLE
- UNCERTAIN
- CLEAR

No neural-network training is performed here.
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


DATASET_ROOT = Path("datasets/synthetic_echoes_multisector_v1")
LABELS_CSV = DATASET_ROOT / "labels.csv"

OUT_PRED_CSV = DATASET_ROOT / "multisector_three_state_predictions.csv"
OUT_SCENE_CSV = DATASET_ROOT / "multisector_scene_summary.csv"
OUT_JSON = DATASET_ROOT / "multisector_three_state_results.json"

OUT_CONFUSION_PNG = DATASET_ROOT / "multisector_sector_confusion_table.png"
OUT_ERROR_HIST_PNG = DATASET_ROOT / "multisector_distance_error_histogram.png"
OUT_STATE_DIST_PNG = DATASET_ROOT / "multisector_state_distribution_by_sector.png"

SECTORS = ["left", "front_left", "front", "front_right", "right"]

SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0

DIRECT_IGNORE_S = 0.00055
DIST_MIN_M = 0.05
DIST_MAX_M = 2.80

# Main noise-floor detector.
NOISE_FLOOR_K = 6.0
NOISE_MIN_REL = 0.08

# Weak/uncertain evidence detector (does NOT output obstacle distance).
WEAK_K = 3.0
WEAK_MIN_REL = 0.03
WEAK_MIN_PROM_REL = 0.015
WEAK_MIN_PROM_ABS = 1e-6

# Transparent uncertainty thresholds from metadata.
NOISE_HIGH_THR = 0.02
REFLECTION_LOW_THR = 0.03
PEAK_SNR_LOW_THR = 1.5


def make_transmit_chirp() -> np.ndarray:
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


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


def robust_threshold(vals: np.ndarray, k: float, min_rel: float) -> tuple[float, float, float, float]:
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    robust_sigma = 1.4826 * mad
    vmax = float(np.max(vals))
    thr = max(median + k * robust_sigma, min_rel * vmax)
    return thr, median, mad, vmax


def detect_noise_floor_peak(delays_s: np.ndarray, vals: np.ndarray) -> tuple[float | None, float | None, dict[str, float]]:
    peaks, _ = find_peaks(vals)
    thr, median, mad, vmax = robust_threshold(vals, NOISE_FLOOR_K, NOISE_MIN_REL)
    for p in peaks:
        if vals[p] >= thr:
            return float(delays_s[p]), float(vals[p]), {"threshold": thr, "median": median, "mad": mad, "max": vmax}
    return None, None, {"threshold": thr, "median": median, "mad": mad, "max": vmax}


def detect_weak_peak(delays_s: np.ndarray, vals: np.ndarray) -> tuple[bool, float]:
    """
    Weak evidence detector for UNCERTAIN state only.
    """
    thr, _, _, vmax = robust_threshold(vals, WEAK_K, WEAK_MIN_REL)
    min_prom = max(WEAK_MIN_PROM_ABS, WEAK_MIN_PROM_REL * vmax)
    peaks, props = find_peaks(vals, prominence=min_prom)
    if peaks.size == 0:
        return False, 0.0
    prominences = props.get("prominences", np.zeros_like(peaks, dtype=np.float64))
    for p, prom in sorted(zip(peaks, prominences), key=lambda x: x[0]):
        pv = float(vals[int(p)])
        if pv >= thr:
            conf_h = pv / max(vmax, 1e-12)
            conf_p = float(prom) / max(vmax, 1e-12)
            conf = float(min(1.0, 0.6 * conf_h + 0.4 * conf_p))
            return True, conf
    return False, 0.0


def sector_metrics(df: pd.DataFrame) -> dict[str, float]:
    true_obs = df["true_has_obstacle"].astype(int)
    pred = df["predicted_state"]

    obs_to_obs = int(((true_obs == 1) & (pred == "OBSTACLE")).sum())
    obs_to_unc = int(((true_obs == 1) & (pred == "UNCERTAIN")).sum())
    obs_to_clr = int(((true_obs == 1) & (pred == "CLEAR")).sum())
    clr_to_clr = int(((true_obs == 0) & (pred == "CLEAR")).sum())
    clr_to_unc = int(((true_obs == 0) & (pred == "UNCERTAIN")).sum())
    clr_to_obs = int(((true_obs == 0) & (pred == "OBSTACLE")).sum())

    obstacle_total = int((true_obs == 1).sum())
    safety_miss_rate = float(obs_to_clr / max(obstacle_total, 1))

    valid = df[(df["true_has_obstacle"] == 1) & (df["predicted_state"] == "OBSTACLE") & df["predicted_distance_m"].notna()]
    if valid.empty:
        mae = float("nan")
        rmse = float("nan")
    else:
        err = np.abs((valid["predicted_distance_m"] - valid["true_distance_m"]) * 100.0).to_numpy(dtype=float)
        mae = float(np.mean(err))
        rmse = float(math.sqrt(np.mean(err**2)))

    return {
        "obstacle_to_obstacle": obs_to_obs,
        "obstacle_to_uncertain": obs_to_unc,
        "obstacle_to_clear": obs_to_clr,
        "clear_to_clear": clr_to_clr,
        "clear_to_uncertain": clr_to_unc,
        "clear_to_obstacle": clr_to_obs,
        "safety_miss_rate": safety_miss_rate,
        "distance_mae_cm_for_obstacle_predictions": mae,
        "distance_rmse_cm_for_obstacle_predictions": rmse,
    }


def recommended_action(row: pd.Series) -> str:
    l = row["left_state"]
    fl = row["front_left_state"]
    f = row["front_state"]
    fr = row["front_right_state"]
    r = row["right_state"]

    if (f == "UNCERTAIN") or (fl == "UNCERTAIN") or (fr == "UNCERTAIN"):
        return "SLOW_DOWN_AND_RESAMPLE"
    elif f == "OBSTACLE" and l == "CLEAR":
        return "TURN_LEFT"
    elif f == "OBSTACLE" and r == "CLEAR":
        return "TURN_RIGHT"
    elif f == "OBSTACLE" and (l != "CLEAR") and (r != "CLEAR"):
        return "STOP_OR_REVERSE"
    elif fl == "OBSTACLE" and fr == "CLEAR":
        return "TURN_RIGHT"
    elif fr == "OBSTACLE" and fl == "CLEAR":
        return "TURN_LEFT"
    elif (f == "CLEAR") and (fl == "CLEAR") and (fr == "CLEAR"):
        return "MOVE_FORWARD"
    else:
        return "SLOW_DOWN_AND_RESAMPLE"


def main() -> None:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

    labels = pd.read_csv(LABELS_CSV)
    tx_chirp = make_transmit_chirp()

    rows: list[dict[str, object]] = []

    for i, row in labels.iterrows():
        wav_path = DATASET_ROOT / str(row["wav_path"])
        corr_path = DATASET_ROOT / str(row["correlation_path"])
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")
        if not corr_path.exists():
            # correlation features are an input artifact; we only require presence here.
            raise FileNotFoundError(f"Missing correlation feature file: {corr_path}")

        signal = load_wav_float(wav_path)
        delays_s, vals = compute_search_region(signal, tx_chirp)

        peak_delay_s, peak_val, dbg = detect_noise_floor_peak(delays_s, vals)
        weak_exists, weak_conf = detect_weak_peak(delays_s, vals)
        peak_snr = float(dbg["max"] / max(dbg["threshold"], 1e-12))

        noise_level = float(pd.to_numeric(row["noise_level"], errors="coerce"))
        reflection_strength = float(pd.to_numeric(row["reflection_strength"], errors="coerce"))

        low_quality = (
            (noise_level >= NOISE_HIGH_THR)
            or (reflection_strength <= REFLECTION_LOW_THR)
            or (peak_snr <= PEAK_SNR_LOW_THR)
            or weak_exists
        )

        if peak_delay_s is not None:
            pred_state = "OBSTACLE"
            pred_dist = delay_to_distance(peak_delay_s)
            confidence = "high"
            selected_mode = "matched_filter_noise_floor"
        elif low_quality:
            pred_state = "UNCERTAIN"
            pred_dist = np.nan
            confidence = "low"
            selected_mode = "uncertain_low_quality_or_weak_evidence"
        else:
            pred_state = "CLEAR"
            pred_dist = np.nan
            confidence = "high"
            selected_mode = "clear_no_reliable_peak"

        true_has = int(row["has_obstacle"])
        true_dist = pd.to_numeric(row["distance_m"], errors="coerce")
        if true_has == 1 and pred_state == "OBSTACLE" and pd.notna(true_dist) and pd.notna(pred_dist):
            abs_err_cm = abs(float(pred_dist) - float(true_dist)) * 100.0
        else:
            abs_err_cm = np.nan

        rows.append(
            {
                "sample_id": row["sample_id"],
                "sector": row["sector"],
                "true_has_obstacle": true_has,
                "true_distance_m": true_dist,
                "predicted_state": pred_state,
                "predicted_distance_m": pred_dist,
                "confidence": confidence,
                "selected_mode": selected_mode,
                "absolute_error_cm": abs_err_cm,
            }
        )

        if (i + 1) % 2500 == 0:
            print(f"Processed {i + 1}/{len(labels)} sector rows...")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_PRED_CSV, index=False)

    # Overall and per-sector metrics.
    overall = sector_metrics(pred_df)
    by_sector = {}
    for sec in SECTORS:
        by_sector[sec] = sector_metrics(pred_df[pred_df["sector"] == sec])

    # Scene-level summary.
    scene_state = pred_df.pivot(index="sample_id", columns="sector", values="predicted_state")
    scene_dist = pred_df.pivot(index="sample_id", columns="sector", values="predicted_distance_m")
    scene_state.columns = [f"{c}_state" for c in scene_state.columns]
    scene_dist.columns = [f"{c}_distance_m" for c in scene_dist.columns]
    scene = scene_state.join(scene_dist).reset_index()

    # Ensure columns exist even if one sector missing unexpectedly.
    for c in [f"{s}_state" for s in SECTORS]:
        if c not in scene.columns:
            scene[c] = "UNCERTAIN"
    for c in [f"{s}_distance_m" for s in SECTORS]:
        if c not in scene.columns:
            scene[c] = np.nan

    scene["any_front_obstacle"] = (
        (scene["front_left_state"] == "OBSTACLE")
        | (scene["front_state"] == "OBSTACLE")
        | (scene["front_right_state"] == "OBSTACLE")
    ).astype(int)
    scene["any_uncertain"] = (
        (scene["left_state"] == "UNCERTAIN")
        | (scene["front_left_state"] == "UNCERTAIN")
        | (scene["front_state"] == "UNCERTAIN")
        | (scene["front_right_state"] == "UNCERTAIN")
        | (scene["right_state"] == "UNCERTAIN")
    ).astype(int)

    scene["recommended_action"] = scene.apply(recommended_action, axis=1)
    scene.to_csv(OUT_SCENE_CSV, index=False)

    action_counts = scene["recommended_action"].value_counts().to_dict()

    # Save JSON summary.
    results = {
        "overall_sector_metrics": overall,
        "per_sector_metrics": by_sector,
        "action_distribution": action_counts,
        "num_sector_rows": int(len(pred_df)),
        "num_scenes": int(scene.shape[0]),
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot 1: sector confusion table (overall 2x3).
    confusion = np.array(
        [
            [
                overall["obstacle_to_obstacle"],
                overall["obstacle_to_uncertain"],
                overall["obstacle_to_clear"],
            ],
            [
                overall["clear_to_obstacle"],
                overall["clear_to_uncertain"],
                overall["clear_to_clear"],
            ],
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks([0, 1, 2], labels=["OBSTACLE", "UNCERTAIN", "CLEAR"])
    ax.set_yticks([0, 1], labels=["True OBSTACLE", "True CLEAR"])
    ax.set_title("Sector Confusion Table (Rule-Based Three-State)")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=11, color="black")
    fig.tight_layout()
    fig.savefig(OUT_CONFUSION_PNG)
    plt.close(fig)

    # Plot 2: distance error histogram.
    err = pred_df["absolute_error_cm"].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.hist(err, bins=50, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Absolute distance error (cm)")
    ax.set_ylabel("Count")
    ax.set_title("Distance Error Histogram (Predicted OBSTACLE on True Obstacles)")
    fig.tight_layout()
    fig.savefig(OUT_ERROR_HIST_PNG)
    plt.close(fig)

    # Plot 3: predicted state distribution by sector (stacked bar).
    state_order = ["OBSTACLE", "UNCERTAIN", "CLEAR"]
    counts = (
        pred_df.groupby(["sector", "predicted_state"]).size().unstack(fill_value=0).reindex(index=SECTORS, columns=state_order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    bottom = np.zeros(len(SECTORS), dtype=np.int64)
    colors = {"OBSTACLE": "tab:red", "UNCERTAIN": "tab:orange", "CLEAR": "tab:green"}
    for st in state_order:
        vals = counts[st].to_numpy()
        ax.bar(SECTORS, vals, bottom=bottom, label=st, color=colors[st])
        bottom += vals
    ax.set_ylabel("Count")
    ax.set_title("Predicted State Distribution by Sector")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_STATE_DIST_PNG)
    plt.close(fig)

    # Print requested summaries.
    print("\nOverall sector-level metrics:")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\nPer-sector metrics:")
    for sec in SECTORS:
        print(f"  [{sec}]")
        for k, v in by_sector[sec].items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    print("\nAction distribution:")
    for action, cnt in action_counts.items():
        print(f"  {action}: {cnt}")

    print(f"\nSafety miss rate (overall): {overall['safety_miss_rate']:.6f}")
    print(f"Distance MAE (cm, overall): {overall['distance_mae_cm_for_obstacle_predictions']:.6f}")
    print(f"Distance RMSE (cm, overall): {overall['distance_rmse_cm_for_obstacle_predictions']:.6f}")

    print(f"\nSaved per-sector predictions: {OUT_PRED_CSV}")
    print(f"Saved scene summary: {OUT_SCENE_CSV}")
    print(f"Saved results JSON: {OUT_JSON}")
    print(
        f"Saved plots: {OUT_CONFUSION_PNG}, {OUT_ERROR_HIST_PNG}, {OUT_STATE_DIST_PNG}"
    )


if __name__ == "__main__":
    main()
