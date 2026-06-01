"""
Temporal clear-confirmation evaluator for multisector acoustic dataset (v1).

This script tests whether repeated measurements (window=3) can recover CLEAR
decisions while preserving safety.
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
PRED_CSV = DATASET_ROOT / "multisector_three_state_predictions.csv"
CALIB_JSON = DATASET_ROOT / "multisector_clear_threshold_calibration.json"

OUT_JSON = DATASET_ROOT / "multisector_temporal_clear_results.json"
OUT_CSV = DATASET_ROOT / "multisector_temporal_clear_predictions.csv"
OUT_CONFUSION = DATASET_ROOT / "multisector_temporal_three_state_confusion.png"
OUT_ACTION_DIST = DATASET_ROOT / "multisector_temporal_action_distribution.png"

SECTORS = ["left", "front_left", "front", "front_right", "right"]
WINDOW_SIZE = 3

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
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    vmax = float(np.max(vals))
    thr = max(med + k * sigma, min_rel * vmax)
    return thr, med, mad, vmax


def compute_diagnostics(signal: np.ndarray, tx_chirp: np.ndarray) -> dict[str, float | int]:
    d, v = search_region(signal, tx_chirp)
    peaks, _ = find_peaks(v)

    nf_thr, _, _, vmax = robust_threshold(v, NOISE_FLOOR_K, NOISE_MIN_REL)
    matched_exists = 0
    for p in peaks:
        if v[p] >= nf_thr:
            matched_exists = 1
            break

    weak_thr, _, _, _ = robust_threshold(v, WEAK_K, WEAK_MIN_REL)
    min_prom = max(WEAK_MIN_PROM_ABS, WEAK_MIN_PROM_REL * vmax)
    weak_peaks, weak_props = find_peaks(v, prominence=min_prom)
    weak_conf = 0.0
    if weak_peaks.size > 0:
        prominences = weak_props.get("prominences", np.zeros_like(weak_peaks, dtype=np.float64))
        for p, prom in sorted(zip(weak_peaks, prominences), key=lambda x: x[0]):
            pv = float(v[int(p)])
            if pv >= weak_thr:
                conf_h = pv / max(vmax, 1e-12)
                conf_p = float(prom) / max(vmax, 1e-12)
                weak_conf = float(min(1.0, 0.6 * conf_h + 0.4 * conf_p))
                break

    peak_value = float(np.max(v))
    peak_snr = float(peak_value / max(nf_thr, 1e-12))
    return {
        "matched_peak_exists": int(matched_exists),
        "peak_snr": peak_snr,
        "peak_value": peak_value,
        "noise_floor": float(nf_thr),
        "weak_confidence": weak_conf,
    }


def temporal_decision(window: pd.DataFrame, snr_thr: float, pv_thr: float, weak_conf_thr: float, noise_thr: float):
    # Any OBSTACLE in window -> OBSTACLE with median obstacle distance.
    obs_rows = window[(window["predicted_state"] == "OBSTACLE") & window["predicted_distance_m"].notna()]
    if not obs_rows.empty:
        return "OBSTACLE", float(np.median(obs_rows["predicted_distance_m"].astype(float))), "any_obstacle_in_window"

    all_clear = bool((window["predicted_state"] == "CLEAR").all())
    if all_clear:
        return "CLEAR", float("nan"), "all_three_clear"

    # all 3 have no matched peak and low evidence -> CLEAR
    no_peak_all = bool((window["matched_peak_exists"] == 0).all())
    low_evidence_all = bool(
        ((window["peak_snr"] <= snr_thr) & (window["peak_value"] <= pv_thr) & (window["weak_confidence"] <= weak_conf_thr) & (window["noise_level"] <= noise_thr)).all()
    )
    if no_peak_all and low_evidence_all:
        return "CLEAR", float("nan"), "all_no_peak_and_low_evidence"

    return "UNCERTAIN", float("nan"), "mixed_or_nonclear_evidence"


def confusion_counts(df: pd.DataFrame, state_col: str, true_col: str) -> dict[str, int]:
    true_obs = df[true_col].astype(int)
    pred = df[state_col]
    return {
        "obstacle_to_obstacle": int(((true_obs == 1) & (pred == "OBSTACLE")).sum()),
        "obstacle_to_uncertain": int(((true_obs == 1) & (pred == "UNCERTAIN")).sum()),
        "obstacle_to_clear": int(((true_obs == 1) & (pred == "CLEAR")).sum()),
        "clear_to_clear": int(((true_obs == 0) & (pred == "CLEAR")).sum()),
        "clear_to_uncertain": int(((true_obs == 0) & (pred == "UNCERTAIN")).sum()),
        "clear_to_obstacle": int(((true_obs == 0) & (pred == "OBSTACLE")).sum()),
    }


def state_to_action(state: str) -> str:
    if state == "OBSTACLE":
        return "STOP_OR_REVERSE"
    if state == "UNCERTAIN":
        return "SLOW_DOWN_AND_RESAMPLE"
    return "MOVE_FORWARD"


def main() -> None:
    if not LABELS_CSV.exists() or not PRED_CSV.exists():
        raise FileNotFoundError("Missing required input CSV(s).")

    labels = pd.read_csv(LABELS_CSV)
    pred = pd.read_csv(PRED_CSV)

    df = labels.merge(pred, on=["sample_id", "sector"], how="inner")

    df["has_obstacle"] = pd.to_numeric(df["has_obstacle"], errors="coerce").fillna(0).astype(int)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["predicted_distance_m"] = pd.to_numeric(df["predicted_distance_m"], errors="coerce")
    df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")

    # If diagnostics are missing, recompute from WAV.
    need_diag = [c for c in ["peak_snr", "peak_value", "weak_confidence", "matched_peak_exists", "noise_floor"] if c not in df.columns]
    if need_diag:
        print(f"Missing diagnostics {need_diag}, recomputing from WAV...")
        tx = make_transmit_chirp()
        diag_rows = []
        for i, row in df.iterrows():
            wav_path = DATASET_ROOT / str(row["wav_path"])
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing WAV file: {wav_path}")
            signal = load_wav_float(wav_path)
            diag_rows.append(compute_diagnostics(signal, tx))
            if (i + 1) % 2500 == 0:
                print(f"Computed diagnostics {i + 1}/{len(df)}")
        diag_df = pd.DataFrame(diag_rows)
        for c in diag_df.columns:
            df[c] = diag_df[c].to_numpy()

    # Thresholds from calibration if available, else defaults.
    snr_thr = 1.0
    pv_thr = float(df["peak_value"].quantile(0.4))
    weak_conf_thr = 0.4
    noise_thr = 0.015
    if CALIB_JSON.exists():
        with CALIB_JSON.open("r", encoding="utf-8") as f:
            cj = json.load(f)
        if "best_configuration" in cj:
            b = cj["best_configuration"]
            snr_thr = float(b.get("snr_clear_max", snr_thr))
            pv_thr = float(b.get("peak_value_clear_max", pv_thr))
            weak_conf_thr = float(b.get("weak_conf_max", weak_conf_thr))
            noise_thr = float(b.get("noise_level_clear_max", noise_thr))

    # Single-frame confusion for comparison.
    single = confusion_counts(df, "predicted_state", "has_obstacle")
    total_obs_single = int((df["has_obstacle"] == 1).sum())
    total_clear_single = int((df["has_obstacle"] == 0).sum())
    single_safety_miss = single["obstacle_to_clear"] / max(total_obs_single, 1)
    single_clear_recovery = single["clear_to_clear"] / max(total_clear_single, 1)
    single_false_obstacle = single["clear_to_obstacle"] / max(total_clear_single, 1)

    # Deterministic windows of 3 by sector + true label type.
    rows = []
    leftovers = []
    for sector in SECTORS:
        for true_cls in [0, 1]:
            sub = df[(df["sector"] == sector) & (df["has_obstacle"] == true_cls)].copy()
            sub = sub.sort_values(["sample_id"]).reset_index(drop=True)
            num_full = len(sub) // WINDOW_SIZE
            leftover = len(sub) - num_full * WINDOW_SIZE
            leftovers.append({"sector": sector, "true_has_obstacle": true_cls, "leftover_samples": int(leftover)})

            for w in range(num_full):
                part = sub.iloc[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE].copy()
                final_state, final_dist, reason = temporal_decision(part, snr_thr, pv_thr, weak_conf_thr, noise_thr)

                true_distance = float(np.median(part["distance_m"].dropna())) if true_cls == 1 else float("nan")
                if true_cls == 1 and final_state == "OBSTACLE" and not np.isnan(final_dist) and not np.isnan(true_distance):
                    abs_err_cm = abs(final_dist - true_distance) * 100.0
                else:
                    abs_err_cm = float("nan")

                rows.append(
                    {
                        "window_id": f"{sector}_{'obs' if true_cls == 1 else 'clr'}_{w:04d}",
                        "sector": sector,
                        "true_has_obstacle": int(true_cls),
                        "sample_ids": ";".join(part["sample_id"].astype(str).tolist()),
                        "sectors_in_window": ";".join(part["sector"].astype(str).tolist()),
                        "num_obstacle_states": int((part["predicted_state"] == "OBSTACLE").sum()),
                        "num_uncertain_states": int((part["predicted_state"] == "UNCERTAIN").sum()),
                        "num_clear_states": int((part["predicted_state"] == "CLEAR").sum()),
                        "window_final_state": final_state,
                        "window_final_distance_m": final_dist,
                        "decision_reason": reason,
                        "true_distance_m": true_distance,
                        "window_abs_error_cm": abs_err_cm,
                        "recommended_action": state_to_action(final_state),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    temporal = confusion_counts(out, "window_final_state", "true_has_obstacle")
    total_obs = int((out["true_has_obstacle"] == 1).sum())
    total_clear = int((out["true_has_obstacle"] == 0).sum())
    safety_miss_rate = temporal["obstacle_to_clear"] / max(total_obs, 1)
    clear_recovery_rate = temporal["clear_to_clear"] / max(total_clear, 1)
    false_obstacle_rate = temporal["clear_to_obstacle"] / max(total_clear, 1)

    valid_obs = out[(out["true_has_obstacle"] == 1) & (out["window_final_state"] == "OBSTACLE") & out["window_final_distance_m"].notna()]
    if valid_obs.empty:
        mae = float("nan")
        rmse = float("nan")
    else:
        err = np.abs((valid_obs["window_final_distance_m"] - valid_obs["true_distance_m"]) * 100.0).to_numpy(dtype=float)
        mae = float(np.mean(err))
        rmse = float(math.sqrt(np.mean(err**2)))

    action_dist = out["recommended_action"].value_counts().to_dict()

    summary = {
        "window_size": WINDOW_SIZE,
        "thresholds": {
            "snr_clear_max": snr_thr,
            "peak_value_clear_max": pv_thr,
            "weak_confidence_max": weak_conf_thr,
            "noise_level_clear_max": noise_thr,
        },
        "single_frame_counts": single,
        "single_frame_rates": {
            "safety_miss_rate": single_safety_miss,
            "clear_recovery_rate": single_clear_recovery,
            "false_obstacle_rate": single_false_obstacle,
        },
        "temporal_counts": temporal,
        "temporal_rates": {
            "safety_miss_rate": safety_miss_rate,
            "clear_recovery_rate": clear_recovery_rate,
            "false_obstacle_rate": false_obstacle_rate,
        },
        "distance_metrics_obstacle_windows": {
            "mae_cm": mae,
            "rmse_cm": rmse,
            "num_obstacle_windows_with_distance": int(len(valid_obs)),
        },
        "action_distribution": action_dist,
        "leftover_samples_excluded": leftovers,
        "num_temporal_windows": int(len(out)),
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot temporal confusion table.
    conf = np.array(
        [
            [temporal["obstacle_to_obstacle"], temporal["obstacle_to_uncertain"], temporal["obstacle_to_clear"]],
            [temporal["clear_to_obstacle"], temporal["clear_to_uncertain"], temporal["clear_to_clear"]],
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    im = ax.imshow(conf, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks([0, 1, 2], labels=["OBSTACLE", "UNCERTAIN", "CLEAR"])
    ax.set_yticks([0, 1], labels=["True OBSTACLE", "True CLEAR"])
    ax.set_title("Temporal Three-State Confusion (Window=3)")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=11, color="black")
    fig.tight_layout()
    fig.savefig(OUT_CONFUSION)
    plt.close(fig)

    # Plot action distribution.
    ad = pd.Series(action_dist)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    ad.plot(kind="bar", ax=ax, color=["tab:green", "tab:orange", "tab:red"])
    ax.set_ylabel("Count")
    ax.set_title("Action Distribution After Temporal Confirmation")
    fig.tight_layout()
    fig.savefig(OUT_ACTION_DIST)
    plt.close(fig)

    print("Single-frame vs Temporal comparison:")
    print(f"  single obstacle->clear: {single['obstacle_to_clear']}")
    print(f"  single clear->clear: {single['clear_to_clear']}")
    print(f"  single clear->obstacle: {single['clear_to_obstacle']}")
    print(f"  single safety_miss_rate: {single_safety_miss:.6f}")
    print(f"  single clear_recovery_rate: {single_clear_recovery:.6f}")
    print(f"  single false_obstacle_rate: {single_false_obstacle:.6f}")

    print(f"  temporal obstacle->clear: {temporal['obstacle_to_clear']}")
    print(f"  temporal clear->clear: {temporal['clear_to_clear']}")
    print(f"  temporal clear->obstacle: {temporal['clear_to_obstacle']}")
    print(f"  temporal safety_miss_rate: {safety_miss_rate:.6f}")
    print(f"  temporal clear_recovery_rate: {clear_recovery_rate:.6f}")
    print(f"  temporal false_obstacle_rate: {false_obstacle_rate:.6f}")
    print(f"  temporal distance MAE: {mae:.6f} cm")
    print(f"  temporal distance RMSE: {rmse:.6f} cm")
    print(f"  temporal action distribution: {action_dist}")
    print(f"\nSaved results JSON: {OUT_JSON}")
    print(f"Saved temporal predictions CSV: {OUT_CSV}")
    print(f"Saved plots: {OUT_CONFUSION}, {OUT_ACTION_DIST}")


if __name__ == "__main__":
    main()
