"""
Evaluate multi-sector three-state acoustic perception with a learned echo-validity gate.

This script keeps safety-first matched-filter logic, then uses the trained feature-based
echo-validity classifier to improve CLEAR vs UNCERTAIN decisions when no strong
noise-floor peak exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import wavfile
from scipy.signal import chirp, correlate, find_peaks, peak_widths


DATASET_ROOT = Path("datasets/synthetic_echoes_multisector_v1")
LABELS_CSV = DATASET_ROOT / "labels.csv"

ECHO_CKPT = Path("neural_network/checkpoints/echo_validity/echo_validity_classifier_best.pt")
ECHO_NORM = Path("neural_network/checkpoints/echo_validity/feature_normalization.json")

OUT_PRED_CSV = DATASET_ROOT / "multisector_echo_validity_predictions.csv"
OUT_SCENE_CSV = DATASET_ROOT / "multisector_echo_validity_scene_summary.csv"
OUT_JSON = DATASET_ROOT / "multisector_echo_validity_results.json"

OUT_CONFUSION_PNG = DATASET_ROOT / "multisector_echo_validity_sector_confusion_table.png"
OUT_STATE_DIST_PNG = DATASET_ROOT / "multisector_echo_validity_state_distribution_by_sector.png"
OUT_ACTION_DIST_PNG = DATASET_ROOT / "multisector_echo_validity_action_distribution.png"
OUT_PROB_HIST_PNG = DATASET_ROOT / "multisector_echo_validity_probability_histogram.png"

SECTORS = ["left", "front_left", "front", "front_right", "right"]

SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0

DIRECT_IGNORE_S = 0.00055
DIST_MIN_M = 0.05
DIST_MAX_M = 2.80

# Same matched-filter detector family used in previous evaluators.
NOISE_FLOOR_K = 6.0
NOISE_MIN_REL = 0.08

# Safety-first learned gate thresholds.
VALID_ECHO_OBSTACLE_PROB = 0.90
# Keep CLEAR gate strict to preserve safety (avoid obstacle->CLEAR misses).
VALID_ECHO_CLEAR_PROB = 0.001
CLEAR_LOW_SNR_MAX = 1.05
CLEAR_LOW_PROM_MAX = 0.30

OLD_BASELINE_COUNTS = {
    "obstacle_to_clear": 0,
    "clear_to_clear": 0,
    "clear_to_obstacle": 604,
    "safety_miss_rate": 0.0,
}


class EchoValidityMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


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


def robust_threshold(vals: np.ndarray, k: float, min_rel: float) -> tuple[float, float, float, float]:
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    sigma = 1.4826 * mad
    vmax = float(np.max(vals))
    thr = max(median + k * sigma, min_rel * vmax)
    return thr, median, mad, vmax


def extract_features_from_region(delays_s: np.ndarray, vals: np.ndarray) -> dict[str, float]:
    peaks, props = find_peaks(vals, prominence=0.0)
    num_peaks = int(peaks.size)
    nf_thr, _, _, _ = robust_threshold(vals, NOISE_FLOOR_K, NOISE_MIN_REL)

    if num_peaks > 0:
        peak_vals = vals[peaks]
        sidx_local = int(np.argmax(peak_vals))
        sidx = int(peaks[sidx_local])
        strongest_peak_value = float(vals[sidx])
        strongest_peak_delay_ms = float(delays_s[sidx] * 1000.0)
        strongest_prom = float(props["prominences"][sidx_local]) if "prominences" in props else 0.0
        widths = peak_widths(vals, [sidx], rel_height=0.5)[0]
        peak_width_ms = float(widths[0] / SAMPLE_RATE * 1000.0) if len(widths) > 0 else np.nan
    else:
        strongest_peak_value = np.nan
        strongest_peak_delay_ms = np.nan
        strongest_prom = np.nan
        peak_width_ms = np.nan

    first_nf_val = np.nan
    first_nf_delay_ms = np.nan
    if num_peaks > 0:
        for p in peaks:
            if vals[p] >= nf_thr:
                first_nf_val = float(vals[p])
                first_nf_delay_ms = float(delays_s[p] * 1000.0)
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


def delay_to_distance(delay_s: float) -> float:
    return delay_s * SPEED_OF_SOUND / 2.0


def delay_ms_to_distance(delay_ms: float) -> float:
    return (delay_ms / 1000.0) * SPEED_OF_SOUND / 2.0


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
    clear_total = int((true_obs == 0).sum())
    safety_miss_rate = float(obs_to_clr / max(obstacle_total, 1))
    clear_recovery_rate = float(clr_to_clr / max(clear_total, 1))
    false_obstacle_rate = float(clr_to_obs / max(clear_total, 1))

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
        "clear_recovery_rate": clear_recovery_rate,
        "false_obstacle_rate": false_obstacle_rate,
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


def load_echo_validity_model():
    if not ECHO_CKPT.exists():
        raise FileNotFoundError(f"Missing echo-validity model: {ECHO_CKPT}")
    if not ECHO_NORM.exists():
        raise FileNotFoundError(f"Missing echo-validity normalization file: {ECHO_NORM}")

    with ECHO_NORM.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    feature_cols = norm["feature_columns"]
    feat_mean = pd.Series(norm["mean"], dtype=float)
    feat_std = pd.Series(norm["std"], dtype=float).mask(lambda s: s < 1e-6, 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EchoValidityMLP(in_dim=len(feature_cols)).to(device)
    ckpt = torch.load(ECHO_CKPT, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model, device, feature_cols, feat_mean, feat_std


def predict_valid_echo_probability(
    feature_dict: dict[str, float],
    model: nn.Module,
    device: torch.device,
    feature_cols: list[str],
    feat_mean: pd.Series,
    feat_std: pd.Series,
) -> float:
    x_row = pd.DataFrame([{c: feature_dict.get(c, np.nan) for c in feature_cols}])[feature_cols]
    x_row = x_row.apply(pd.to_numeric, errors="coerce").fillna(feat_mean)
    x_row = (x_row - feat_mean) / feat_std
    x_np = np.nan_to_num(x_row.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        x_t = torch.from_numpy(x_np).to(device)
        prob = torch.sigmoid(model(x_t)).cpu().numpy().reshape(-1)[0]
    return float(prob)


def main() -> None:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

    labels = pd.read_csv(LABELS_CSV)
    tx_chirp = make_transmit_chirp()
    model, device, feature_cols, feat_mean, feat_std = load_echo_validity_model()

    rows: list[dict[str, object]] = []

    for i, row in labels.iterrows():
        wav_path = DATASET_ROOT / str(row["wav_path"])
        corr_path = DATASET_ROOT / str(row["correlation_path"])
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file: {wav_path}")
        if not corr_path.exists():
            raise FileNotFoundError(f"Missing correlation feature file: {corr_path}")

        signal = load_wav_float(wav_path)
        delays_s, vals = compute_search_region(signal, tx_chirp)
        feat = extract_features_from_region(delays_s, vals)
        echo_prob = predict_valid_echo_probability(feat, model, device, feature_cols, feat_mean, feat_std)

        mf_peak_exists = pd.notna(feat["first_noise_floor_peak_delay_ms"])
        mf_dist = (
            delay_ms_to_distance(float(feat["first_noise_floor_peak_delay_ms"]))
            if mf_peak_exists
            else np.nan
        )

        peak_snr = float(feat["peak_snr"]) if pd.notna(feat["peak_snr"]) else 0.0
        peak_prom = float(feat["peak_prominence"]) if pd.notna(feat["peak_prominence"]) else 0.0

        if mf_peak_exists:
            pred_state = "OBSTACLE"
            pred_dist = mf_dist
            confidence = "high"
            selected_mode = "matched_filter_obstacle"
        elif echo_prob >= VALID_ECHO_OBSTACLE_PROB:
            if pd.notna(feat["strongest_peak_delay_ms"]):
                pred_dist = delay_ms_to_distance(float(feat["strongest_peak_delay_ms"]))
            elif pd.notna(feat["first_noise_floor_peak_delay_ms"]):
                pred_dist = delay_ms_to_distance(float(feat["first_noise_floor_peak_delay_ms"]))
            else:
                pred_dist = np.nan
            pred_state = "OBSTACLE"
            confidence = "medium"
            selected_mode = "learned_valid_echo_obstacle"
        elif (echo_prob <= VALID_ECHO_CLEAR_PROB) and (peak_snr <= CLEAR_LOW_SNR_MAX) and (peak_prom <= CLEAR_LOW_PROM_MAX):
            pred_state = "CLEAR"
            pred_dist = np.nan
            confidence = "high"
            selected_mode = "learned_clear_gate"
        else:
            pred_state = "UNCERTAIN"
            pred_dist = np.nan
            confidence = "low"
            selected_mode = "uncertain"

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
                "echo_validity_probability": echo_prob,
                "matched_filter_peak_exists": int(bool(mf_peak_exists)),
                "matched_filter_distance_m": mf_dist,
                "predicted_state": pred_state,
                "predicted_distance_m": pred_dist,
                "confidence": confidence,
                "selected_mode": selected_mode,
                "absolute_error_cm": abs_err_cm,
                "strongest_peak_value": feat["strongest_peak_value"],
                "strongest_peak_delay_ms": feat["strongest_peak_delay_ms"],
                "first_noise_floor_peak_value": feat["first_noise_floor_peak_value"],
                "first_noise_floor_peak_delay_ms": feat["first_noise_floor_peak_delay_ms"],
                "peak_snr": feat["peak_snr"],
                "peak_prominence": feat["peak_prominence"],
                "peak_width": feat["peak_width"],
                "noise_floor": feat["noise_floor"],
                "num_peaks": feat["num_peaks"],
            }
        )

        if (i + 1) % 2500 == 0:
            print(f"Processed {i + 1}/{len(labels)} sector rows...")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_PRED_CSV, index=False)

    overall = sector_metrics(pred_df)
    by_sector = {sec: sector_metrics(pred_df[pred_df["sector"] == sec]) for sec in SECTORS}

    # Scene-level summary.
    scene_state = pred_df.pivot(index="sample_id", columns="sector", values="predicted_state")
    scene_dist = pred_df.pivot(index="sample_id", columns="sector", values="predicted_distance_m")
    scene_state.columns = [f"{c}_state" for c in scene_state.columns]
    scene_dist.columns = [f"{c}_distance_m" for c in scene_dist.columns]
    scene = scene_state.join(scene_dist).reset_index()

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

    results = {
        "echo_validity_thresholds": {
            "obstacle_probability_min": VALID_ECHO_OBSTACLE_PROB,
            "clear_probability_max": VALID_ECHO_CLEAR_PROB,
            "clear_peak_snr_max": CLEAR_LOW_SNR_MAX,
            "clear_peak_prominence_max": CLEAR_LOW_PROM_MAX,
        },
        "overall_sector_metrics": overall,
        "per_sector_metrics": by_sector,
        "action_distribution": action_counts,
        "num_sector_rows": int(len(pred_df)),
        "num_scenes": int(scene.shape[0]),
        "old_baseline_reference": OLD_BASELINE_COUNTS,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot 1: confusion table.
    confusion = np.array(
        [
            [overall["obstacle_to_obstacle"], overall["obstacle_to_uncertain"], overall["obstacle_to_clear"]],
            [overall["clear_to_obstacle"], overall["clear_to_uncertain"], overall["clear_to_clear"]],
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks([0, 1, 2], labels=["OBSTACLE", "UNCERTAIN", "CLEAR"])
    ax.set_yticks([0, 1], labels=["True OBSTACLE", "True CLEAR"])
    ax.set_title("Sector Confusion Table (Echo-Validity Gated)")
    for r in range(confusion.shape[0]):
        for c in range(confusion.shape[1]):
            ax.text(c, r, str(confusion[r, c]), ha="center", va="center", fontsize=11, color="black")
    fig.tight_layout()
    fig.savefig(OUT_CONFUSION_PNG)
    plt.close(fig)

    # Plot 2: predicted state distribution by sector.
    state_order = ["OBSTACLE", "UNCERTAIN", "CLEAR"]
    counts = (
        pred_df.groupby(["sector", "predicted_state"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SECTORS, columns=state_order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=140)
    bottom = np.zeros(len(SECTORS), dtype=np.int64)
    colors = {"OBSTACLE": "tab:red", "UNCERTAIN": "tab:orange", "CLEAR": "tab:green"}
    for st in state_order:
        vals = counts[st].to_numpy()
        ax.bar(SECTORS, vals, bottom=bottom, label=st, color=colors[st])
        bottom += vals
    ax.set_ylabel("Count")
    ax.set_title("Predicted State Distribution by Sector (Echo-Validity Gated)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_STATE_DIST_PNG)
    plt.close(fig)

    # Plot 3: action distribution.
    action_series = scene["recommended_action"].value_counts()
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
    ax.bar(action_series.index.tolist(), action_series.values.tolist(), color="tab:blue")
    ax.set_ylabel("Scene count")
    ax.set_title("Recommended Action Distribution (Echo-Validity Gated)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT_ACTION_DIST_PNG)
    plt.close(fig)

    # Plot 4: echo-validity probability histogram by true class.
    probs_obs = pred_df.loc[pred_df["true_has_obstacle"] == 1, "echo_validity_probability"].to_numpy(dtype=float)
    probs_clr = pred_df.loc[pred_df["true_has_obstacle"] == 0, "echo_validity_probability"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=140)
    ax.hist(probs_clr, bins=50, alpha=0.7, label="True clear")
    ax.hist(probs_obs, bins=50, alpha=0.7, label="True obstacle")
    ax.axvline(VALID_ECHO_CLEAR_PROB, color="tab:green", linestyle="--", linewidth=1.5, label="CLEAR gate prob<=0.10")
    ax.axvline(VALID_ECHO_OBSTACLE_PROB, color="tab:red", linestyle="--", linewidth=1.5, label="OBSTACLE gate prob>=0.90")
    ax.set_xlabel("Echo-validity probability")
    ax.set_ylabel("Count")
    ax.set_title("Echo-Validity Probability: True Obstacle vs True Clear")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PROB_HIST_PNG)
    plt.close(fig)

    print("\nOld multi-sector baseline (reference):")
    for k, v in OLD_BASELINE_COUNTS.items():
        print(f"  {k}: {v}")

    print("\nNew echo-validity-gated metrics (overall):")
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

    print(f"\nSaved per-sector predictions: {OUT_PRED_CSV}")
    print(f"Saved scene summary: {OUT_SCENE_CSV}")
    print(f"Saved results JSON: {OUT_JSON}")
    print(f"Saved plots: {OUT_CONFUSION_PNG}, {OUT_STATE_DIST_PNG}, {OUT_ACTION_DIST_PNG}, {OUT_PROB_HIST_PNG}")


if __name__ == "__main__":
    main()
