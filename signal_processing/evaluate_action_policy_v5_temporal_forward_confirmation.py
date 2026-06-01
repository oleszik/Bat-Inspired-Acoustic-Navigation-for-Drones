"""
Evaluate action policy v5 with temporal forward confirmation.

Policy decisions only use runtime-available prediction features.
Ground truth is used only for evaluation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ROOT = Path("datasets/synthetic_echoes_multisector_v1")
SCENE_CSV = DATASET_ROOT / "multisector_echo_validity_scene_summary.csv"
PRED_CSV = DATASET_ROOT / "multisector_echo_validity_predictions.csv"
V4_JSON = DATASET_ROOT / "action_policy_v4_analysis.json"
V4_SCENE_CSV = DATASET_ROOT / "action_policy_v4_scene_summary.csv"

OUT_JSON = DATASET_ROOT / "action_policy_v5_temporal_analysis.json"
OUT_SCENE_CSV = DATASET_ROOT / "action_policy_v5_temporal_scene_summary.csv"
OUT_PNG = DATASET_ROOT / "action_policy_v5_temporal_distribution.png"

SECTORS = ["left", "front_left", "front", "front_right", "right"]
FRONT_GROUP = ["front_left", "front", "front_right"]

STATE_CLEAR = "CLEAR"
STATE_OBS = "OBSTACLE"
STATE_UNC = "UNCERTAIN"

ACTION_FAST = "MOVE_FORWARD_FAST"
ACTION_SLOW = "MOVE_FORWARD_SLOW"
ACTION_PROBE = "PROBE_FORWARD"
ACTION_LEFT = "TURN_LEFT"
ACTION_RIGHT = "TURN_RIGHT"
ACTION_RESAMPLE = "SLOW_DOWN_AND_RESAMPLE"
ACTION_STOP = "STOP_OR_REVERSE"

ACTION_ORDER_V5 = [ACTION_FAST, ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
ACTION_ORDER_V4 = [ACTION_FAST, ACTION_SLOW, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]


def action_distribution(series: pd.Series) -> tuple[dict[str, int], dict[str, float]]:
    total = int(series.shape[0])
    counts = {str(k): int(v) for k, v in series.value_counts().to_dict().items()}
    pct = {k: (100.0 * v / total if total > 0 else 0.0) for k, v in counts.items()}
    return counts, pct


def _to_bool_peak_exists(x: object) -> bool:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return False
    return float(v) > 0.5


def _to_float(x: object) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else float("nan")


def is_strong_clear(sec: dict[str, object]) -> bool:
    return (
        str(sec["predicted_state"]).upper() == STATE_CLEAR
        and (not _to_bool_peak_exists(sec["matched_filter_peak_exists"]))
        and _to_float(sec["echo_validity_probability"]) <= 0.05
        and _to_float(sec["peak_snr"]) <= 1.0
        and _to_float(sec["peak_prominence"]) <= 0.10
    )


def is_soft_clear(sec: dict[str, object]) -> bool:
    return (
        str(sec["predicted_state"]).upper() == STATE_CLEAR
        and (not _to_bool_peak_exists(sec["matched_filter_peak_exists"]))
        and _to_float(sec["echo_validity_probability"]) <= 0.10
        and _to_float(sec["peak_snr"]) <= 1.5
        and _to_float(sec["peak_prominence"]) <= 0.20
    )


def sector_clearance_distance(sec: dict[str, object]) -> float:
    d1 = _to_float(sec["predicted_distance_m"])
    if np.isfinite(d1):
        return d1
    d2 = _to_float(sec["matched_filter_distance_m"])
    if np.isfinite(d2):
        return d2
    return float("nan")


def choose_turn_side_when_both_clear(left_sec: dict[str, object], right_sec: dict[str, object]) -> str:
    l = sector_clearance_distance(left_sec)
    r = sector_clearance_distance(right_sec)
    if np.isfinite(l) and np.isfinite(r):
        return ACTION_LEFT if l >= r else ACTION_RIGHT
    if np.isfinite(l):
        return ACTION_LEFT
    if np.isfinite(r):
        return ACTION_RIGHT
    return ACTION_LEFT


def fallback_turn_resample_stop(sec: dict[str, dict[str, object]]) -> str:
    left = str(sec["left"]["predicted_state"]).upper()
    fl = str(sec["front_left"]["predicted_state"]).upper()
    f = str(sec["front"]["predicted_state"]).upper()
    fr = str(sec["front_right"]["predicted_state"]).upper()
    right = str(sec["right"]["predicted_state"]).upper()
    any_front_obstacle = (fl == STATE_OBS) or (f == STATE_OBS) or (fr == STATE_OBS)

    # Conservative collision fallback.
    if (left == STATE_OBS) and (right == STATE_OBS) and any_front_obstacle:
        return ACTION_STOP

    if f == STATE_OBS:
        if (left == STATE_CLEAR) and (right != STATE_CLEAR):
            return ACTION_LEFT
        if (right == STATE_CLEAR) and (left != STATE_CLEAR):
            return ACTION_RIGHT
        if (left == STATE_CLEAR) and (right == STATE_CLEAR):
            return choose_turn_side_when_both_clear(sec["left"], sec["right"])
        return ACTION_STOP

    if f == STATE_CLEAR:
        if (fl == STATE_OBS) and (fr == STATE_CLEAR):
            return ACTION_RIGHT
        if (fr == STATE_OBS) and (fl == STATE_CLEAR):
            return ACTION_LEFT
        return ACTION_RESAMPLE

    # front uncertain or unknown: no forward.
    if (left == STATE_CLEAR) and (right != STATE_CLEAR):
        return ACTION_LEFT
    if (right == STATE_CLEAR) and (left != STATE_CLEAR):
        return ACTION_RIGHT
    return ACTION_RESAMPLE


def build_scene_runtime_table(scene_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    required_pred = {
        "sample_id",
        "sector",
        "predicted_state",
        "predicted_distance_m",
        "echo_validity_probability",
        "matched_filter_peak_exists",
        "matched_filter_distance_m",
        "confidence",
        "peak_snr",
        "peak_prominence",
        "peak_width",
        "noise_floor",
        "strongest_peak_value",
        "first_noise_floor_peak_value",
        "selected_mode",
        "true_has_obstacle",
    }
    if "sample_id" not in scene_df.columns:
        raise ValueError("Scene CSV missing sample_id.")
    if not required_pred.issubset(pred_df.columns):
        raise ValueError(f"Prediction CSV missing required columns: {required_pred - set(pred_df.columns)}")

    pred = pred_df.copy()
    pred["sample_id"] = pred["sample_id"].astype(str)
    pred["sector"] = pred["sector"].astype(str)
    pred = pred.drop_duplicates(["sample_id", "sector"], keep="first")

    feature_cols = [
        "predicted_state",
        "predicted_distance_m",
        "echo_validity_probability",
        "matched_filter_peak_exists",
        "matched_filter_distance_m",
        "confidence",
        "peak_snr",
        "peak_prominence",
        "peak_width",
        "noise_floor",
        "strongest_peak_value",
        "first_noise_floor_peak_value",
        "selected_mode",
        "true_has_obstacle",
    ]

    merged = scene_df.copy()
    merged["sample_id"] = merged["sample_id"].astype(str)
    for sec in SECTORS:
        sdf = pred[pred["sector"] == sec][["sample_id"] + feature_cols].copy()
        sdf = sdf.rename(columns={c: f"{sec}_{c}" for c in feature_cols})
        merged = merged.merge(sdf, on="sample_id", how="left")
    return merged


def add_true_conditions(scene: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    truth = pred.copy()
    truth["sample_id"] = truth["sample_id"].astype(str)
    truth = truth.pivot(index="sample_id", columns="sector", values="true_has_obstacle")
    for sec in SECTORS:
        if sec not in truth.columns:
            truth[sec] = 0
    truth = truth.fillna(0).astype(int)
    truth["true_front_obstacle"] = truth["front"] == 1
    truth["true_any_front_group_obstacle"] = truth[FRONT_GROUP].max(axis=1) == 1
    truth["true_all_front_clear"] = truth[FRONT_GROUP].sum(axis=1) == 0
    return scene.merge(truth.reset_index(), on="sample_id", how="left")


def main() -> None:
    for p in [SCENE_CSV, PRED_CSV, V4_JSON, V4_SCENE_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    scene_in = pd.read_csv(SCENE_CSV)
    pred = pd.read_csv(PRED_CSV)
    v4 = json.loads(V4_JSON.read_text(encoding="utf-8"))
    _ = pd.read_csv(V4_SCENE_CSV)  # presence check only, no mutation.

    scene = build_scene_runtime_table(scene_in, pred)

    # Build deterministic temporal windows of 3 by front-group predicted-state pattern.
    # Method:
    # 1) Key = (front_left_state, front_state, front_right_state)
    # 2) sort by sample_id inside each key
    # 3) chunk sequentially into fixed groups of 3
    scene["front_pattern_key"] = (
        scene["front_left_predicted_state"].astype(str)
        + "|"
        + scene["front_predicted_state"].astype(str)
        + "|"
        + scene["front_right_predicted_state"].astype(str)
    )
    scene = scene.sort_values(["front_pattern_key", "sample_id"]).reset_index(drop=True)
    scene["pattern_row_idx"] = scene.groupby("front_pattern_key").cumcount()
    scene["window_idx_in_pattern"] = (scene["pattern_row_idx"] // 3).astype(int)
    scene["temporal_window_id"] = scene["front_pattern_key"] + "::" + scene["window_idx_in_pattern"].astype(str)
    scene["window_size"] = scene.groupby("temporal_window_id")["sample_id"].transform("size")

    # Precompute per-row sector runtime dicts.
    row_sec = {}
    for idx, row in scene.iterrows():
        sec = {}
        for s in SECTORS:
            sec[s] = {
                "predicted_state": row.get(f"{s}_predicted_state", np.nan),
                "predicted_distance_m": row.get(f"{s}_predicted_distance_m", np.nan),
                "echo_validity_probability": row.get(f"{s}_echo_validity_probability", np.nan),
                "matched_filter_peak_exists": row.get(f"{s}_matched_filter_peak_exists", np.nan),
                "matched_filter_distance_m": row.get(f"{s}_matched_filter_distance_m", np.nan),
                "confidence": row.get(f"{s}_confidence", np.nan),
                "peak_snr": row.get(f"{s}_peak_snr", np.nan),
                "peak_prominence": row.get(f"{s}_peak_prominence", np.nan),
                "peak_width": row.get(f"{s}_peak_width", np.nan),
                "noise_floor": row.get(f"{s}_noise_floor", np.nan),
                "strongest_peak_value": row.get(f"{s}_strongest_peak_value", np.nan),
                "first_noise_floor_peak_value": row.get(f"{s}_first_noise_floor_peak_value", np.nan),
                "selected_mode": row.get(f"{s}_selected_mode", np.nan),
            }
        row_sec[idx] = sec

    # Evaluate window-level temporal evidence.
    window_decision: dict[str, str] = {}
    window_debug: dict[str, dict[str, object]] = {}
    for wid, g in scene.groupby("temporal_window_id", sort=False):
        idxs = g.index.tolist()
        # Use only full windows of 3 for temporal confirmation; partial windows fallback later.
        if len(idxs) < 3:
            window_decision[wid] = "FALLBACK_V4_TURN_RESAMPLE_STOP"
            window_debug[wid] = {
                "window_size": len(idxs),
                "strong_clear_count_front": 0,
                "soft_clear_count_front": 0,
                "any_front_group_obstacle_prediction": None,
                "any_front_group_uncertain": None,
                "any_front_group_matched_peak": None,
            }
            continue

        any_front_group_obstacle_prediction = False
        any_front_group_uncertain = False
        any_front_group_matched_peak = False
        strong_all_three_count = 0
        slow_rule_count = 0
        strong_front_count = 0

        for idx in idxs:
            sec = row_sec[idx]
            fl, f, fr = sec["front_left"], sec["front"], sec["front_right"]
            fl_state = str(fl["predicted_state"]).upper()
            f_state = str(f["predicted_state"]).upper()
            fr_state = str(fr["predicted_state"]).upper()

            any_front_group_obstacle_prediction |= (fl_state == STATE_OBS) or (f_state == STATE_OBS) or (fr_state == STATE_OBS)
            any_front_group_uncertain |= (fl_state == STATE_UNC) or (f_state == STATE_UNC) or (fr_state == STATE_UNC)
            any_front_group_matched_peak |= (
                _to_bool_peak_exists(fl["matched_filter_peak_exists"])
                or _to_bool_peak_exists(f["matched_filter_peak_exists"])
                or _to_bool_peak_exists(fr["matched_filter_peak_exists"])
            )

            strong_all_three = is_strong_clear(fl) and is_strong_clear(f) and is_strong_clear(fr)
            slow_rule_ok = is_strong_clear(f) and is_soft_clear(fl) and is_soft_clear(fr) and (not any_front_group_obstacle_prediction)
            strong_front = is_strong_clear(f)

            strong_all_three_count += int(strong_all_three)
            slow_rule_count += int(slow_rule_ok)
            strong_front_count += int(strong_front)

        if any_front_group_obstacle_prediction:
            win_action = "NO_FORWARD_DUE_TO_FRONT_OBSTACLE_PRED"
        elif strong_all_three_count == 3:
            win_action = ACTION_FAST
        elif slow_rule_count >= 2:
            win_action = ACTION_SLOW
        elif strong_front_count >= 1:
            win_action = ACTION_PROBE
        else:
            win_action = "FALLBACK_V4_TURN_RESAMPLE_STOP"

        window_decision[wid] = win_action
        window_debug[wid] = {
            "window_size": len(idxs),
            "strong_clear_count_front": int(strong_all_three_count),
            "soft_clear_count_front": int(slow_rule_count),
            "any_front_group_obstacle_prediction": bool(any_front_group_obstacle_prediction),
            "any_front_group_uncertain": bool(any_front_group_uncertain),
            "any_front_group_matched_peak": bool(any_front_group_matched_peak),
        }

    # Apply per-scene action.
    actions = []
    for idx, row in scene.iterrows():
        sec = row_sec[idx]
        wid = row["temporal_window_id"]
        win_action = window_decision.get(wid, "FALLBACK_V4_TURN_RESAMPLE_STOP")

        if win_action in {ACTION_FAST, ACTION_SLOW, ACTION_PROBE}:
            action = win_action
        else:
            action = fallback_turn_resample_stop(sec)

        # Final safety override for full forward only.
        fl_state = str(sec["front_left"]["predicted_state"]).upper()
        f_state = str(sec["front"]["predicted_state"]).upper()
        fr_state = str(sec["front_right"]["predicted_state"]).upper()
        any_front_not_clear = (fl_state != STATE_CLEAR) or (f_state != STATE_CLEAR) or (fr_state != STATE_CLEAR)
        if (action in {ACTION_FAST, ACTION_SLOW}) and any_front_not_clear:
            action = ACTION_RESAMPLE
        actions.append(action)
    scene["recommended_action_v5"] = actions

    scene = add_true_conditions(scene, pred)
    if scene[["true_front_obstacle", "true_any_front_group_obstacle"]].isna().any().any():
        raise ValueError("Missing true-label rows after merge.")

    # Metrics.
    total_scenes = int(scene.shape[0])
    v5_counts, v5_pct = action_distribution(scene["recommended_action_v5"])
    v4_counts = v4.get("v4_policy_action_counts", {})
    v4_pct = v4.get("v4_policy_action_percentages", {})

    n_true_front_obstacle = int(scene["true_front_obstacle"].sum())
    n_true_any_front_group_obstacle = int(scene["true_any_front_group_obstacle"].sum())
    n_true_all_front_clear = int(scene["true_all_front_clear"].sum())
    n_true_left_blocked = int((scene["left"] == 1).sum())
    n_true_right_blocked = int((scene["right"] == 1).sum())
    n_front_blocked = int(scene["true_front_obstacle"].sum())

    full_forward = scene["recommended_action_v5"].isin([ACTION_FAST, ACTION_SLOW])
    probe = scene["recommended_action_v5"] == ACTION_PROBE

    fast_true_front_obs = int(((scene["recommended_action_v5"] == ACTION_FAST) & scene["true_front_obstacle"]).sum())
    slow_true_front_obs = int(((scene["recommended_action_v5"] == ACTION_SLOW) & scene["true_front_obstacle"]).sum())
    full_forward_any_front_obs = int((full_forward & scene["true_any_front_group_obstacle"]).sum())
    probe_any_front_obs = int((probe & scene["true_any_front_group_obstacle"]).sum())
    probe_true_front_obs = int((probe & scene["true_front_obstacle"]).sum())
    probe_all_front_clear = int((probe & scene["true_all_front_clear"]).sum())
    turn_left_left_blocked = int(((scene["recommended_action_v5"] == ACTION_LEFT) & (scene["left"] == 1)).sum())
    turn_right_right_blocked = int(((scene["recommended_action_v5"] == ACTION_RIGHT) & (scene["right"] == 1)).sum())

    full_forward_clear = int((full_forward & scene["true_all_front_clear"]).sum())
    full_or_probe_clear = int(((full_forward | probe) & scene["true_all_front_clear"]).sum())
    stop_resample_clear = int((scene["recommended_action_v5"].isin([ACTION_STOP, ACTION_RESAMPLE]) & scene["true_all_front_clear"]).sum())

    turn_front_blocked = int((scene["recommended_action_v5"].isin([ACTION_LEFT, ACTION_RIGHT]) & scene["true_front_obstacle"]).sum())
    stop_resample_front_blocked = int((scene["recommended_action_v5"].isin([ACTION_STOP, ACTION_RESAMPLE]) & scene["true_front_obstacle"]).sum())
    probe_front_blocked = int((probe & scene["true_front_obstacle"]).sum())
    full_forward_front_blocked = int((full_forward & scene["true_front_obstacle"]).sum())

    # Readiness heuristic: strict full-forward safety + useful clear-space mobility (forward/probe).
    unsafe_full_forward_count = full_forward_any_front_obs
    probe_risk_count = probe_any_front_obs
    forward_probe_rate_clear = full_or_probe_clear / max(n_true_all_front_clear, 1)
    ready_2d = (unsafe_full_forward_count == 0) and (forward_probe_rate_clear >= 0.25)

    # Add window-debug columns to scene output.
    scene["window_decision"] = scene["temporal_window_id"].map(window_decision)
    scene["window_size"] = scene["temporal_window_id"].map(lambda w: window_debug.get(w, {}).get("window_size", np.nan))
    scene["window_strong_clear_count_front"] = scene["temporal_window_id"].map(
        lambda w: window_debug.get(w, {}).get("strong_clear_count_front", np.nan)
    )
    scene["window_soft_clear_count_front"] = scene["temporal_window_id"].map(
        lambda w: window_debug.get(w, {}).get("soft_clear_count_front", np.nan)
    )

    result = {
        "total_scenes": total_scenes,
        "temporal_grouping_method": (
            "Deterministic grouping by predicted front-pattern key "
            "(front_left_state|front_state|front_right_state), sorted by sample_id, chunked into windows of 3."
        ),
        "v4_policy_action_counts": v4_counts,
        "v4_policy_action_percentages": v4_pct,
        "v5_policy_action_counts": v5_counts,
        "v5_policy_action_percentages": v5_pct,
        "safety_checks_v5": {
            "move_forward_fast_when_true_front_obstacle_count": fast_true_front_obs,
            "move_forward_fast_when_true_front_obstacle_rate": fast_true_front_obs / max(n_true_front_obstacle, 1),
            "move_forward_slow_when_true_front_obstacle_count": slow_true_front_obs,
            "move_forward_slow_when_true_front_obstacle_rate": slow_true_front_obs / max(n_true_front_obstacle, 1),
            "full_forward_motion_when_any_true_front_group_obstacle_count": full_forward_any_front_obs,
            "full_forward_motion_when_any_true_front_group_obstacle_rate": full_forward_any_front_obs
            / max(n_true_any_front_group_obstacle, 1),
            "probe_forward_when_true_front_obstacle_count": probe_true_front_obs,
            "probe_forward_when_true_front_obstacle_rate": probe_true_front_obs / max(n_true_front_obstacle, 1),
            "probe_forward_when_any_true_front_group_obstacle_count": probe_any_front_obs,
            "probe_forward_when_any_true_front_group_obstacle_rate": probe_any_front_obs
            / max(n_true_any_front_group_obstacle, 1),
            "turn_left_when_true_left_sector_obstacle_count": turn_left_left_blocked,
            "turn_left_when_true_left_sector_obstacle_rate": turn_left_left_blocked / max(n_true_left_blocked, 1),
            "turn_right_when_true_right_sector_obstacle_count": turn_right_right_blocked,
            "turn_right_when_true_right_sector_obstacle_rate": turn_right_right_blocked / max(n_true_right_blocked, 1),
            "unsafe_full_forward_count": unsafe_full_forward_count,
            "probe_forward_risk_count": probe_risk_count,
        },
        "efficiency_checks_v5": {
            "full_forward_when_all_front_truly_clear_count": full_forward_clear,
            "full_forward_when_all_front_truly_clear_rate": full_forward_clear / max(n_true_all_front_clear, 1),
            "probe_forward_when_all_front_truly_clear_count": probe_all_front_clear,
            "probe_forward_when_all_front_truly_clear_rate": probe_all_front_clear / max(n_true_all_front_clear, 1),
            "full_or_probe_when_all_front_truly_clear_count": full_or_probe_clear,
            "full_or_probe_when_all_front_truly_clear_rate": forward_probe_rate_clear,
            "stop_or_resample_when_all_front_truly_clear_count": stop_resample_clear,
            "stop_or_resample_when_all_front_truly_clear_rate": stop_resample_clear / max(n_true_all_front_clear, 1),
            "when_front_truly_blocked": {
                "turn_rate": turn_front_blocked / max(n_front_blocked, 1),
                "stop_or_resample_rate": stop_resample_front_blocked / max(n_front_blocked, 1),
                "probe_rate": probe_front_blocked / max(n_front_blocked, 1),
                "full_forward_rate": full_forward_front_blocked / max(n_front_blocked, 1),
            },
        },
        "v5_minus_v4_deltas_percentage_points": {
            k: v5_pct.get(k, 0.0) - v4_pct.get(k, 0.0)
            for k in sorted(set(list(v4_pct.keys()) + list(v5_pct.keys())))
        },
        "ready_for_simple_2d_navigation_simulation": ready_2d,
    }

    scene.to_csv(OUT_SCENE_CSV, index=False)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Plot v4 vs v5 distribution.
    labels = ACTION_ORDER_V5 + [k for k in sorted(set(list(v4_pct.keys()) + list(v5_pct.keys()))) if k not in ACTION_ORDER_V5]
    labels = list(dict.fromkeys(labels))
    x = np.arange(len(labels), dtype=float)
    w = 0.38
    vals4 = [v4_pct.get(a, 0.0) for a in labels]
    vals5 = [v5_pct.get(a, 0.0) for a in labels]

    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=140)
    ax.bar(x - w / 2.0, vals4, width=w, label="Policy v4")
    ax.bar(x + w / 2.0, vals5, width=w, label="Policy v5 temporal")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Scene percentage (%)")
    ax.set_title("Action Distribution: Policy v4 vs v5 Temporal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)

    print("\nPolicy v4 vs v5 action distribution:")
    for a in labels:
        print(f"  {a}: v4={v4_pct.get(a, 0.0):.2f}%  v5={v5_pct.get(a, 0.0):.2f}%")

    print("\nSafety checks (v5):")
    print(f"  unsafe full-forward count: {unsafe_full_forward_count}")
    print(f"  PROBE_FORWARD risk count (any true front-group obstacle): {probe_risk_count}")
    print(
        f"  full forward (FAST/SLOW) when any true front-group obstacle: "
        f"{full_forward_any_front_obs} ({100.0 * full_forward_any_front_obs / max(n_true_any_front_group_obstacle, 1):.2f}%)"
    )
    print(
        f"  PROBE_FORWARD when any true front-group obstacle: "
        f"{probe_any_front_obs} ({100.0 * probe_any_front_obs / max(n_true_any_front_group_obstacle, 1):.2f}%)"
    )

    print("\nEfficiency checks (v5):")
    print(
        f"  full forward when all front truly clear: {full_forward_clear} "
        f"({100.0 * full_forward_clear / max(n_true_all_front_clear, 1):.2f}%)"
    )
    print(
        f"  probe forward when all front truly clear: {probe_all_front_clear} "
        f"({100.0 * probe_all_front_clear / max(n_true_all_front_clear, 1):.2f}%)"
    )
    print(
        f"  full/probe combined when all front truly clear: {full_or_probe_clear} "
        f"({100.0 * forward_probe_rate_clear:.2f}%)"
    )
    print(
        f"  stop/resample when all front truly clear: {stop_resample_clear} "
        f"({100.0 * stop_resample_clear / max(n_true_all_front_clear, 1):.2f}%)"
    )

    print("\nDecision:")
    print(f"  ready_for_simple_2d_navigation_simulation: {ready_2d}")

    print(f"\nSaved v5 scene summary: {OUT_SCENE_CSV}")
    print(f"Saved v5 analysis JSON: {OUT_JSON}")
    print(f"Saved v5 distribution plot: {OUT_PNG}")


if __name__ == "__main__":
    main()

