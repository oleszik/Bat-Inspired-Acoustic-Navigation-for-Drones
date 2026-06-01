"""
Evaluate action policy v4 with conservative front-clear confidence guards.

Policy decisions use runtime-available prediction/features only.
Ground-truth labels are used only for evaluation metrics.
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
V3_JSON = DATASET_ROOT / "action_policy_v3_analysis.json"
V3_UNSAFE_JSON = DATASET_ROOT / "action_policy_v3_unsafe_forward_cases.json"

OUT_JSON = DATASET_ROOT / "action_policy_v4_analysis.json"
OUT_SCENE_CSV = DATASET_ROOT / "action_policy_v4_scene_summary.csv"
OUT_PNG = DATASET_ROOT / "action_policy_v4_distribution.png"

SECTORS = ["left", "front_left", "front", "front_right", "right"]
FRONT_GROUP = ["front_left", "front", "front_right"]

STATE_CLEAR = "CLEAR"
STATE_OBS = "OBSTACLE"
STATE_UNC = "UNCERTAIN"

ACTION_FAST = "MOVE_FORWARD_FAST"
ACTION_SLOW = "MOVE_FORWARD_SLOW"
ACTION_LEFT = "TURN_LEFT"
ACTION_RIGHT = "TURN_RIGHT"
ACTION_RESAMPLE = "SLOW_DOWN_AND_RESAMPLE"
ACTION_STOP = "STOP_OR_REVERSE"
ACTION_ORDER = [ACTION_FAST, ACTION_SLOW, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]


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
    # Runtime estimate only.
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


def policy_v4_action(sec: dict[str, dict[str, object]]) -> str:
    left = str(sec["left"]["predicted_state"]).upper()
    fl = str(sec["front_left"]["predicted_state"]).upper()
    f = str(sec["front"]["predicted_state"]).upper()
    fr = str(sec["front_right"]["predicted_state"]).upper()
    right = str(sec["right"]["predicted_state"]).upper()

    any_front_obstacle = (fl == STATE_OBS) or (f == STATE_OBS) or (fr == STATE_OBS)

    fast_allowed = is_strong_clear(sec["front_left"]) and is_strong_clear(sec["front"]) and is_strong_clear(sec["front_right"])
    slow_allowed = (
        is_strong_clear(sec["front"])
        and is_soft_clear(sec["front_left"])
        and is_soft_clear(sec["front_right"])
        and (not any_front_obstacle)
    )

    # front == OBSTACLE
    if f == STATE_OBS:
        if (left == STATE_CLEAR) and (right != STATE_CLEAR):
            action = ACTION_LEFT
        elif (right == STATE_CLEAR) and (left != STATE_CLEAR):
            action = ACTION_RIGHT
        elif (left == STATE_CLEAR) and (right == STATE_CLEAR):
            action = choose_turn_side_when_both_clear(sec["left"], sec["right"])
        else:
            action = ACTION_STOP

    # front == CLEAR
    elif f == STATE_CLEAR:
        if fast_allowed:
            action = ACTION_FAST
        elif slow_allowed:
            action = ACTION_SLOW
        elif (fl == STATE_OBS) and (fr == STATE_CLEAR):
            action = ACTION_RIGHT
        elif (fr == STATE_OBS) and (fl == STATE_CLEAR):
            action = ACTION_LEFT
        else:
            action = ACTION_RESAMPLE

    # front == UNCERTAIN
    elif f == STATE_UNC:
        # no forward movement
        if (left == STATE_CLEAR) and (right != STATE_CLEAR):
            action = ACTION_LEFT
        elif (right == STATE_CLEAR) and (left != STATE_CLEAR):
            action = ACTION_RIGHT
        else:
            action = ACTION_RESAMPLE
    else:
        action = ACTION_RESAMPLE

    # Final override.
    any_front_not_clear = (fl != STATE_CLEAR) or (f != STATE_CLEAR) or (fr != STATE_CLEAR)
    if (action in {ACTION_FAST, ACTION_SLOW}) and any_front_not_clear:
        action = ACTION_RESAMPLE

    return action


def build_scene_runtime_table(scene_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in scene_df.columns:
        raise ValueError("Scene CSV missing sample_id.")
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
    if not required_pred.issubset(pred_df.columns):
        raise ValueError(f"Prediction CSV missing required columns: {required_pred - set(pred_df.columns)}")

    pred = pred_df.copy()
    pred["sample_id"] = pred["sample_id"].astype(str)
    pred["sector"] = pred["sector"].astype(str)
    pred = pred.drop_duplicates(["sample_id", "sector"], keep="first")

    # Wide columns per sector.
    wide_parts = []
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
    for sec in SECTORS:
        sdf = pred[pred["sector"] == sec][["sample_id"] + feature_cols].copy()
        sdf = sdf.rename(columns={c: f"{sec}_{c}" for c in feature_cols})
        wide_parts.append(sdf)

    merged = scene_df.copy()
    merged["sample_id"] = merged["sample_id"].astype(str)
    for part in wide_parts:
        merged = merged.merge(part, on="sample_id", how="left")

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
    for p in [SCENE_CSV, PRED_CSV, V3_JSON, V3_UNSAFE_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    scene_in = pd.read_csv(SCENE_CSV)
    pred = pd.read_csv(PRED_CSV)
    v3 = json.loads(V3_JSON.read_text(encoding="utf-8"))
    v3_unsafe = json.loads(V3_UNSAFE_JSON.read_text(encoding="utf-8"))

    scene = build_scene_runtime_table(scene_in, pred)

    # Compute v4 action per scene.
    actions = []
    for _, row in scene.iterrows():
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
        actions.append(policy_v4_action(sec))
    scene["recommended_action_v4"] = actions

    scene = add_true_conditions(scene, pred)
    if scene[["true_front_obstacle", "true_any_front_group_obstacle"]].isna().any().any():
        raise ValueError("Missing true-label rows after merge.")

    total_scenes = int(scene.shape[0])
    v4_counts, v4_pct = action_distribution(scene["recommended_action_v4"])
    v3_counts = v3.get("v3_policy_action_counts", {})
    v3_pct = v3.get("v3_policy_action_percentages", {})
    v2_counts = v3.get("v2_policy_action_counts", {})
    v2_pct = v3.get("v2_policy_action_percentages", {})

    # Safety checks.
    n_true_front_obstacle = int(scene["true_front_obstacle"].sum())
    n_true_any_front_group_obstacle = int(scene["true_any_front_group_obstacle"].sum())
    n_true_all_front_clear = int(scene["true_all_front_clear"].sum())
    n_front_blocked = int(scene["true_front_obstacle"].sum())

    fast_true_front_obs = int(((scene["recommended_action_v4"] == ACTION_FAST) & scene["true_front_obstacle"]).sum())
    slow_true_front_obs = int(((scene["recommended_action_v4"] == ACTION_SLOW) & scene["true_front_obstacle"]).sum())
    any_forward_any_front_obs = int(
        (scene["recommended_action_v4"].isin([ACTION_FAST, ACTION_SLOW]) & scene["true_any_front_group_obstacle"]).sum()
    )

    # Fix check for known unsafe samples.
    known_unsafe_ids = set(str(x) for x in v3_unsafe.get("unsafe_sample_ids", []))
    known_rows = scene[scene["sample_id"].astype(str).isin(known_unsafe_ids)].copy()
    known_still_unsafe = known_rows[
        known_rows["recommended_action_v4"].isin([ACTION_FAST, ACTION_SLOW]) & known_rows["true_any_front_group_obstacle"]
    ]["sample_id"].astype(str).tolist()

    # Efficiency checks.
    forward_when_front_clear = int(
        (scene["recommended_action_v4"].isin([ACTION_FAST, ACTION_SLOW]) & scene["true_all_front_clear"]).sum()
    )
    cautious_when_front_clear = int(
        (scene["recommended_action_v4"].isin([ACTION_STOP, ACTION_RESAMPLE]) & scene["true_all_front_clear"]).sum()
    )
    turn_when_front_blocked = int(
        (scene["recommended_action_v4"].isin([ACTION_LEFT, ACTION_RIGHT]) & scene["true_front_obstacle"]).sum()
    )
    stop_or_resample_when_front_blocked = int(
        (scene["recommended_action_v4"].isin([ACTION_STOP, ACTION_RESAMPLE]) & scene["true_front_obstacle"]).sum()
    )
    forward_when_front_blocked = int(
        (scene["recommended_action_v4"].isin([ACTION_FAST, ACTION_SLOW]) & scene["true_front_obstacle"]).sum()
    )

    unsafe_forward_count = any_forward_any_front_obs
    fixed_known_unsafe = len(known_still_unsafe) == 0
    forward_rate_clear = forward_when_front_clear / max(n_true_all_front_clear, 1)
    ready_2d = (unsafe_forward_count == 0) and (forward_rate_clear >= 0.25)

    result = {
        "total_scenes": total_scenes,
        "v2_policy_action_counts": v2_counts,
        "v2_policy_action_percentages": v2_pct,
        "v3_policy_action_counts": v3_counts,
        "v3_policy_action_percentages": v3_pct,
        "v4_policy_action_counts": v4_counts,
        "v4_policy_action_percentages": v4_pct,
        "safety_checks_v4": {
            "move_forward_fast_when_true_front_obstacle_count": fast_true_front_obs,
            "move_forward_fast_when_true_front_obstacle_rate": fast_true_front_obs / max(n_true_front_obstacle, 1),
            "move_forward_slow_when_true_front_obstacle_count": slow_true_front_obs,
            "move_forward_slow_when_true_front_obstacle_rate": slow_true_front_obs / max(n_true_front_obstacle, 1),
            "any_forward_motion_when_any_true_front_group_obstacle_count": any_forward_any_front_obs,
            "any_forward_motion_when_any_true_front_group_obstacle_rate": any_forward_any_front_obs
            / max(n_true_any_front_group_obstacle, 1),
            "unsafe_forward_count": unsafe_forward_count,
            "known_unsafe_scene_ids_from_v3": sorted(list(known_unsafe_ids)),
            "known_unsafe_scene_ids_still_unsafe_in_v4": sorted(list(known_still_unsafe)),
            "known_unsafe_scenes_fixed": fixed_known_unsafe,
        },
        "efficiency_checks_v4": {
            "forward_motion_when_all_front_truly_clear_count": forward_when_front_clear,
            "forward_motion_when_all_front_truly_clear_rate": forward_rate_clear,
            "stop_or_resample_when_all_front_truly_clear_count": cautious_when_front_clear,
            "stop_or_resample_when_all_front_truly_clear_rate": cautious_when_front_clear / max(
                n_true_all_front_clear, 1
            ),
            "when_front_truly_blocked": {
                "turn_rate": turn_when_front_blocked / max(n_front_blocked, 1),
                "stop_or_resample_rate": stop_or_resample_when_front_blocked / max(n_front_blocked, 1),
                "forward_rate": forward_when_front_blocked / max(n_front_blocked, 1),
            },
        },
        "comparison_deltas_percentage_points": {
            "v4_minus_v2": {
                k: v4_pct.get(k, 0.0) - v2_pct.get(k, 0.0)
                for k in sorted(set(list(v2_pct.keys()) + list(v4_pct.keys())))
            },
            "v4_minus_v3": {
                k: v4_pct.get(k, 0.0) - v3_pct.get(k, 0.0)
                for k in sorted(set(list(v3_pct.keys()) + list(v4_pct.keys())))
            },
        },
        "ready_for_simple_2d_navigation_simulation": ready_2d,
    }

    scene.to_csv(OUT_SCENE_CSV, index=False)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Plot distribution v2/v3/v4.
    labels = ACTION_ORDER + [k for k in sorted(set(list(v2_pct.keys()) + list(v3_pct.keys()) + list(v4_pct.keys()))) if k not in ACTION_ORDER]
    labels = list(dict.fromkeys(labels))
    x = np.arange(len(labels), dtype=float)
    w = 0.26
    vals2 = [v2_pct.get(a, 0.0) for a in labels]
    vals3 = [v3_pct.get(a, 0.0) for a in labels]
    vals4 = [v4_pct.get(a, 0.0) for a in labels]

    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=140)
    ax.bar(x - w, vals2, width=w, label="Policy v2")
    ax.bar(x, vals3, width=w, label="Policy v3")
    ax.bar(x + w, vals4, width=w, label="Policy v4")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Scene percentage (%)")
    ax.set_title("Action Distribution: Policy v2 vs v3 vs v4")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)

    print("\nPolicy v2 vs v3 vs v4 action distribution:")
    for a in labels:
        print(f"  {a}: v2={v2_pct.get(a, 0.0):.2f}%  v3={v3_pct.get(a, 0.0):.2f}%  v4={v4_pct.get(a, 0.0):.2f}%")

    print("\nSafety checks (v4):")
    print(
        f"  MOVE_FORWARD_FAST when true front obstacle: {fast_true_front_obs} "
        f"({100.0 * fast_true_front_obs / max(n_true_front_obstacle, 1):.2f}%)"
    )
    print(
        f"  MOVE_FORWARD_SLOW when true front obstacle: {slow_true_front_obs} "
        f"({100.0 * slow_true_front_obs / max(n_true_front_obstacle, 1):.2f}%)"
    )
    print(
        f"  Any forward motion when any true front-group obstacle: {any_forward_any_front_obs} "
        f"({100.0 * any_forward_any_front_obs / max(n_true_any_front_group_obstacle, 1):.2f}%)"
    )
    print(f"  Known unsafe scenes fixed (scene_01718, scene_03486): {fixed_known_unsafe}")
    if known_still_unsafe:
        print(f"  Still unsafe known scenes: {known_still_unsafe}")

    print("\nEfficiency checks (v4):")
    print(
        f"  Forward when all front truly clear: {forward_when_front_clear} "
        f"({100.0 * forward_rate_clear:.2f}%)"
    )
    print(
        f"  Stop/resample when all front truly clear: {cautious_when_front_clear} "
        f"({100.0 * cautious_when_front_clear / max(n_true_all_front_clear, 1):.2f}%)"
    )
    print(
        "  When front truly blocked -> turn/stop-resample/forward rates: "
        f"{100.0 * turn_when_front_blocked / max(n_front_blocked, 1):.2f}% / "
        f"{100.0 * stop_or_resample_when_front_blocked / max(n_front_blocked, 1):.2f}% / "
        f"{100.0 * forward_when_front_blocked / max(n_front_blocked, 1):.2f}%"
    )

    print("\nDecision:")
    print(f"  unsafe_forward_count: {unsafe_forward_count}")
    print(f"  fixed_known_unsafe_scenes: {fixed_known_unsafe}")
    print(f"  forward_motion_rate_on_truly_clear_front: {100.0 * forward_rate_clear:.2f}%")
    print(f"  ready_for_simple_2d_navigation_simulation: {ready_2d}")

    print(f"\nSaved v4 scene summary: {OUT_SCENE_CSV}")
    print(f"Saved v4 analysis JSON: {OUT_JSON}")
    print(f"Saved v4 distribution plot: {OUT_PNG}")


if __name__ == "__main__":
    main()

