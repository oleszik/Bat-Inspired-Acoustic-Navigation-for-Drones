"""
Evaluate action policy v3 with a hard forward-safety guard.

Policy v3 is derived from the existing scene-level acoustic predictions and is
compared against policy v2 metrics from action_policy_v2_analysis.json.
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
V2_JSON = DATASET_ROOT / "action_policy_v2_analysis.json"

OUT_JSON = DATASET_ROOT / "action_policy_v3_analysis.json"
OUT_SCENE_CSV = DATASET_ROOT / "action_policy_v3_scene_summary.csv"
OUT_PNG = DATASET_ROOT / "action_policy_v3_distribution.png"

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


def choose_turn_side_when_both_clear(row: pd.Series) -> str:
    l = pd.to_numeric(row.get("left_distance_m"), errors="coerce")
    r = pd.to_numeric(row.get("right_distance_m"), errors="coerce")
    if pd.notna(l) and pd.notna(r):
        return ACTION_LEFT if float(l) >= float(r) else ACTION_RIGHT
    if pd.notna(l):
        return ACTION_LEFT
    if pd.notna(r):
        return ACTION_RIGHT
    return ACTION_LEFT


def policy_v3_action(row: pd.Series) -> str:
    left = str(row["left_state"])
    front_left = str(row["front_left_state"])
    front = str(row["front_state"])
    front_right = str(row["front_right_state"])
    right = str(row["right_state"])

    any_front_obstacle = (front_left == STATE_OBS) or (front == STATE_OBS) or (front_right == STATE_OBS)

    # If both side sectors are blocked and any front sector is obstacle, force stop.
    if (left == STATE_OBS) and (right == STATE_OBS) and any_front_obstacle:
        base_action = ACTION_STOP
    else:
        # front == OBSTACLE
        if front == STATE_OBS:
            if (left == STATE_CLEAR) and (right != STATE_CLEAR):
                base_action = ACTION_LEFT
            elif (right == STATE_CLEAR) and (left != STATE_CLEAR):
                base_action = ACTION_RIGHT
            elif (left == STATE_CLEAR) and (right == STATE_CLEAR):
                base_action = choose_turn_side_when_both_clear(row)
            else:
                base_action = ACTION_STOP

        # front == CLEAR
        elif front == STATE_CLEAR:
            if (front_left == STATE_CLEAR) and (front_right == STATE_CLEAR):
                # Hard guard allows FAST only with all three front-group sectors clear.
                base_action = ACTION_FAST
            elif (front_left != STATE_OBS) and (front_right != STATE_OBS) and (not any_front_obstacle):
                # Hard guard for SLOW: no front-group obstacle.
                base_action = ACTION_SLOW
            elif (front_left == STATE_OBS) and (front_right == STATE_CLEAR):
                base_action = ACTION_RIGHT
            elif (front_right == STATE_OBS) and (front_left == STATE_CLEAR):
                base_action = ACTION_LEFT
            else:
                base_action = ACTION_RESAMPLE

        # front == UNCERTAIN
        elif front == STATE_UNC:
            if (front_left == STATE_CLEAR) and (front_right == STATE_CLEAR) and (not any_front_obstacle):
                base_action = ACTION_SLOW
            else:
                base_action = ACTION_RESAMPLE
        else:
            base_action = ACTION_RESAMPLE

    # Final safety override: never allow forward action if any front-group obstacle exists.
    if (base_action in {ACTION_FAST, ACTION_SLOW}) and any_front_obstacle:
        return ACTION_RESAMPLE
    return base_action


def add_true_conditions(scene: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    truth = pred.pivot(index="sample_id", columns="sector", values="true_has_obstacle")
    for sec in ["left", "front_left", "front", "front_right", "right"]:
        if sec not in truth.columns:
            truth[sec] = 0
    truth = truth.fillna(0).astype(int)
    truth["true_front_obstacle"] = truth["front"] == 1
    truth["true_any_front_group_obstacle"] = truth[["front_left", "front", "front_right"]].max(axis=1) == 1
    truth["true_all_front_clear"] = truth[["front_left", "front", "front_right"]].sum(axis=1) == 0
    return scene.merge(truth.reset_index(), on="sample_id", how="left")


def main() -> None:
    for p in [SCENE_CSV, PRED_CSV, V2_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    scene = pd.read_csv(SCENE_CSV)
    pred = pd.read_csv(PRED_CSV)
    v2 = json.loads(V2_JSON.read_text(encoding="utf-8"))

    required_scene = {
        "sample_id",
        "left_state",
        "front_left_state",
        "front_state",
        "front_right_state",
        "right_state",
        "left_distance_m",
        "right_distance_m",
    }
    if not required_scene.issubset(scene.columns):
        raise ValueError(f"Scene CSV missing columns: {required_scene - set(scene.columns)}")
    required_pred = {"sample_id", "sector", "true_has_obstacle"}
    if not required_pred.issubset(pred.columns):
        raise ValueError(f"Prediction CSV missing columns: {required_pred - set(pred.columns)}")

    scene = scene.copy()
    scene["recommended_action_v3"] = scene.apply(policy_v3_action, axis=1)
    scene = add_true_conditions(scene, pred)

    if scene[["true_front_obstacle", "true_any_front_group_obstacle"]].isna().any().any():
        raise ValueError("Missing true-label rows after merge.")

    total_scenes = int(scene.shape[0])
    v3_counts, v3_pct = action_distribution(scene["recommended_action_v3"])

    v2_counts = v2.get("v2_policy_action_counts", {})
    v2_pct = v2.get("v2_policy_action_percentages", {})

    # Safety checks.
    n_true_front_obstacle = int(scene["true_front_obstacle"].sum())
    n_true_any_front_group_obstacle = int(scene["true_any_front_group_obstacle"].sum())
    n_true_all_front_clear = int(scene["true_all_front_clear"].sum())
    n_true_left_blocked = int((scene["left"] == 1).sum())
    n_true_right_blocked = int((scene["right"] == 1).sum())
    n_front_blocked = int(scene["true_front_obstacle"].sum())

    fast_true_front_obs = int(((scene["recommended_action_v3"] == ACTION_FAST) & scene["true_front_obstacle"]).sum())
    slow_true_front_obs = int(((scene["recommended_action_v3"] == ACTION_SLOW) & scene["true_front_obstacle"]).sum())
    any_forward_any_front_obs = int(
        (
            scene["recommended_action_v3"].isin([ACTION_FAST, ACTION_SLOW])
            & scene["true_any_front_group_obstacle"]
        ).sum()
    )
    turn_left_left_blocked = int(((scene["recommended_action_v3"] == ACTION_LEFT) & (scene["left"] == 1)).sum())
    turn_right_right_blocked = int(((scene["recommended_action_v3"] == ACTION_RIGHT) & (scene["right"] == 1)).sum())

    # Efficiency checks.
    forward_when_front_clear = int(
        (
            scene["recommended_action_v3"].isin([ACTION_FAST, ACTION_SLOW])
            & scene["true_all_front_clear"]
        ).sum()
    )
    cautious_when_front_clear = int(
        (
            scene["recommended_action_v3"].isin([ACTION_RESAMPLE, ACTION_STOP])
            & scene["true_all_front_clear"]
        ).sum()
    )
    turn_when_front_blocked = int(
        (
            scene["recommended_action_v3"].isin([ACTION_LEFT, ACTION_RIGHT])
            & scene["true_front_obstacle"]
        ).sum()
    )
    stop_or_resample_when_front_blocked = int(
        (
            scene["recommended_action_v3"].isin([ACTION_STOP, ACTION_RESAMPLE])
            & scene["true_front_obstacle"]
        ).sum()
    )
    forward_when_front_blocked = int(
        (
            scene["recommended_action_v3"].isin([ACTION_FAST, ACTION_SLOW])
            & scene["true_front_obstacle"]
        ).sum()
    )

    unsafe_removed = any_forward_any_front_obs == 0
    forward_useful = (v3_pct.get(ACTION_FAST, 0.0) + v3_pct.get(ACTION_SLOW, 0.0)) >= 3.0
    ready_2d = unsafe_removed and forward_useful

    interpretation = []
    interpretation.append(
        "Unsafe forward cases removed by hard guard." if unsafe_removed else "Unsafe forward cases remain."
    )
    interpretation.append(
        "Forward motion remains useful." if forward_useful else "Forward motion is likely too limited."
    )
    interpretation.append(
        "Policy v3 is ready for simple 2D navigation simulation with safety monitoring."
        if ready_2d
        else "Policy v3 is not yet ready for practical 2D simulation efficiency."
    )

    result = {
        "total_scenes": total_scenes,
        "v2_policy_action_counts": v2_counts,
        "v2_policy_action_percentages": v2_pct,
        "v3_policy_action_counts": v3_counts,
        "v3_policy_action_percentages": v3_pct,
        "safety_checks_v3": {
            "move_forward_fast_when_true_front_obstacle_count": fast_true_front_obs,
            "move_forward_fast_when_true_front_obstacle_rate": fast_true_front_obs / max(n_true_front_obstacle, 1),
            "move_forward_slow_when_true_front_obstacle_count": slow_true_front_obs,
            "move_forward_slow_when_true_front_obstacle_rate": slow_true_front_obs / max(n_true_front_obstacle, 1),
            "any_forward_motion_when_any_true_front_group_obstacle_count": any_forward_any_front_obs,
            "any_forward_motion_when_any_true_front_group_obstacle_rate": any_forward_any_front_obs
            / max(n_true_any_front_group_obstacle, 1),
            "turn_left_when_true_left_sector_obstacle_count": turn_left_left_blocked,
            "turn_left_when_true_left_sector_obstacle_rate": turn_left_left_blocked / max(n_true_left_blocked, 1),
            "turn_right_when_true_right_sector_obstacle_count": turn_right_right_blocked,
            "turn_right_when_true_right_sector_obstacle_rate": turn_right_right_blocked / max(n_true_right_blocked, 1),
        },
        "efficiency_checks_v3": {
            "forward_motion_when_all_front_truly_clear_count": forward_when_front_clear,
            "forward_motion_when_all_front_truly_clear_rate": forward_when_front_clear / max(n_true_all_front_clear, 1),
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
        "v2_vs_v3_deltas_percentage_points": {
            k: v3_pct.get(k, 0.0) - v2_pct.get(k, 0.0)
            for k in sorted(set(list(v2_pct.keys()) + list(v3_pct.keys())))
        },
        "unsafe_forward_cases_removed": unsafe_removed,
        "forward_motion_still_useful": forward_useful,
        "ready_for_simple_2d_navigation_simulation": ready_2d,
        "interpretation": " ".join(interpretation),
    }

    scene.to_csv(OUT_SCENE_CSV, index=False)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Plot: v2 vs v3 distribution.
    labels = ACTION_ORDER + [k for k in sorted(v2_pct.keys()) if k not in ACTION_ORDER]
    labels = list(dict.fromkeys(labels))
    x = np.arange(len(labels), dtype=float)
    w = 0.38
    v2_vals = [v2_pct.get(a, 0.0) for a in labels]
    v3_vals = [v3_pct.get(a, 0.0) for a in labels]

    fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=140)
    ax.bar(x - w / 2.0, v2_vals, width=w, label="Policy v2")
    ax.bar(x + w / 2.0, v3_vals, width=w, label="Policy v3")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Scene percentage (%)")
    ax.set_title("Action Distribution: Policy v2 vs Policy v3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)

    # Print summary.
    print("\nPolicy v2 vs v3 action distribution:")
    for a in labels:
        print(f"  {a}: v2={v2_pct.get(a, 0.0):.2f}%  v3={v3_pct.get(a, 0.0):.2f}%")

    print("\nSafety checks (v3):")
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
    print(
        f"  TURN_LEFT when true left blocked: {turn_left_left_blocked} "
        f"({100.0 * turn_left_left_blocked / max(n_true_left_blocked, 1):.2f}%)"
    )
    print(
        f"  TURN_RIGHT when true right blocked: {turn_right_right_blocked} "
        f"({100.0 * turn_right_right_blocked / max(n_true_right_blocked, 1):.2f}%)"
    )

    print("\nEfficiency checks (v3):")
    print(
        f"  Forward when all front truly clear: {forward_when_front_clear} "
        f"({100.0 * forward_when_front_clear / max(n_true_all_front_clear, 1):.2f}%)"
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
    print(f"  unsafe_forward_cases_removed: {unsafe_removed}")
    print(f"  forward_motion_still_useful: {forward_useful}")
    print(f"  ready_for_simple_2d_navigation_simulation: {ready_2d}")
    print(f"  {result['interpretation']}")

    print(f"\nSaved v3 scene summary: {OUT_SCENE_CSV}")
    print(f"Saved v3 analysis JSON: {OUT_JSON}")
    print(f"Saved v3 distribution plot: {OUT_PNG}")


if __name__ == "__main__":
    main()

