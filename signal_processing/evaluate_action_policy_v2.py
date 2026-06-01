"""
Evaluate a less-conservative scene-level action policy (v2) for multi-sector perception.

This script does not change acoustic predictions. It only remaps scene-level actions
from the existing predicted sector states/distances and compares against the old policy.
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

OUT_JSON = DATASET_ROOT / "action_policy_v2_analysis.json"
OUT_SCENE_CSV = DATASET_ROOT / "action_policy_v2_scene_summary.csv"
OUT_PNG = DATASET_ROOT / "action_policy_v2_distribution.png"

STATE_CLEAR = "CLEAR"
STATE_OBS = "OBSTACLE"
STATE_UNC = "UNCERTAIN"

OLD_ACTION = "recommended_action"
NEW_ACTION = "recommended_action_v2"

OLD_ACTION_ORDER = [
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SLOW_DOWN_AND_RESAMPLE",
    "STOP_OR_REVERSE",
]
NEW_ACTION_ORDER = [
    "MOVE_FORWARD_FAST",
    "MOVE_FORWARD_SLOW",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SLOW_DOWN_AND_RESAMPLE",
    "STOP_OR_REVERSE",
]


def action_distribution(series: pd.Series) -> tuple[dict[str, int], dict[str, float]]:
    total = int(series.shape[0])
    counts = {str(k): int(v) for k, v in series.value_counts().to_dict().items()}
    pct = {k: (100.0 * v / total if total > 0 else 0.0) for k, v in counts.items()}
    return counts, pct


def choose_turn_side_when_both_clear(row: pd.Series) -> str:
    """
    If both side sectors are CLEAR while front is OBSTACLE:
    choose side with larger clearance distance if available.
    Distances can be NaN for CLEAR states; default to TURN_LEFT then.
    """
    l = pd.to_numeric(row.get("left_distance_m"), errors="coerce")
    r = pd.to_numeric(row.get("right_distance_m"), errors="coerce")
    if pd.notna(l) and pd.notna(r):
        return "TURN_LEFT" if float(l) >= float(r) else "TURN_RIGHT"
    if pd.notna(l):
        return "TURN_LEFT"
    if pd.notna(r):
        return "TURN_RIGHT"
    return "TURN_LEFT"


def policy_v2_action(row: pd.Series) -> str:
    left = str(row["left_state"])
    front_left = str(row["front_left_state"])
    front = str(row["front_state"])
    front_right = str(row["front_right_state"])
    right = str(row["right_state"])

    any_front_obstacle = (front_left == STATE_OBS) or (front == STATE_OBS) or (front_right == STATE_OBS)

    # Global safety override.
    if (left == STATE_OBS) and (right == STATE_OBS) and any_front_obstacle:
        return "STOP_OR_REVERSE"

    # Front obstacle branch.
    if front == STATE_OBS:
        if (left == STATE_CLEAR) and (right != STATE_CLEAR):
            return "TURN_LEFT"
        if (right == STATE_CLEAR) and (left != STATE_CLEAR):
            return "TURN_RIGHT"
        if (left == STATE_CLEAR) and (right == STATE_CLEAR):
            return choose_turn_side_when_both_clear(row)
        return "STOP_OR_REVERSE"

    # Front clear branch.
    if front == STATE_CLEAR:
        if (front_left == STATE_CLEAR) and (front_right == STATE_CLEAR):
            return "MOVE_FORWARD_FAST"
        if (front_left != STATE_OBS) and (front_right != STATE_OBS):
            return "MOVE_FORWARD_SLOW"
        if (front_left == STATE_OBS) and (front_right == STATE_CLEAR):
            return "TURN_RIGHT"
        if (front_right == STATE_OBS) and (front_left == STATE_CLEAR):
            return "TURN_LEFT"
        return "SLOW_DOWN_AND_RESAMPLE"

    # Front uncertain branch.
    if front == STATE_UNC:
        if (front_left == STATE_CLEAR) and (front_right == STATE_CLEAR):
            return "MOVE_FORWARD_SLOW"
        return "SLOW_DOWN_AND_RESAMPLE"

    return "SLOW_DOWN_AND_RESAMPLE"


def add_true_conditions(scene: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    truth = pred.pivot(index="sample_id", columns="sector", values="true_has_obstacle")
    for sec in ["left", "front_left", "front", "front_right", "right"]:
        if sec not in truth.columns:
            truth[sec] = 0
    truth = truth.fillna(0).astype(int)
    truth["true_front_obstacle"] = truth["front"] == 1
    truth["true_front_clear"] = truth["front"] == 0
    truth["true_any_front_group_obstacle"] = truth[["front_left", "front", "front_right"]].max(axis=1) == 1
    truth["true_all_front_clear"] = truth[["front_left", "front", "front_right"]].sum(axis=1) == 0
    merged = scene.merge(truth.reset_index(), on="sample_id", how="left")
    return merged


def main() -> None:
    if not SCENE_CSV.exists():
        raise FileNotFoundError(f"Missing scene summary: {SCENE_CSV}")
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {PRED_CSV}")

    scene = pd.read_csv(SCENE_CSV)
    pred = pd.read_csv(PRED_CSV)

    required_scene_cols = {
        "sample_id",
        "left_state",
        "front_left_state",
        "front_state",
        "front_right_state",
        "right_state",
        "left_distance_m",
        "right_distance_m",
        OLD_ACTION,
    }
    if not required_scene_cols.issubset(scene.columns):
        missing = required_scene_cols - set(scene.columns)
        raise ValueError(f"Scene CSV missing required columns: {missing}")
    required_pred_cols = {"sample_id", "sector", "true_has_obstacle"}
    if not required_pred_cols.issubset(pred.columns):
        missing = required_pred_cols - set(pred.columns)
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    scene = scene.copy()
    scene[NEW_ACTION] = scene.apply(policy_v2_action, axis=1)
    scene = add_true_conditions(scene, pred)

    if scene[["true_front_obstacle", "true_any_front_group_obstacle"]].isna().any().any():
        raise ValueError("Missing true-label merge rows by sample_id.")

    total_scenes = int(scene.shape[0])

    old_counts, old_pct = action_distribution(scene[OLD_ACTION])
    new_counts, new_pct = action_distribution(scene[NEW_ACTION])

    # Safety checks for v2.
    n_true_front_obstacle = int(scene["true_front_obstacle"].sum())
    n_true_any_front_group_obstacle = int(scene["true_any_front_group_obstacle"].sum())
    n_true_all_front_clear = int(scene["true_all_front_clear"].sum())
    n_true_left_blocked = int((scene["left"] == 1).sum())
    n_true_right_blocked = int((scene["right"] == 1).sum())

    move_fast_true_front_obs = int(
        ((scene[NEW_ACTION] == "MOVE_FORWARD_FAST") & scene["true_front_obstacle"]).sum()
    )
    move_slow_true_front_obs = int(
        ((scene[NEW_ACTION] == "MOVE_FORWARD_SLOW") & scene["true_front_obstacle"]).sum()
    )
    any_forward_any_front_obs = int(
        (
            scene[NEW_ACTION].isin(["MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW"])
            & scene["true_any_front_group_obstacle"]
        ).sum()
    )
    turn_left_when_left_blocked = int(((scene[NEW_ACTION] == "TURN_LEFT") & (scene["left"] == 1)).sum())
    turn_right_when_right_blocked = int(((scene[NEW_ACTION] == "TURN_RIGHT") & (scene["right"] == 1)).sum())

    # Efficiency checks for v2.
    forward_when_all_front_clear = int(
        (
            scene[NEW_ACTION].isin(["MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW"])
            & scene["true_all_front_clear"]
        ).sum()
    )
    cautious_when_all_front_clear = int(
        (
            scene[NEW_ACTION].isin(["STOP_OR_REVERSE", "SLOW_DOWN_AND_RESAMPLE"])
            & scene["true_all_front_clear"]
        ).sum()
    )

    front_blocked = scene["true_front_obstacle"]
    n_front_blocked = int(front_blocked.sum())
    turn_when_front_blocked = int(
        (scene[NEW_ACTION].isin(["TURN_LEFT", "TURN_RIGHT"]) & front_blocked).sum()
    )
    stop_or_resample_when_front_blocked = int(
        (scene[NEW_ACTION].isin(["STOP_OR_REVERSE", "SLOW_DOWN_AND_RESAMPLE"]) & front_blocked).sum()
    )
    forward_when_front_blocked = int(
        (scene[NEW_ACTION].isin(["MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW"]) & front_blocked).sum()
    )

    interpretation = []
    if any_forward_any_front_obs == 0:
        interpretation.append("Policy v2 remains safety-oriented for front-group obstacles.")
    else:
        interpretation.append("Policy v2 introduces unsafe forward motion in blocked front-group scenes.")

    old_forward_pct = old_pct.get("MOVE_FORWARD", 0.0)
    new_forward_pct = new_pct.get("MOVE_FORWARD_FAST", 0.0) + new_pct.get("MOVE_FORWARD_SLOW", 0.0)
    if new_forward_pct > old_forward_pct:
        interpretation.append("Forward movement recovery improved versus old policy.")
    else:
        interpretation.append("Forward movement did not improve versus old policy.")

    cautious_rate = cautious_when_all_front_clear / max(n_true_all_front_clear, 1)
    if cautious_rate > 0.5:
        interpretation.append("Policy is still conservative in truly clear front scenes.")
    else:
        interpretation.append("Policy caution level in truly clear scenes is moderate.")

    ready_for_2d = (any_forward_any_front_obs == 0) and (new_forward_pct >= 10.0)
    interpretation.append(
        "Ready for simple 2D navigation simulation with safety monitoring."
        if ready_for_2d
        else "Not yet ready for efficient 2D simulation; useful as safety-first baseline."
    )

    result = {
        "total_scenes": total_scenes,
        "old_policy_action_counts": old_counts,
        "old_policy_action_percentages": old_pct,
        "v2_policy_action_counts": new_counts,
        "v2_policy_action_percentages": new_pct,
        "safety_checks_v2": {
            "move_forward_fast_when_true_front_obstacle_count": move_fast_true_front_obs,
            "move_forward_fast_when_true_front_obstacle_rate": move_fast_true_front_obs / max(
                n_true_front_obstacle, 1
            ),
            "move_forward_slow_when_true_front_obstacle_count": move_slow_true_front_obs,
            "move_forward_slow_when_true_front_obstacle_rate": move_slow_true_front_obs / max(
                n_true_front_obstacle, 1
            ),
            "any_forward_motion_when_any_true_front_group_obstacle_count": any_forward_any_front_obs,
            "any_forward_motion_when_any_true_front_group_obstacle_rate": any_forward_any_front_obs
            / max(n_true_any_front_group_obstacle, 1),
            "turn_left_when_true_left_sector_obstacle_count": turn_left_when_left_blocked,
            "turn_left_when_true_left_sector_obstacle_rate": turn_left_when_left_blocked / max(
                n_true_left_blocked, 1
            ),
            "turn_right_when_true_right_sector_obstacle_count": turn_right_when_right_blocked,
            "turn_right_when_true_right_sector_obstacle_rate": turn_right_when_right_blocked
            / max(n_true_right_blocked, 1),
        },
        "efficiency_checks_v2": {
            "forward_motion_when_all_front_truly_clear_count": forward_when_all_front_clear,
            "forward_motion_when_all_front_truly_clear_rate": forward_when_all_front_clear
            / max(n_true_all_front_clear, 1),
            "stop_or_resample_when_all_front_truly_clear_count": cautious_when_all_front_clear,
            "stop_or_resample_when_all_front_truly_clear_rate": cautious_when_all_front_clear
            / max(n_true_all_front_clear, 1),
            "when_front_truly_blocked": {
                "turn_rate": turn_when_front_blocked / max(n_front_blocked, 1),
                "stop_or_resample_rate": stop_or_resample_when_front_blocked / max(n_front_blocked, 1),
                "forward_rate": forward_when_front_blocked / max(n_front_blocked, 1),
            },
        },
        "interpretation": " ".join(interpretation),
    }

    # Save scene summary with v2 action.
    scene.to_csv(OUT_SCENE_CSV, index=False)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Plot old vs v2 action distribution.
    labels = OLD_ACTION_ORDER + [x for x in NEW_ACTION_ORDER if x not in OLD_ACTION_ORDER]
    labels = list(dict.fromkeys(labels))
    old_vals = [old_pct.get(a, 0.0) for a in labels]
    new_vals = [new_pct.get(a, 0.0) for a in labels]

    x = np.arange(len(labels), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=140)
    ax.bar(x - w / 2.0, old_vals, width=w, label="Old policy")
    ax.bar(x + w / 2.0, new_vals, width=w, label="Policy v2")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Scene percentage (%)")
    ax.set_title("Scene-Level Action Distribution: Old vs Policy v2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)

    # Print requested summary.
    print("\nPolicy v2 action distribution:")
    for a in NEW_ACTION_ORDER + sorted([k for k in new_counts.keys() if k not in NEW_ACTION_ORDER]):
        if a in new_counts:
            print(f"  {a}: {new_counts[a]} ({new_pct[a]:.2f}%)")

    print("\nSafety checks (v2):")
    print(
        "  MOVE_FORWARD_FAST when true front obstacle: "
        f"{move_fast_true_front_obs} ({100.0 * move_fast_true_front_obs / max(n_true_front_obstacle, 1):.2f}%)"
    )
    print(
        "  MOVE_FORWARD_SLOW when true front obstacle: "
        f"{move_slow_true_front_obs} ({100.0 * move_slow_true_front_obs / max(n_true_front_obstacle, 1):.2f}%)"
    )
    print(
        "  Any forward motion when any true front-group obstacle: "
        f"{any_forward_any_front_obs} ({100.0 * any_forward_any_front_obs / max(n_true_any_front_group_obstacle, 1):.2f}%)"
    )
    print(
        "  TURN_LEFT when true left sector blocked: "
        f"{turn_left_when_left_blocked} ({100.0 * turn_left_when_left_blocked / max(n_true_left_blocked, 1):.2f}%)"
    )
    print(
        "  TURN_RIGHT when true right sector blocked: "
        f"{turn_right_when_right_blocked} ({100.0 * turn_right_when_right_blocked / max(n_true_right_blocked, 1):.2f}%)"
    )

    print("\nEfficiency checks (v2):")
    print(
        "  Forward motion when all front truly clear: "
        f"{forward_when_all_front_clear} ({100.0 * forward_when_all_front_clear / max(n_true_all_front_clear, 1):.2f}%)"
    )
    print(
        "  Stop/resample when all front truly clear: "
        f"{cautious_when_all_front_clear} ({100.0 * cautious_when_all_front_clear / max(n_true_all_front_clear, 1):.2f}%)"
    )
    print(
        "  When front truly blocked -> turn/stop-resample/forward rates: "
        f"{100.0 * turn_when_front_blocked / max(n_front_blocked, 1):.2f}% / "
        f"{100.0 * stop_or_resample_when_front_blocked / max(n_front_blocked, 1):.2f}% / "
        f"{100.0 * forward_when_front_blocked / max(n_front_blocked, 1):.2f}%"
    )

    print("\nInterpretation:")
    print(f"  {result['interpretation']}")

    print(f"\nSaved v2 scene summary: {OUT_SCENE_CSV}")
    print(f"Saved v2 analysis JSON: {OUT_JSON}")
    print(f"Saved v2 distribution plot: {OUT_PNG}")


if __name__ == "__main__":
    main()

