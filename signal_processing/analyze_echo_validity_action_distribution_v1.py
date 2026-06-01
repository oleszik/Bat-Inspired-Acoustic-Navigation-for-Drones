"""
Analyze scene-level recommended actions after echo-validity gate integration.

This script checks whether action outputs are usable for navigation by comparing
recommended actions against true front-sector obstacle conditions.
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
RESULTS_JSON = DATASET_ROOT / "multisector_echo_validity_results.json"

OUT_JSON = DATASET_ROOT / "echo_validity_action_analysis.json"
OUT_PNG = DATASET_ROOT / "echo_validity_action_distribution.png"
OUT_COND_PNG = DATASET_ROOT / "echo_validity_action_by_front_condition.png"

FRONT_SECTORS = ["front_left", "front", "front_right"]
ACTION_ORDER = [
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SLOW_DOWN_AND_RESAMPLE",
    "STOP_OR_REVERSE",
]


def action_stats(series: pd.Series) -> dict[str, dict[str, float]]:
    total = int(series.shape[0])
    counts = series.value_counts().to_dict()
    out: dict[str, dict[str, float]] = {}
    # Report canonical actions first, then extras.
    ordered = ACTION_ORDER + [a for a in sorted(counts.keys()) if a not in ACTION_ORDER]
    for action in ordered:
        c = int(counts.get(action, 0))
        out[action] = {
            "count": c,
            "percent": float((100.0 * c / total) if total > 0 else 0.0),
        }
    return out


def subset_action_stats(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, object]:
    sub = df.loc[mask].copy()
    return {
        "subset_name": name,
        "num_scenes": int(sub.shape[0]),
        "action_distribution": action_stats(sub["recommended_action"]) if len(sub) else {},
    }


def short_interpretation(
    move_forward_any_front_obstacle_rate: float,
    clear_all_front_slow_or_stop_rate: float,
    overall_action_pct: dict[str, dict[str, float]],
) -> str:
    move_forward_pct = overall_action_pct.get("MOVE_FORWARD", {}).get("percent", 0.0)
    turn_left_pct = overall_action_pct.get("TURN_LEFT", {}).get("percent", 0.0)
    turn_right_pct = overall_action_pct.get("TURN_RIGHT", {}).get("percent", 0.0)
    slow_pct = overall_action_pct.get("SLOW_DOWN_AND_RESAMPLE", {}).get("percent", 0.0)
    stop_pct = overall_action_pct.get("STOP_OR_REVERSE", {}).get("percent", 0.0)

    safety_line = (
        "Policy looks unsafe because MOVE_FORWARD is used with true front obstacles."
        if move_forward_any_front_obstacle_rate > 0.005
        else "Policy is safety-oriented: MOVE_FORWARD with true front obstacles is near zero."
    )
    caution_line = (
        "Policy is still too cautious in clear space; slow/stop remains high when front sectors are truly clear."
        if clear_all_front_slow_or_stop_rate > 0.50
        else "Policy caution level is moderate; clear-space slow/stop is not dominant."
    )
    action_line = (
        f"Action diversity exists: MOVE_FORWARD {move_forward_pct:.2f}%, "
        f"TURN_LEFT {turn_left_pct:.2f}%, TURN_RIGHT {turn_right_pct:.2f}%, "
        f"SLOW_DOWN_AND_RESAMPLE {slow_pct:.2f}%, STOP_OR_REVERSE {stop_pct:.2f}%."
    )
    next_step = (
        "Next improvement: reduce UNCERTAIN in true-clear front sectors "
        "using temporal clear confirmation and stricter false-obstacle suppression."
    )
    return f"{safety_line} {caution_line} {action_line} {next_step}"


def main() -> None:
    for p in [SCENE_CSV, PRED_CSV, RESULTS_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    scene = pd.read_csv(SCENE_CSV)
    pred = pd.read_csv(PRED_CSV)
    _ = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    needed_scene = {"sample_id", "recommended_action"}
    if not needed_scene.issubset(scene.columns):
        raise ValueError(f"Scene CSV missing required columns: {needed_scene - set(scene.columns)}")
    needed_pred = {"sample_id", "sector", "true_has_obstacle"}
    if not needed_pred.issubset(pred.columns):
        raise ValueError(f"Prediction CSV missing required columns: {needed_pred - set(pred.columns)}")

    # Build true sector conditions per sample_id.
    truth_pivot = pred.pivot(index="sample_id", columns="sector", values="true_has_obstacle")
    for sec in ["left", "front_left", "front", "front_right", "right"]:
        if sec not in truth_pivot.columns:
            truth_pivot[sec] = 0
    truth_pivot = truth_pivot.fillna(0).astype(int)
    truth_pivot["true_front_obstacle"] = truth_pivot["front"] == 1
    truth_pivot["true_front_clear"] = truth_pivot["front"] == 0
    truth_pivot["true_any_front_obstacle"] = truth_pivot[FRONT_SECTORS].max(axis=1) == 1
    truth_pivot["true_all_front_clear"] = truth_pivot[FRONT_SECTORS].sum(axis=1) == 0

    merged = scene.merge(
        truth_pivot[
            [
                "left",
                "front_left",
                "front",
                "front_right",
                "right",
                "true_front_obstacle",
                "true_front_clear",
                "true_any_front_obstacle",
                "true_all_front_clear",
            ]
        ].reset_index(),
        on="sample_id",
        how="left",
    )

    if merged[["true_front_obstacle", "true_any_front_obstacle"]].isna().any().any():
        raise ValueError("Some sample_id rows in scene summary are missing true front-sector labels.")

    # Action distribution overall.
    overall_actions = action_stats(merged["recommended_action"])
    action_counts = {k: int(v["count"]) for k, v in overall_actions.items()}
    action_percentages = {k: float(v["percent"]) for k, v in overall_actions.items()}

    # Analyze actions by true front conditions.
    by_condition = {
        "true_front_clear": subset_action_stats(merged, merged["true_front_clear"], "true_front_clear"),
        "true_front_obstacle": subset_action_stats(merged, merged["true_front_obstacle"], "true_front_obstacle"),
        "true_all_front_clear": subset_action_stats(merged, merged["true_all_front_clear"], "true_all_front_clear"),
        "true_any_front_obstacle": subset_action_stats(
            merged, merged["true_any_front_obstacle"], "true_any_front_obstacle"
        ),
    }

    # Safety checks requested.
    n_true_front_obstacle = int(merged["true_front_obstacle"].sum())
    n_true_any_front_obstacle = int(merged["true_any_front_obstacle"].sum())
    n_true_all_front_clear = int(merged["true_all_front_clear"].sum())

    mf_when_front_obstacle = int(
        ((merged["recommended_action"] == "MOVE_FORWARD") & merged["true_front_obstacle"]).sum()
    )
    mf_when_any_front_obstacle = int(
        ((merged["recommended_action"] == "MOVE_FORWARD") & merged["true_any_front_obstacle"]).sum()
    )
    slow_or_stop_when_all_front_clear = int(
        (
            merged["recommended_action"].isin(["SLOW_DOWN_AND_RESAMPLE", "STOP_OR_REVERSE"])
            & merged["true_all_front_clear"]
        ).sum()
    )
    slow_when_all_front_clear = int(
        ((merged["recommended_action"] == "SLOW_DOWN_AND_RESAMPLE") & merged["true_all_front_clear"]).sum()
    )
    stop_when_all_front_clear = int(
        ((merged["recommended_action"] == "STOP_OR_REVERSE") & merged["true_all_front_clear"]).sum()
    )
    turn_left_when_left_blocked = int(
        ((merged["recommended_action"] == "TURN_LEFT") & (merged["left"] == 1)).sum()
    )
    n_true_left_blocked = int((merged["left"] == 1).sum())
    turn_right_when_right_blocked = int(
        ((merged["recommended_action"] == "TURN_RIGHT") & (merged["right"] == 1)).sum()
    )
    n_true_right_blocked = int((merged["right"] == 1).sum())

    safety = {
        "move_forward_when_true_front_obstacle": {
            "count": mf_when_front_obstacle,
            "rate": float(mf_when_front_obstacle / max(n_true_front_obstacle, 1)),
        },
        "move_forward_when_any_true_front_obstacle": {
            "count": mf_when_any_front_obstacle,
            "rate": float(mf_when_any_front_obstacle / max(n_true_any_front_obstacle, 1)),
        },
        "slow_or_stop_when_all_front_clear": {
            "count": slow_or_stop_when_all_front_clear,
            "rate": float(slow_or_stop_when_all_front_clear / max(n_true_all_front_clear, 1)),
        },
        "slow_down_when_all_front_clear": {
            "count": slow_when_all_front_clear,
            "rate": float(slow_when_all_front_clear / max(n_true_all_front_clear, 1)),
        },
        "stop_or_reverse_when_all_front_clear": {
            "count": stop_when_all_front_clear,
            "rate": float(stop_when_all_front_clear / max(n_true_all_front_clear, 1)),
        },
    }

    interpretation = short_interpretation(
        move_forward_any_front_obstacle_rate=safety["move_forward_when_any_true_front_obstacle"]["rate"],
        clear_all_front_slow_or_stop_rate=safety["slow_or_stop_when_all_front_clear"]["rate"],
        overall_action_pct=overall_actions,
    )

    result = {
        "total_scenes": int(merged.shape[0]),
        "action_counts": action_counts,
        "action_percentages": action_percentages,
        "move_forward_with_true_front_obstacle_count": int(mf_when_front_obstacle),
        "move_forward_with_true_front_obstacle_rate": float(mf_when_front_obstacle / max(n_true_front_obstacle, 1)),
        "move_forward_with_any_true_front_group_obstacle_count": int(mf_when_any_front_obstacle),
        "move_forward_with_any_true_front_group_obstacle_rate": float(
            mf_when_any_front_obstacle / max(n_true_any_front_obstacle, 1)
        ),
        "cautious_when_all_front_clear_count": int(slow_or_stop_when_all_front_clear),
        "cautious_when_all_front_clear_rate": float(
            slow_or_stop_when_all_front_clear / max(n_true_all_front_clear, 1)
        ),
        "turn_left_when_left_blocked_count": int(turn_left_when_left_blocked),
        "turn_left_when_left_blocked_rate": float(turn_left_when_left_blocked / max(n_true_left_blocked, 1)),
        "turn_right_when_right_blocked_count": int(turn_right_when_right_blocked),
        "turn_right_when_right_blocked_rate": float(turn_right_when_right_blocked / max(n_true_right_blocked, 1)),
        "safety_checks": safety,
        "action_distribution_by_true_front_condition": by_condition,
        "interpretation": interpretation,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Plot action distribution (percentages).
    action_labels = ACTION_ORDER + [a for a in overall_actions.keys() if a not in ACTION_ORDER]
    action_labels = list(dict.fromkeys(action_labels))
    pct_vals = [overall_actions.get(a, {}).get("percent", 0.0) for a in action_labels]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=140)
    bars = ax.bar(action_labels, pct_vals, color="tab:blue")
    ax.set_ylabel("Percentage of scenes (%)")
    ax.set_title("Scene-Level Recommended Action Distribution (Echo-Validity Gated)")
    ax.tick_params(axis="x", rotation=20)
    for b, p in zip(bars, pct_vals):
        ax.text(b.get_x() + b.get_width() / 2.0, p + 0.5, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)

    # Optional grouped bar plot: action distribution by front condition.
    clear_mask = merged["true_all_front_clear"]
    blocked_mask = merged["true_any_front_obstacle"]
    clear_actions = merged.loc[clear_mask, "recommended_action"]
    blocked_actions = merged.loc[blocked_mask, "recommended_action"]
    clear_counts = clear_actions.value_counts()
    blocked_counts = blocked_actions.value_counts()

    labels = ACTION_ORDER + sorted(
        set(clear_counts.index.tolist() + blocked_counts.index.tolist()) - set(ACTION_ORDER)
    )
    labels = list(dict.fromkeys(labels))
    clear_pct = [
        100.0 * float(clear_counts.get(a, 0)) / max(int(clear_mask.sum()), 1)
        for a in labels
    ]
    blocked_pct = [
        100.0 * float(blocked_counts.get(a, 0)) / max(int(blocked_mask.sum()), 1)
        for a in labels
    ]

    x = np.arange(len(labels), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.2, 4.9), dpi=140)
    ax.bar(x - w / 2.0, clear_pct, width=w, label="All front sectors clear")
    ax.bar(x + w / 2.0, blocked_pct, width=w, label="Any front sector blocked")
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylabel("Percentage within condition (%)")
    ax.set_title("Action Distribution by True Front Condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_COND_PNG)
    plt.close(fig)

    print("\nAction distribution:")
    for a, vals in overall_actions.items():
        print(f"  {a}: {vals['count']} ({vals['percent']:.2f}%)")

    print("\nActions by true front-sector condition:")
    for cond_name, cond_result in by_condition.items():
        print(f"  [{cond_name}] scenes={cond_result['num_scenes']}")
        for a, vals in cond_result["action_distribution"].items():
            print(f"    {a}: {vals['count']} ({vals['percent']:.2f}%)")

    print("\nSafety checks:")
    print(
        "  MOVE_FORWARD when true front obstacle: "
        f"{safety['move_forward_when_true_front_obstacle']['count']} "
        f"({100.0 * safety['move_forward_when_true_front_obstacle']['rate']:.2f}%)"
    )
    print(
        "  MOVE_FORWARD when any true front obstacle: "
        f"{safety['move_forward_when_any_true_front_obstacle']['count']} "
        f"({100.0 * safety['move_forward_when_any_true_front_obstacle']['rate']:.2f}%)"
    )
    print(
        "  SLOW/STOP when all front sectors truly clear: "
        f"{safety['slow_or_stop_when_all_front_clear']['count']} "
        f"({100.0 * safety['slow_or_stop_when_all_front_clear']['rate']:.2f}%)"
    )
    print(
        "  TURN_LEFT when true left sector has obstacle: "
        f"{turn_left_when_left_blocked} "
        f"({100.0 * turn_left_when_left_blocked / max(n_true_left_blocked, 1):.2f}%)"
    )
    print(
        "  TURN_RIGHT when true right sector has obstacle: "
        f"{turn_right_when_right_blocked} "
        f"({100.0 * turn_right_when_right_blocked / max(n_true_right_blocked, 1):.2f}%)"
    )

    print("\nInterpretation:")
    print(f"  {interpretation}")

    print(f"\nSaved JSON: {OUT_JSON}")
    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved grouped plot: {OUT_COND_PNG}")


if __name__ == "__main__":
    main()
