import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT = Path("runs/phase2_mapper_guided_navigation_v3/mapper_guided_navigation_results.json")
DEFAULT_OUTPUT_DIR = Path("runs/phase2_mapper_guided_navigation_v3_failure_analysis")

ACCEPTED_V3_REFERENCE = {
    "success_rate": 0.7817,
    "exploration_coverage": 0.6608,
    "collision_rate": 0.0,
    "fake_doorway_approach_rate": 0.0,
    "doorway_crossing_success_rate": 0.9283,
    "timeout_rate": 0.0617,
    "final_map_accuracy": 0.9183,
    "final_wall_f1": 0.3386,
    "final_doorway_f1": 0.1532,
}

LOW_COVERAGE_THRESHOLD = 0.65
BAD_MAP_ACCURACY_THRESHOLD = 0.90
GOOD_COVERAGE_THRESHOLD = 0.65
GOOD_MAP_ACCURACY_THRESHOLD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine accepted Phase 2D v3 failure modes from saved aggregate results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def rate_to_count(rate: float, total: int) -> int:
    return int(round(max(0.0, min(1.0, rate)) * total))


def action_rates(summary: Dict[str, Any]) -> Dict[str, float]:
    dist = summary.get("action_distribution", {})
    if not isinstance(dist, dict):
        dist = {}
    return {
        "probe_forward_rate": as_float(dist.get("PROBE_FORWARD", {}).get("rate") if isinstance(dist.get("PROBE_FORWARD"), dict) else 0.0),
        "move_forward_slow_rate": as_float(dist.get("MOVE_FORWARD_SLOW", {}).get("rate") if isinstance(dist.get("MOVE_FORWARD_SLOW"), dict) else 0.0),
        "move_forward_fast_rate": as_float(dist.get("MOVE_FORWARD_FAST", {}).get("rate") if isinstance(dist.get("MOVE_FORWARD_FAST"), dict) else 0.0),
        "turn_left_rate": as_float(dist.get("TURN_LEFT", {}).get("rate") if isinstance(dist.get("TURN_LEFT"), dict) else 0.0),
        "turn_right_rate": as_float(dist.get("TURN_RIGHT", {}).get("rate") if isinstance(dist.get("TURN_RIGHT"), dict) else 0.0),
        "turn_rate_from_actions": (
            as_float(dist.get("TURN_LEFT", {}).get("rate") if isinstance(dist.get("TURN_LEFT"), dict) else 0.0)
            + as_float(dist.get("TURN_RIGHT", {}).get("rate") if isinstance(dist.get("TURN_RIGHT"), dict) else 0.0)
        ),
        "slow_down_resample_rate": as_float(
            dist.get("SLOW_DOWN_AND_RESAMPLE", {}).get("rate") if isinstance(dist.get("SLOW_DOWN_AND_RESAMPLE"), dict) else 0.0
        ),
        "stop_or_reverse_rate": as_float(dist.get("STOP_OR_REVERSE", {}).get("rate") if isinstance(dist.get("STOP_OR_REVERSE"), dict) else 0.0),
    }


def core_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    actions = action_rates(summary)
    row = {
        "success_rate": as_float(summary.get("success_rate")),
        "timeout_rate": as_float(summary.get("timeout_rate")),
        "collision_rate": as_float(summary.get("collision_rate")),
        "fake_doorway_approach_rate": as_float(summary.get("fake_doorway_approach_rate")),
        "doorway_crossing_success_rate": as_float(summary.get("doorway_crossing_success_rate")),
        "mean_coverage": as_float(summary.get("exploration_coverage")),
        "mean_final_map_accuracy": as_float(summary.get("final_map_accuracy")),
        "mean_wall_f1": as_float(summary.get("final_wall_f1")),
        "mean_doorway_f1": as_float(summary.get("final_doorway_f1")),
        "mean_episode_length": as_float(summary.get("mean_steps")),
        "mean_path_length": as_float(summary.get("mean_path_length")),
        "coverage_gain_per_100_steps": as_float(summary.get("coverage_gain_per_100_steps")),
        "probe_action_ratio": as_float(summary.get("probe_action_ratio", actions["probe_forward_rate"])),
        "move_forward_ratio": as_float(summary.get("move_forward_ratio")),
        "turn_ratio": as_float(summary.get("turn_ratio", actions["turn_rate_from_actions"])),
        "failure_reason_counts": json.dumps(summary.get("failure_reason_counts", {}), sort_keys=True),
    }
    row.update(actions)
    return row


def iter_map_difficulty(data: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for difficulty, diff_payload in data.get("by_difficulty", {}).items():
        if not isinstance(diff_payload, dict):
            continue
        maps = diff_payload.get("maps", {})
        if not isinstance(maps, dict):
            continue
        for map_name, summary in maps.items():
            if isinstance(summary, dict):
                yield str(map_name), str(difficulty), summary


def map_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for map_name, _, summary in iter_map_difficulty(data):
        grouped.setdefault(map_name, []).append(summary)
    rows = []
    for map_name, summaries in sorted(grouped.items()):
        row = {"map": map_name}
        row.update(mean_metrics(summaries))
        rows.append(row)
    return rows


def difficulty_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for difficulty, diff_payload in data.get("by_difficulty", {}).items():
        if isinstance(diff_payload, dict) and isinstance(diff_payload.get("aggregate"), dict):
            row = {"difficulty": difficulty}
            row.update(core_metrics(diff_payload["aggregate"]))
            rows.append(row)
    return rows


def map_difficulty_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for map_name, difficulty, summary in iter_map_difficulty(data):
        row = {"map": map_name, "difficulty": difficulty}
        row.update(core_metrics(summary))
        row.update(classify_aggregate(summary))
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["map"]), str(r["difficulty"])))


def mean_metrics(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        return core_metrics({})
    metric_rows = [core_metrics(summary) for summary in summaries]
    row: Dict[str, Any] = {}
    for key in metric_rows[0].keys():
        if key == "failure_reason_counts":
            merged: Dict[str, int] = {}
            for summary in summaries:
                counts = summary.get("failure_reason_counts", {})
                if isinstance(counts, dict):
                    for reason, count in counts.items():
                        merged[str(reason)] = merged.get(str(reason), 0) + as_int(count)
            row[key] = json.dumps(merged, sort_keys=True)
        else:
            values = [as_float(metric_row.get(key)) for metric_row in metric_rows]
            row[key] = sum(values) / max(1, len(values))
    return row


def classify_aggregate(summary: Dict[str, Any]) -> Dict[str, Any]:
    success_rate = as_float(summary.get("success_rate"))
    timeout_rate = as_float(summary.get("timeout_rate"))
    coverage = as_float(summary.get("exploration_coverage"))
    map_accuracy = as_float(summary.get("final_map_accuracy"))
    doorway_cross = as_float(summary.get("doorway_crossing_success_rate"))
    fake_door = as_float(summary.get("fake_doorway_approach_rate"))
    collision = as_float(summary.get("collision_rate"))

    low_coverage = coverage < LOW_COVERAGE_THRESHOLD
    bad_map_quality = map_accuracy < BAD_MAP_ACCURACY_THRESHOLD
    doorway_related = doorway_cross < ACCEPTED_V3_REFERENCE["doorway_crossing_success_rate"] - 0.03 or fake_door > 0.0

    timeout_class = "not_timeout_dominant"
    if timeout_rate > 0.0:
        if low_coverage:
            timeout_class = "timeout_low_coverage"
        elif coverage >= GOOD_COVERAGE_THRESHOLD and bad_map_quality:
            timeout_class = "timeout_good_coverage_bad_map"
        elif coverage >= GOOD_COVERAGE_THRESHOLD and map_accuracy >= GOOD_MAP_ACCURACY_THRESHOLD:
            timeout_class = "timeout_good_map_no_success"
        elif doorway_related:
            timeout_class = "timeout_doorway_related"
        else:
            timeout_class = "timeout_unknown"

    dominant_failure = "other_failure"
    if collision > 0.0:
        dominant_failure = "collision"
    elif fake_door > 0.0:
        dominant_failure = "fake_doorway_approach"
    elif doorway_related:
        dominant_failure = "doorway_crossing_failure"
    elif timeout_rate >= max(0.05, 1.0 - success_rate):
        dominant_failure = timeout_class
    elif low_coverage:
        dominant_failure = "low_coverage_failure"
    elif bad_map_quality:
        dominant_failure = "bad_map_quality_failure"

    return {
        "low_coverage_flag": low_coverage,
        "bad_map_quality_flag": bad_map_quality,
        "doorway_related_flag": doorway_related,
        "timeout_classification": timeout_class,
        "dominant_failure_mode": dominant_failure,
        "failure_rate": max(0.0, 1.0 - success_rate),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], preferred_fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    if preferred_fields:
        fields.extend([field for field in preferred_fields if any(field in row for row in rows)])
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def overall_failure_breakdown(data: Dict[str, Any]) -> Dict[str, Any]:
    overall = data.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    config = data.get("config", {})
    if not isinstance(config, dict):
        config = {}
    total_episodes = (
        as_int(config.get("episodes_per_map"), 0)
        * len(config.get("maps", []) if isinstance(config.get("maps"), list) else [])
        * len(config.get("difficulties", []) if isinstance(config.get("difficulties"), list) else [])
    )
    if total_episodes <= 0:
        failure_counts = overall.get("failure_reason_counts", {})
        if isinstance(failure_counts, dict):
            total_episodes = sum(as_int(value) for value in failure_counts.values())
    total_episodes = max(1, total_episodes)

    success_count = rate_to_count(as_float(overall.get("success_rate")), total_episodes)
    timeout_count = rate_to_count(as_float(overall.get("timeout_rate")), total_episodes)
    collision_count = rate_to_count(as_float(overall.get("collision_rate")), total_episodes)
    fake_door_count = rate_to_count(as_float(overall.get("fake_doorway_approach_rate")), total_episodes)
    doorway_failure_count = rate_to_count(1.0 - as_float(overall.get("doorway_crossing_success_rate")), total_episodes)
    low_coverage_count = rate_to_count(1.0 if as_float(overall.get("exploration_coverage")) < LOW_COVERAGE_THRESHOLD else 0.0, total_episodes)
    bad_map_count = rate_to_count(1.0 if as_float(overall.get("final_map_accuracy")) < BAD_MAP_ACCURACY_THRESHOLD else 0.0, total_episodes)
    known_failure_count = max(timeout_count, collision_count, fake_door_count, doorway_failure_count)
    other_failure_count = max(0, total_episodes - success_count - known_failure_count)

    rows = [
        ("success", success_count),
        ("timeout", timeout_count),
        ("collision", collision_count),
        ("fake_doorway_approach", fake_door_count),
        ("doorway_crossing_failure", doorway_failure_count),
        ("low_coverage_failure_aggregate_flag", low_coverage_count),
        ("bad_map_quality_failure_aggregate_flag", bad_map_count),
        ("other_failure", other_failure_count),
    ]
    return {
        "total_episodes_estimated": total_episodes,
        "breakdown": {
            name: {"count": int(count), "rate": float(count / total_episodes)}
            for name, count in rows
        },
        "note": "Counts are reconstructed from aggregate rates because the v3 results file does not contain per-episode records.",
    }


def reference_deltas(overall: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    deltas: Dict[str, Dict[str, float]] = {}
    for key, reference_value in ACCEPTED_V3_REFERENCE.items():
        observed = as_float(overall.get(key))
        deltas[key] = {
            "input_value": observed,
            "accepted_reference_value": reference_value,
            "delta": observed - reference_value,
        }
    return deltas


def rows_matching(rows: List[Dict[str, Any]], predicate: Any) -> List[Dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def worst_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        "lowest_success": min(rows, key=lambda row: as_float(row.get("success_rate"), 1.0)),
        "highest_timeout": max(rows, key=lambda row: as_float(row.get("timeout_rate"))),
        "lowest_coverage": min(rows, key=lambda row: as_float(row.get("mean_coverage"), 1.0)),
        "lowest_final_map_accuracy": min(rows, key=lambda row: as_float(row.get("mean_final_map_accuracy"), 1.0)),
    }


def timeout_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    timeout_rows = rows_matching(rows, lambda row: as_float(row.get("timeout_rate")) > 0.0)
    if not timeout_rows:
        return {"timeout_group_count": 0, "dominant_map_difficulty": None, "classification_counts": {}}
    dominant = max(timeout_rows, key=lambda row: as_float(row.get("timeout_rate")))
    classification_counts: Dict[str, int] = {}
    weighted = {
        "mean_coverage_at_timeout": 0.0,
        "mean_map_accuracy_at_timeout": 0.0,
        "mean_wall_f1_at_timeout": 0.0,
        "mean_doorway_f1_at_timeout": 0.0,
        "mean_episode_length_at_timeout": 0.0,
    }
    total_weight = 0.0
    for row in timeout_rows:
        weight = max(0.0001, as_float(row.get("timeout_rate")))
        total_weight += weight
        classification = str(row.get("timeout_classification", "timeout_unknown"))
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
        weighted["mean_coverage_at_timeout"] += weight * as_float(row.get("mean_coverage"))
        weighted["mean_map_accuracy_at_timeout"] += weight * as_float(row.get("mean_final_map_accuracy"))
        weighted["mean_wall_f1_at_timeout"] += weight * as_float(row.get("mean_wall_f1"))
        weighted["mean_doorway_f1_at_timeout"] += weight * as_float(row.get("mean_doorway_f1"))
        weighted["mean_episode_length_at_timeout"] += weight * as_float(row.get("mean_episode_length"))
    for key in weighted:
        weighted[key] /= max(1e-8, total_weight)
    return {
        "timeout_group_count": len(timeout_rows),
        "dominant_map_difficulty": {
            "map": dominant.get("map"),
            "difficulty": dominant.get("difficulty"),
            "timeout_rate": dominant.get("timeout_rate"),
            "coverage": dominant.get("mean_coverage"),
            "final_map_accuracy": dominant.get("mean_final_map_accuracy"),
            "classification": dominant.get("timeout_classification"),
        },
        "classification_counts": classification_counts,
        **weighted,
    }


def success_vs_failure_rows(data: Dict[str, Any], md_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    successful_groups = rows_matching(md_rows, lambda row: as_float(row.get("success_rate")) >= 0.80)
    failed_groups = rows_matching(md_rows, lambda row: as_float(row.get("success_rate")) < 0.80)
    rows = []
    for label, group in [("high_success_groups", successful_groups), ("failure_prone_groups", failed_groups)]:
        row = {"group": label, "map_difficulty_group_count": len(group)}
        row.update(mean_rows(group))
        rows.append(row)
    overall = data.get("overall", {})
    if isinstance(overall, dict):
        row = {"group": "overall", "map_difficulty_group_count": len(md_rows)}
        row.update(core_metrics(overall))
        rows.append(row)
    return rows


def mean_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = [
        "success_rate",
        "timeout_rate",
        "collision_rate",
        "fake_doorway_approach_rate",
        "doorway_crossing_success_rate",
        "mean_coverage",
        "mean_final_map_accuracy",
        "mean_wall_f1",
        "mean_doorway_f1",
        "mean_episode_length",
        "probe_forward_rate",
        "move_forward_slow_rate",
        "turn_rate_from_actions",
        "probe_action_ratio",
        "move_forward_ratio",
        "turn_ratio",
    ]
    return {key: sum(as_float(row.get(key)) for row in rows) / len(rows) for key in numeric_keys}


def recommendation(summary: Dict[str, Any]) -> str:
    worst = summary["worst_map_difficulty"]
    timeout = summary["timeout_analysis"]
    timeout_classes = timeout.get("classification_counts", {})
    main_class = max(timeout_classes.items(), key=lambda item: item[1])[0] if timeout_classes else "unknown"
    worst_low_success = worst.get("lowest_success", {})
    worst_map = worst_low_success.get("map", "unknown")
    worst_difficulty = worst_low_success.get("difficulty", "unknown")

    if main_class == "timeout_low_coverage":
        return (
            f"Safest v3.3 direction: add a localized frontier-recovery patch only for low-coverage timeout states, "
            f"starting with {worst_map}/{worst_difficulty}. Preserve accepted v3 probing, map fusion, doorway suppression, "
            "and global planner structure."
        )
    if main_class == "timeout_good_coverage_bad_map":
        return (
            "Safest v3.3 direction: investigate map fusion or final thresholding on timeout groups with adequate coverage "
            "but weak map quality. Do not change planner behavior first."
        )
    if main_class == "timeout_good_map_no_success":
        return (
            "Safest v3.3 direction: inspect success/termination logic for episodes with good coverage and map quality. "
            "Planner changes are not justified by this evidence alone."
        )
    if main_class == "timeout_doorway_related":
        return (
            f"Safest v3.3 direction: make a doorway-specific target-selection patch for {worst_map}/{worst_difficulty}; "
            "avoid broad movement/probe changes."
        )
    return (
        "Safest v3.3 direction: keep accepted v3 unchanged and improve diagnostics or mapper evidence first; "
        "the aggregate failure evidence is broad or ambiguous."
    )


def markdown_table(rows: List[Dict[str, Any]], fields: List[str], limit: Optional[int] = None) -> str:
    selected = rows[:limit] if limit else rows
    if not selected:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in selected:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, summary: Dict[str, Any], by_map: List[Dict[str, Any]], by_diff: List[Dict[str, Any]], md_rows: List[Dict[str, Any]]) -> None:
    overall = summary["overall_metrics"]
    failure_breakdown = summary["overall_failure_breakdown"]["breakdown"]
    worst = summary["worst_map_difficulty"]
    timeout = summary["timeout_analysis"]
    rec = summary["recommendation"]

    sorted_worst = sorted(md_rows, key=lambda row: (as_float(row.get("success_rate")), -as_float(row.get("timeout_rate"))))
    lines = [
        "# Phase 2D Accepted v3 Failure Analysis",
        "",
        "## Scope",
        "",
        "- Input: `runs/phase2_mapper_guided_navigation_v3/mapper_guided_navigation_results.json`",
        "- The v3 results file contains aggregate summaries, not per-episode records; case CSVs are aggregate map/difficulty cases.",
        "- The input file metrics are compared against the accepted v3 headline metrics in `v3_failure_summary.json`.",
        "- Accepted v3 code and rejected experiment scripts are not modified by this analyzer.",
        "",
        "## Overall Metrics",
        "",
        markdown_table([overall], [
            "success_rate",
            "timeout_rate",
            "collision_rate",
            "fake_doorway_approach_rate",
            "doorway_crossing_success_rate",
            "mean_coverage",
            "mean_final_map_accuracy",
            "mean_wall_f1",
            "mean_doorway_f1",
        ]),
        "",
        "## Failure Breakdown",
        "",
        markdown_table(
            [{"failure_mode": key, **value} for key, value in failure_breakdown.items()],
            ["failure_mode", "count", "rate"],
        ),
        "",
        "## Worst Map/Difficulty Combinations",
        "",
        f"- Lowest success: `{worst.get('lowest_success', {}).get('map')}` / `{worst.get('lowest_success', {}).get('difficulty')}` "
        f"success `{as_float(worst.get('lowest_success', {}).get('success_rate')):.4f}`",
        f"- Highest timeout: `{worst.get('highest_timeout', {}).get('map')}` / `{worst.get('highest_timeout', {}).get('difficulty')}` "
        f"timeout `{as_float(worst.get('highest_timeout', {}).get('timeout_rate')):.4f}`",
        f"- Lowest coverage: `{worst.get('lowest_coverage', {}).get('map')}` / `{worst.get('lowest_coverage', {}).get('difficulty')}` "
        f"coverage `{as_float(worst.get('lowest_coverage', {}).get('mean_coverage')):.4f}`",
        f"- Lowest map accuracy: `{worst.get('lowest_final_map_accuracy', {}).get('map')}` / `{worst.get('lowest_final_map_accuracy', {}).get('difficulty')}` "
        f"map accuracy `{as_float(worst.get('lowest_final_map_accuracy', {}).get('mean_final_map_accuracy')):.4f}`",
        "",
        markdown_table(sorted_worst, [
            "map",
            "difficulty",
            "success_rate",
            "timeout_rate",
            "mean_coverage",
            "mean_final_map_accuracy",
            "dominant_failure_mode",
            "timeout_classification",
        ], limit=12),
        "",
        "## Per-Map Summary",
        "",
        markdown_table(by_map, ["map", "success_rate", "timeout_rate", "mean_coverage", "mean_final_map_accuracy", "mean_wall_f1", "mean_doorway_f1"]),
        "",
        "## Per-Difficulty Summary",
        "",
        markdown_table(by_diff, ["difficulty", "success_rate", "timeout_rate", "mean_coverage", "mean_final_map_accuracy", "mean_wall_f1", "mean_doorway_f1"]),
        "",
        "## Timeout Analysis",
        "",
        f"- Dominant timeout group: `{timeout.get('dominant_map_difficulty', {}).get('map')}` / `{timeout.get('dominant_map_difficulty', {}).get('difficulty')}`",
        f"- Dominant timeout classification: `{timeout.get('dominant_map_difficulty', {}).get('classification')}`",
        f"- Weighted timeout coverage: `{as_float(timeout.get('mean_coverage_at_timeout')):.4f}`",
        f"- Weighted timeout map accuracy: `{as_float(timeout.get('mean_map_accuracy_at_timeout')):.4f}`",
        f"- Timeout classification counts: `{json.dumps(timeout.get('classification_counts', {}), sort_keys=True)}`",
        "",
        "## Recommendation",
        "",
        rec,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_raw = data.get("overall", {})
    if not isinstance(overall_raw, dict):
        raise ValueError("Input JSON is missing an overall aggregate.")

    by_map = map_rows(data)
    by_difficulty = difficulty_rows(data)
    by_map_difficulty = map_difficulty_rows(data)
    timeout_cases = rows_matching(by_map_difficulty, lambda row: as_float(row.get("timeout_rate")) > 0.0)
    low_coverage_cases = rows_matching(by_map_difficulty, lambda row: bool(row.get("low_coverage_flag")))
    bad_map_quality_cases = rows_matching(by_map_difficulty, lambda row: bool(row.get("bad_map_quality_flag")))
    success_failure = success_vs_failure_rows(data, by_map_difficulty)

    summary: Dict[str, Any] = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "analysis_limitations": [
            "The accepted v3 results JSON stores aggregate by-difficulty and map/difficulty summaries only.",
            "Per-episode timeout, low-coverage, and bad-map-quality case files are therefore aggregate map/difficulty case files.",
        ],
        "accepted_v3_reference_metrics": ACCEPTED_V3_REFERENCE,
        "overall_metrics": core_metrics(overall_raw),
        "input_vs_accepted_reference_deltas": reference_deltas(overall_raw),
        "overall_failure_breakdown": overall_failure_breakdown(data),
        "worst_map_difficulty": worst_rows(by_map_difficulty),
        "timeout_analysis": timeout_analysis(by_map_difficulty),
        "map_difficulty_group_count": len(by_map_difficulty),
        "timeout_case_group_count": len(timeout_cases),
        "low_coverage_case_group_count": len(low_coverage_cases),
        "bad_map_quality_case_group_count": len(bad_map_quality_cases),
    }
    summary["recommendation"] = recommendation(summary)

    preferred = [
        "map",
        "difficulty",
        "success_rate",
        "timeout_rate",
        "collision_rate",
        "fake_doorway_approach_rate",
        "doorway_crossing_success_rate",
        "mean_coverage",
        "mean_final_map_accuracy",
        "mean_wall_f1",
        "mean_doorway_f1",
        "mean_episode_length",
        "probe_action_ratio",
        "move_forward_ratio",
        "turn_ratio",
        "dominant_failure_mode",
        "timeout_classification",
    ]
    write_csv(args.output_dir / "v3_failure_by_map.csv", by_map, ["map", *preferred[2:]])
    write_csv(args.output_dir / "v3_failure_by_difficulty.csv", by_difficulty, ["difficulty", *preferred[2:]])
    write_csv(args.output_dir / "v3_failure_by_map_difficulty.csv", by_map_difficulty, preferred)
    write_csv(args.output_dir / "v3_timeout_cases.csv", timeout_cases, preferred)
    write_csv(args.output_dir / "v3_low_coverage_cases.csv", low_coverage_cases, preferred)
    write_csv(args.output_dir / "v3_bad_map_quality_cases.csv", bad_map_quality_cases, preferred)
    write_csv(args.output_dir / "v3_success_vs_failure_comparison.csv", success_failure)

    (args.output_dir / "v3_failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "v3_failure_summary.md", summary, by_map, by_difficulty, by_map_difficulty)

    worst = summary["worst_map_difficulty"]
    print(f"Saved analysis to: {args.output_dir}")
    print(
        "Worst success group: "
        f"{worst.get('lowest_success', {}).get('map')}/{worst.get('lowest_success', {}).get('difficulty')} "
        f"success={as_float(worst.get('lowest_success', {}).get('success_rate')):.4f}"
    )
    print(
        "Dominant timeout group: "
        f"{summary['timeout_analysis'].get('dominant_map_difficulty', {}).get('map')}/"
        f"{summary['timeout_analysis'].get('dominant_map_difficulty', {}).get('difficulty')} "
        f"class={summary['timeout_analysis'].get('dominant_map_difficulty', {}).get('classification')}"
    )
    print(f"Recommendation: {summary['recommendation']}")


if __name__ == "__main__":
    main()
