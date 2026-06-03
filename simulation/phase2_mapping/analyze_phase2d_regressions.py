from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


RESULT_PATHS = {
    "v3": Path("runs/phase2_mapper_guided_navigation_v3/mapper_guided_navigation_results.json"),
    "v4": Path("runs/phase2_mapper_guided_navigation_v4/mapper_guided_navigation_results.json"),
    "v31": Path("runs/phase2_mapper_guided_navigation_v31/mapper_guided_navigation_results.json"),
}
OUT_DIR = Path("runs/phase2_mapper_guided_navigation_regression_analysis")

OVERALL_METRICS = [
    "success_rate",
    "exploration_coverage",
    "collision_rate",
    "fake_doorway_approach_rate",
    "doorway_crossing_success_rate",
    "timeout_rate",
    "final_map_accuracy",
    "final_wall_f1",
    "final_doorway_f1",
]
MAP_METRICS = [
    "success_rate",
    "exploration_coverage",
    "timeout_rate",
    "final_map_accuracy",
    "final_wall_f1",
    "final_doorway_f1",
]
ACTION_NAMES = [
    "MOVE_FORWARD_FAST",
    "MOVE_FORWARD_SLOW",
    "PROBE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SLOW_DOWN_AND_RESAMPLE",
    "STOP_OR_REVERSE",
]


def load_results() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    missing = [str(path) for path in RESULT_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result files: {missing}")
    for version, path in RESULT_PATHS.items():
        results[version] = json.loads(path.read_text(encoding="utf-8"))
    return results


def fnum(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def get_metric(agg: Dict[str, Any], metric: str) -> float:
    return fnum(agg.get(metric))


def action_rate(agg: Dict[str, Any], action: str) -> float:
    dist = agg.get("action_distribution", {})
    if isinstance(dist, dict):
        row = dist.get(action, {})
        if isinstance(row, dict):
            return fnum(row.get("rate"))
    return 0.0


def turn_rate(agg: Dict[str, Any]) -> float:
    if "turn_ratio" in agg:
        return get_metric(agg, "turn_ratio")
    return action_rate(agg, "TURN_LEFT") + action_rate(agg, "TURN_RIGHT")


def failure_count(agg: Dict[str, Any], reason: str) -> int:
    counts = agg.get("failure_reason_counts", {})
    if isinstance(counts, dict):
        return int(counts.get(reason, 0))
    return 0


def by_map_averages(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    maps: Dict[str, List[Dict[str, Any]]] = {}
    for diff_data in data.get("by_difficulty", {}).values():
        for map_name, agg in diff_data.get("maps", {}).items():
            maps.setdefault(map_name, []).append(agg)
    out: Dict[str, Dict[str, float]] = {}
    for map_name, aggs in maps.items():
        row: Dict[str, float] = {}
        for metric in MAP_METRICS:
            row[metric] = mean(get_metric(agg, metric) for agg in aggs)
        row["probe_forward_rate"] = mean(action_rate(agg, "PROBE_FORWARD") for agg in aggs)
        row["move_forward_slow_rate"] = mean(action_rate(agg, "MOVE_FORWARD_SLOW") for agg in aggs)
        row["turn_rate"] = mean(turn_rate(agg) for agg in aggs)
        row["stop_or_reverse_rate"] = mean(action_rate(agg, "STOP_OR_REVERSE") for agg in aggs)
        row["mean_steps"] = mean(get_metric(agg, "mean_steps") for agg in aggs)
        row["timeout_failures"] = sum(failure_count(agg, "timeout") for agg in aggs)
        out[map_name] = row
    return out


def iter_map_difficulty(data: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for difficulty, diff_data in data.get("by_difficulty", {}).items():
        for map_name, agg in diff_data.get("maps", {}).items():
            yield str(map_name), str(difficulty), agg


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def delta_rows(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    v3_by_combo = {(m, d): agg for m, d, agg in iter_map_difficulty(results["v3"])}
    for version in ["v4", "v31"]:
        for map_name, difficulty, agg in iter_map_difficulty(results[version]):
            base = v3_by_combo[(map_name, difficulty)]
            row: Dict[str, Any] = {
                "version": version,
                "map": map_name,
                "difficulty": difficulty,
            }
            for metric in MAP_METRICS:
                row[f"{metric}_v3"] = get_metric(base, metric)
                row[f"{metric}_{version}"] = get_metric(agg, metric)
                row[f"delta_{metric}"] = get_metric(agg, metric) - get_metric(base, metric)
            row["probe_forward_rate_v3"] = action_rate(base, "PROBE_FORWARD")
            row[f"probe_forward_rate_{version}"] = action_rate(agg, "PROBE_FORWARD")
            row["delta_probe_forward_rate"] = row[f"probe_forward_rate_{version}"] - row["probe_forward_rate_v3"]
            row["move_forward_slow_rate_v3"] = action_rate(base, "MOVE_FORWARD_SLOW")
            row[f"move_forward_slow_rate_{version}"] = action_rate(agg, "MOVE_FORWARD_SLOW")
            row["delta_move_forward_slow_rate"] = row[f"move_forward_slow_rate_{version}"] - row["move_forward_slow_rate_v3"]
            row["turn_rate_v3"] = turn_rate(base)
            row[f"turn_rate_{version}"] = turn_rate(agg)
            row["delta_turn_rate"] = row[f"turn_rate_{version}"] - row["turn_rate_v3"]
            row["timeout_failures_v3"] = failure_count(base, "timeout")
            row[f"timeout_failures_{version}"] = failure_count(agg, "timeout")
            row["delta_timeout_failures"] = row[f"timeout_failures_{version}"] - row["timeout_failures_v3"]
            rows.append(row)
    return rows


def diagnose_failure(row: Dict[str, Any]) -> str:
    cov_drop = fnum(row.get("delta_exploration_coverage"))
    acc_drop = fnum(row.get("delta_final_map_accuracy"))
    timeout_delta = fnum(row.get("delta_timeout_rate"))
    probe_delta = fnum(row.get("delta_probe_forward_rate"))
    move_delta = fnum(row.get("delta_move_forward_slow_rate"))
    turn_delta = fnum(row.get("delta_turn_rate"))
    if timeout_delta > 0.15 and cov_drop < -0.03:
        return "timeout after poor coverage"
    if timeout_delta > 0.15 and cov_drop >= -0.03:
        return "timeout despite moderate coverage"
    if acc_drop < -0.10 and probe_delta < -0.05:
        return "bad map quality with insufficient probing"
    if acc_drop < -0.10 and move_delta > 0.05:
        return "bad map quality with more movement"
    if turn_delta > 0.05:
        return "excessive turning"
    if cov_drop < -0.03:
        return "poor coverage expansion"
    return "minor or mixed regression"


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def build_tables(results: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_map = {version: by_map_averages(data) for version, data in results.items()}
    per_map_rows: List[Dict[str, Any]] = []
    for map_name in sorted(by_map["v3"]):
        row: Dict[str, Any] = {"map": map_name}
        for version in ["v3", "v4", "v31"]:
            agg = by_map[version][map_name]
            for metric in MAP_METRICS:
                row[f"{version}_{metric}"] = agg[metric]
            row[f"{version}_probe_forward_rate"] = agg["probe_forward_rate"]
            row[f"{version}_move_forward_slow_rate"] = agg["move_forward_slow_rate"]
            row[f"{version}_turn_rate"] = agg["turn_rate"]
            row[f"{version}_timeout_failures"] = agg["timeout_failures"]
        row["v4_delta_map_accuracy"] = row["v4_final_map_accuracy"] - row["v3_final_map_accuracy"]
        row["v31_delta_map_accuracy"] = row["v31_final_map_accuracy"] - row["v3_final_map_accuracy"]
        row["v4_failure_diagnosis"] = diagnose_failure(
            {
                "delta_exploration_coverage": row["v4_exploration_coverage"] - row["v3_exploration_coverage"],
                "delta_final_map_accuracy": row["v4_delta_map_accuracy"],
                "delta_timeout_rate": row["v4_timeout_rate"] - row["v3_timeout_rate"],
                "delta_probe_forward_rate": row["v4_probe_forward_rate"] - row["v3_probe_forward_rate"],
                "delta_move_forward_slow_rate": row["v4_move_forward_slow_rate"] - row["v3_move_forward_slow_rate"],
                "delta_turn_rate": row["v4_turn_rate"] - row["v3_turn_rate"],
            }
        )
        row["v31_failure_diagnosis"] = diagnose_failure(
            {
                "delta_exploration_coverage": row["v31_exploration_coverage"] - row["v3_exploration_coverage"],
                "delta_final_map_accuracy": row["v31_delta_map_accuracy"],
                "delta_timeout_rate": row["v31_timeout_rate"] - row["v3_timeout_rate"],
                "delta_probe_forward_rate": row["v31_probe_forward_rate"] - row["v3_probe_forward_rate"],
                "delta_move_forward_slow_rate": row["v31_move_forward_slow_rate"] - row["v3_move_forward_slow_rate"],
                "delta_turn_rate": row["v31_turn_rate"] - row["v3_turn_rate"],
            }
        )
        per_map_rows.append(row)

    per_diff_rows: List[Dict[str, Any]] = []
    for difficulty in sorted(results["v3"].get("by_difficulty", {})):
        row = {"difficulty": difficulty}
        for version in ["v3", "v4", "v31"]:
            agg = results[version]["by_difficulty"][difficulty]["aggregate"]
            for metric in MAP_METRICS:
                row[f"{version}_{metric}"] = get_metric(agg, metric)
            row[f"{version}_probe_forward_rate"] = get_metric(agg, "probe_action_ratio")
            row[f"{version}_move_forward_rate"] = get_metric(agg, "move_forward_ratio")
            row[f"{version}_turn_rate"] = get_metric(agg, "turn_ratio")
        row["v4_delta_map_accuracy"] = row["v4_final_map_accuracy"] - row["v3_final_map_accuracy"]
        row["v31_delta_map_accuracy"] = row["v31_final_map_accuracy"] - row["v3_final_map_accuracy"]
        per_diff_rows.append(row)

    action_rows: List[Dict[str, Any]] = []
    for version, data in results.items():
        overall = data["overall"]
        row = {
            "version": version,
            "probe_forward_rate": get_metric(overall, "probe_action_ratio"),
            "move_forward_slow_rate": action_rate(overall, "MOVE_FORWARD_SLOW"),
            "move_forward_rate": get_metric(overall, "move_forward_ratio"),
            "turn_rate": get_metric(overall, "turn_ratio"),
            "stop_or_reverse_rate": action_rate(overall, "STOP_OR_REVERSE"),
            "average_episode_length": get_metric(overall, "mean_steps"),
            "timeout_rate": get_metric(overall, "timeout_rate"),
            "timeout_failures": failure_count(overall, "timeout"),
        }
        for action in ACTION_NAMES:
            row[f"{action}_rate"] = action_rate(overall, action)
        action_rows.append(row)

    failure_rows: List[Dict[str, Any]] = []
    for version, data in results.items():
        overall = data["overall"]
        counts = overall.get("failure_reason_counts", {})
        if not isinstance(counts, dict):
            counts = {}
        failure_rows.append(
            {
                "version": version,
                "success_rate": get_metric(overall, "success_rate"),
                "timeout_rate": get_metric(overall, "timeout_rate"),
                "coverage": get_metric(overall, "exploration_coverage"),
                "map_accuracy": get_metric(overall, "final_map_accuracy"),
                "doorway_crossing_success_rate": get_metric(overall, "doorway_crossing_success_rate"),
                "timeout_failures": int(counts.get("timeout", 0)),
                "low_coverage_failures": int(counts.get("low_coverage", 0)),
                "doorway_not_crossed_failures": int(counts.get("doorway_not_crossed", 0)),
                "collision_failures": int(counts.get("collision", 0)),
            }
        )

    quality_rows: List[Dict[str, Any]] = []
    for row in delta_rows(results):
        quality_rows.append(
            {
                "version": row["version"],
                "map": row["map"],
                "difficulty": row["difficulty"],
                "delta_map_accuracy": row["delta_final_map_accuracy"],
                "delta_wall_f1": row["delta_final_wall_f1"],
                "delta_doorway_f1": row["delta_final_doorway_f1"],
                "delta_probe_forward_rate": row["delta_probe_forward_rate"],
                "delta_move_forward_slow_rate": row["delta_move_forward_slow_rate"],
                "delta_turn_rate": row["delta_turn_rate"],
                "delta_timeout_rate": row["delta_timeout_rate"],
                "failure_diagnosis": diagnose_failure(row),
            }
        )

    return {
        "per_map": per_map_rows,
        "per_difficulty": per_diff_rows,
        "action": action_rows,
        "failure": failure_rows,
        "quality": quality_rows,
        "delta": delta_rows(results),
    }


def summarize(results: Dict[str, Dict[str, Any]], tables: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    overall: Dict[str, Any] = {}
    for version, data in results.items():
        ov = data["overall"]
        overall[version] = {metric: get_metric(ov, metric) for metric in OVERALL_METRICS}
        overall[version]["probe_forward_rate"] = get_metric(ov, "probe_action_ratio")
        overall[version]["move_forward_rate"] = get_metric(ov, "move_forward_ratio")
        overall[version]["turn_rate"] = get_metric(ov, "turn_ratio")
        overall[version]["mean_steps"] = get_metric(ov, "mean_steps")

    quality_rows = tables["quality"]
    worst_quality = sorted(quality_rows, key=lambda r: fnum(r["delta_map_accuracy"]))[:10]
    for row in quality_rows:
        row["map_accuracy_drop_magnitude"] = -fnum(row["delta_map_accuracy"])

    map_acc_drops = [-fnum(r["delta_map_accuracy"]) for r in quality_rows]
    probe_drops = [-fnum(r["delta_probe_forward_rate"]) for r in quality_rows]
    move_increases = [fnum(r["delta_move_forward_slow_rate"]) for r in quality_rows]
    timeout_increases = [fnum(r["delta_timeout_rate"]) for r in quality_rows]

    v4_rows = [r for r in tables["delta"] if r["version"] == "v4"]
    v31_rows = [r for r in tables["delta"] if r["version"] == "v31"]
    worst_v4 = min(v4_rows, key=lambda r: fnum(r["delta_final_map_accuracy"]))
    worst_v31 = min(v31_rows, key=lambda r: fnum(r["delta_final_map_accuracy"]))

    return {
        "inputs": {version: str(path) for version, path in RESULT_PATHS.items()},
        "overall_metrics": overall,
        "overall_deltas_vs_v3": {
            version: {
                metric: overall[version][metric] - overall["v3"][metric]
                for metric in overall["v3"]
                if metric in overall[version]
            }
            for version in ["v4", "v31"]
        },
        "worst_map_difficulty_by_map_accuracy_drop": worst_quality,
        "correlations": {
            "map_accuracy_drop_vs_probe_rate_drop": correlation(map_acc_drops, probe_drops),
            "map_accuracy_drop_vs_move_forward_slow_rate_increase": correlation(map_acc_drops, move_increases),
            "map_accuracy_drop_vs_timeout_rate_increase": correlation(map_acc_drops, timeout_increases),
        },
        "worst_v4_map_difficulty": {
            "map": worst_v4["map"],
            "difficulty": worst_v4["difficulty"],
            "delta_map_accuracy": worst_v4["delta_final_map_accuracy"],
            "diagnosis": diagnose_failure(worst_v4),
        },
        "worst_v31_map_difficulty": {
            "map": worst_v31["map"],
            "difficulty": worst_v31["difficulty"],
            "delta_map_accuracy": worst_v31["delta_final_map_accuracy"],
            "diagnosis": diagnose_failure(worst_v31),
        },
        "failure_mode_summary": {
            "v4": infer_main_failure(results["v3"]["overall"], results["v4"]["overall"]),
            "v31": infer_main_failure(results["v3"]["overall"], results["v31"]["overall"]),
        },
        "recommendation": recommend_next_step(results),
    }


def infer_main_failure(v3: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    cov_delta = get_metric(candidate, "exploration_coverage") - get_metric(v3, "exploration_coverage")
    timeout_delta = get_metric(candidate, "timeout_rate") - get_metric(v3, "timeout_rate")
    acc_delta = get_metric(candidate, "final_map_accuracy") - get_metric(v3, "final_map_accuracy")
    probe_delta = get_metric(candidate, "probe_action_ratio") - get_metric(v3, "probe_action_ratio")
    move_delta = get_metric(candidate, "move_forward_ratio") - get_metric(v3, "move_forward_ratio")
    if timeout_delta > 0.15 and cov_delta < -0.03:
        return "timeout after poor coverage, with reduced map-quality accumulation"
    if acc_delta < -0.10 and probe_delta < -0.05 and move_delta > 0.05:
        return "map-quality collapse correlated with less probing and more movement"
    if timeout_delta > 0.15:
        return "timeout regression dominates"
    return "mixed regression"


def recommend_next_step(results: Dict[str, Dict[str, Any]]) -> str:
    v3 = results["v3"]["overall"]
    v31 = results["v31"]["overall"]
    probe_delta = get_metric(v31, "probe_action_ratio") - get_metric(v3, "probe_action_ratio")
    acc_delta = get_metric(v31, "final_map_accuracy") - get_metric(v3, "final_map_accuracy")
    timeout_delta = get_metric(v31, "timeout_rate") - get_metric(v3, "timeout_rate")
    if probe_delta < -0.05 and acc_delta < -0.10:
        return (
            "Create v3.2 with a map-quality-preserving probe schedule: restore v3 probing cadence in low-confidence "
            "and doorway/corridor regions, and only suppress probes after local map confidence is already high."
        )
    if timeout_delta > 0.10:
        return (
            "Create v3.2 by restoring v3 probing but improving frontier selection and stale-target handling; do not "
            "change success/termination logic until map-quality diagnostics recover."
        )
    return "Create map-quality diagnostic plots before additional planner changes."


def markdown_report(summary: Dict[str, Any], tables: Dict[str, List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    lines.append("# Phase 2D Regression Analysis")
    lines.append("")
    lines.append("Accepted v3 is compared against rejected v4 and rejected v3.1.")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| metric | v3 | v4 | v3.1 | v4 delta | v3.1 delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    overall = summary["overall_metrics"]
    deltas = summary["overall_deltas_vs_v3"]
    for metric in [
        "success_rate",
        "exploration_coverage",
        "collision_rate",
        "fake_doorway_approach_rate",
        "doorway_crossing_success_rate",
        "timeout_rate",
        "final_map_accuracy",
        "final_wall_f1",
        "final_doorway_f1",
    ]:
        lines.append(
            f"| {metric} | {overall['v3'][metric]:.4f} | {overall['v4'][metric]:.4f} | "
            f"{overall['v31'][metric]:.4f} | {deltas['v4'][metric]:+.4f} | {deltas['v31'][metric]:+.4f} |"
        )
    lines.append("")
    lines.append("## Action Behavior")
    lines.append("")
    lines.append("| version | probe rate | move-forward rate | turn rate | mean steps |")
    lines.append("|---|---:|---:|---:|---:|")
    for version in ["v3", "v4", "v31"]:
        row = overall[version]
        lines.append(
            f"| {version} | {row['probe_forward_rate']:.4f} | {row['move_forward_rate']:.4f} | "
            f"{row['turn_rate']:.4f} | {row['mean_steps']:.2f} |"
        )
    lines.append("")
    lines.append("## Map-Quality Collapse")
    lines.append("")
    lines.append(
        f"- Accuracy drop vs probe-rate drop correlation: "
        f"{summary['correlations']['map_accuracy_drop_vs_probe_rate_drop']:.4f}"
    )
    lines.append(
        f"- Accuracy drop vs movement increase correlation: "
        f"{summary['correlations']['map_accuracy_drop_vs_move_forward_slow_rate_increase']:.4f}"
    )
    lines.append(
        f"- Accuracy drop vs timeout increase correlation: "
        f"{summary['correlations']['map_accuracy_drop_vs_timeout_rate_increase']:.4f}"
    )
    lines.append("")
    lines.append("Worst map/difficulty accuracy drops:")
    for row in summary["worst_map_difficulty_by_map_accuracy_drop"][:5]:
        lines.append(
            f"- {row['version']} {row['map']} {row['difficulty']}: "
            f"delta_map_accuracy={row['delta_map_accuracy']:.4f}, diagnosis={row['failure_diagnosis']}"
        )
    lines.append("")
    lines.append("## Failure Mode Diagnosis")
    lines.append("")
    lines.append(f"- v4: {summary['failure_mode_summary']['v4']}")
    lines.append(f"- v3.1: {summary['failure_mode_summary']['v31']}")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(summary["recommendation"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    tables = build_tables(results)
    summary = summarize(results, tables)

    write_csv(
        OUT_DIR / "per_map_regression_table.csv",
        tables["per_map"],
        list(tables["per_map"][0].keys()) if tables["per_map"] else ["map"],
    )
    write_csv(
        OUT_DIR / "per_difficulty_regression_table.csv",
        tables["per_difficulty"],
        list(tables["per_difficulty"][0].keys()) if tables["per_difficulty"] else ["difficulty"],
    )
    write_csv(
        OUT_DIR / "action_rate_comparison.csv",
        tables["action"],
        list(tables["action"][0].keys()) if tables["action"] else ["version"],
    )
    write_csv(
        OUT_DIR / "failure_mode_comparison.csv",
        tables["failure"],
        list(tables["failure"][0].keys()) if tables["failure"] else ["version"],
    )
    write_csv(
        OUT_DIR / "map_quality_comparison.csv",
        tables["quality"],
        list(tables["quality"][0].keys()) if tables["quality"] else ["version"],
    )

    (OUT_DIR / "regression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "regression_summary.md").write_text(markdown_report(summary, tables), encoding="utf-8")

    worst = summary["worst_map_difficulty_by_map_accuracy_drop"][0]
    print(f"Saved regression analysis to: {OUT_DIR}")
    print(
        "Worst map/difficulty: "
        f"{worst['version']} {worst['map']} {worst['difficulty']} "
        f"delta_map_accuracy={worst['delta_map_accuracy']:.4f}"
    )
    print(f"v4 failure: {summary['failure_mode_summary']['v4']}")
    print(f"v3.1 failure: {summary['failure_mode_summary']['v31']}")
    print(f"Recommendation: {summary['recommendation']}")


if __name__ == "__main__":
    main()
