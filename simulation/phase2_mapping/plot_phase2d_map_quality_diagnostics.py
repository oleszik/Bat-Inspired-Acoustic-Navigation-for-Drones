from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required for map-quality diagnostics") from exc


VERSIONS = ["v3", "v4", "v31"]
VERSION_LABELS = {"v3": "v3", "v4": "v4", "v31": "v3.1"}
RESULT_PATHS = {
    "v3": Path("runs/phase2_mapper_guided_navigation_v3/mapper_guided_navigation_results.json"),
    "v4": Path("runs/phase2_mapper_guided_navigation_v4/mapper_guided_navigation_results.json"),
    "v31": Path("runs/phase2_mapper_guided_navigation_v31/mapper_guided_navigation_results.json"),
}
PLOT_DIRS = {
    "v3": Path("runs/phase2_mapper_guided_navigation_v3/plots"),
    "v4": Path("runs/phase2_mapper_guided_navigation_v4/plots"),
    "v31": Path("runs/phase2_mapper_guided_navigation_v31/plots"),
}
REGRESSION_DIR = Path("runs/phase2_mapper_guided_navigation_regression_analysis")
OUT_DIR = Path("runs/phase2_mapper_guided_navigation_map_quality_diagnostics")

OVERALL_PLOT_METRICS = [
    ("success_rate", "success_rate"),
    ("exploration_coverage", "coverage"),
    ("timeout_rate", "timeout_rate"),
    ("final_map_accuracy", "final_map_accuracy"),
    ("final_wall_f1", "final_wall_f1"),
    ("final_doorway_f1", "final_doorway_f1"),
    ("probe_action_ratio", "PROBE_FORWARD rate"),
    ("move_forward_ratio", "MOVE_FORWARD_SLOW rate"),
    ("turn_ratio", "turn rate"),
]
PER_MAP_METRICS = [
    ("exploration_coverage", "coverage"),
    ("final_map_accuracy", "final_map_accuracy"),
    ("timeout_rate", "timeout_rate"),
    ("final_wall_f1", "final_wall_f1"),
    ("final_doorway_f1", "final_doorway_f1"),
]
PER_DIFF_METRICS = [
    ("exploration_coverage", "coverage"),
    ("final_map_accuracy", "final_map_accuracy"),
    ("timeout_rate", "timeout_rate"),
]
REGRESSION_METRICS = [
    ("exploration_coverage", "coverage"),
    ("final_map_accuracy", "final_map_accuracy"),
    ("timeout_rate", "timeout_rate"),
    ("success_rate", "success_rate"),
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def action_rate(agg: Dict[str, Any], action: str) -> float:
    dist = agg.get("action_distribution", {})
    if isinstance(dist, dict):
        row = dist.get(action, {})
        if isinstance(row, dict):
            return fnum(row.get("rate"))
    return 0.0


def metric_value(agg: Dict[str, Any], metric: str) -> float:
    if metric == "move_forward_ratio":
        return fnum(agg.get("move_forward_ratio"), action_rate(agg, "MOVE_FORWARD_SLOW"))
    if metric == "probe_action_ratio":
        return fnum(agg.get("probe_action_ratio"), action_rate(agg, "PROBE_FORWARD"))
    if metric == "turn_ratio":
        return fnum(agg.get("turn_ratio"), action_rate(agg, "TURN_LEFT") + action_rate(agg, "TURN_RIGHT"))
    return fnum(agg.get(metric))


def by_map_averages(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    by_map: Dict[str, List[Dict[str, Any]]] = {}
    for diff_data in data.get("by_difficulty", {}).values():
        for map_name, agg in diff_data.get("maps", {}).items():
            by_map.setdefault(map_name, []).append(agg)
    out: Dict[str, Dict[str, float]] = {}
    for map_name, aggs in by_map.items():
        row: Dict[str, float] = {}
        metrics = {m for m, _ in PER_MAP_METRICS + OVERALL_PLOT_METRICS}
        for metric in metrics:
            row[metric] = mean(metric_value(agg, metric) for agg in aggs)
        out[map_name] = row
    return out


def by_difficulty_averages(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for difficulty, diff_data in data.get("by_difficulty", {}).items():
        agg = diff_data.get("aggregate", {})
        out[difficulty] = {metric: metric_value(agg, metric) for metric, _ in PER_DIFF_METRICS + OVERALL_PLOT_METRICS}
    return out


def iter_combo(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for version, data in results.items():
        for difficulty, diff_data in data.get("by_difficulty", {}).items():
            for map_name, agg in diff_data.get("maps", {}).items():
                rows.append(
                    {
                        "version": version,
                        "map": map_name,
                        "difficulty": difficulty,
                        "coverage": metric_value(agg, "exploration_coverage"),
                        "success_rate": metric_value(agg, "success_rate"),
                        "final_map_accuracy": metric_value(agg, "final_map_accuracy"),
                        "timeout_rate": metric_value(agg, "timeout_rate"),
                        "probe_forward_rate": metric_value(agg, "probe_action_ratio"),
                        "move_forward_slow_rate": action_rate(agg, "MOVE_FORWARD_SLOW"),
                        "move_forward_rate": metric_value(agg, "move_forward_ratio"),
                        "turn_rate": metric_value(agg, "turn_ratio"),
                    }
                )
    return rows


def correlation(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def save_overall_bars(results: Dict[str, Dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), dpi=140)
    axes_flat = axes.flatten()
    xs = np.arange(len(VERSIONS))
    for ax, (metric, title) in zip(axes_flat, OVERALL_PLOT_METRICS):
        vals = [metric_value(results[v]["overall"], metric) for v in VERSIONS]
        ax.bar(xs, vals, color=["tab:blue", "tab:orange", "tab:green"])
        ax.set_xticks(xs, [VERSION_LABELS[v] for v in VERSIONS])
        ax.set_title(title)
        ax.set_ylim(0.0, max(1.0, max(vals) * 1.15))
        for i, val in enumerate(vals):
            ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / "overall_metric_bars.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_heatmap(matrix: np.ndarray, rows: List[str], cols: List[str], title: str, path: Path, cmap: str = "viridis") -> Path:
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.7), max(4, len(rows) * 0.7)), dpi=140)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(rows)), rows)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_per_map_heatmaps(results: Dict[str, Dict[str, Any]]) -> List[Path]:
    by_map = {version: by_map_averages(data) for version, data in results.items()}
    maps = sorted(by_map["v3"])
    paths: List[Path] = []
    for metric, label in PER_MAP_METRICS:
        matrix = np.asarray([[by_map[v][map_name][metric] for v in VERSIONS] for map_name in maps], dtype=float)
        paths.append(
            save_heatmap(
                matrix,
                maps,
                [VERSION_LABELS[v] for v in VERSIONS],
                f"Per-map {label}",
                OUT_DIR / f"per_map_{metric}_heatmap.png",
            )
        )
    return paths


def save_per_difficulty_heatmaps(results: Dict[str, Dict[str, Any]]) -> List[Path]:
    by_diff = {version: by_difficulty_averages(data) for version, data in results.items()}
    diffs = ["clean", "mild_noise", "medium_noise", "hard_noise"]
    paths: List[Path] = []
    for metric, label in PER_DIFF_METRICS:
        matrix = np.asarray([[by_diff[v][difficulty][metric] for v in VERSIONS] for difficulty in diffs], dtype=float)
        paths.append(
            save_heatmap(
                matrix,
                diffs,
                [VERSION_LABELS[v] for v in VERSIONS],
                f"Per-difficulty {label}",
                OUT_DIR / f"per_difficulty_{metric}_heatmap.png",
            )
        )
    return paths


def save_regression_heatmaps(results: Dict[str, Dict[str, Any]]) -> List[Path]:
    rows: List[str] = []
    matrices: Dict[str, List[List[float]]] = {metric: [] for metric, _ in REGRESSION_METRICS}
    for difficulty in ["clean", "mild_noise", "medium_noise", "hard_noise"]:
        for map_name in sorted(results["v3"]["by_difficulty"][difficulty]["maps"]):
            rows.append(f"{map_name}\n{difficulty}")
            base = results["v3"]["by_difficulty"][difficulty]["maps"][map_name]
            for metric, _ in REGRESSION_METRICS:
                matrices[metric].append(
                    [
                        metric_value(results["v4"]["by_difficulty"][difficulty]["maps"][map_name], metric)
                        - metric_value(base, metric),
                        metric_value(results["v31"]["by_difficulty"][difficulty]["maps"][map_name], metric)
                        - metric_value(base, metric),
                    ]
                )
    paths: List[Path] = []
    for metric, label in REGRESSION_METRICS:
        matrix = np.asarray(matrices[metric], dtype=float)
        paths.append(
            save_heatmap(
                matrix,
                rows,
                ["v4 - v3", "v3.1 - v3"],
                f"Regression delta: {label}",
                OUT_DIR / f"regression_delta_{metric}_heatmap.png",
                cmap="coolwarm",
            )
        )
    return paths


def save_correlation_plot(rows: List[Dict[str, Any]]) -> Tuple[Path, Dict[str, float]]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=140)
    pairs = [
        ("probe_forward_rate", "final_map_accuracy", "PROBE_FORWARD rate", "final_map_accuracy"),
        ("move_forward_slow_rate", "final_map_accuracy", "MOVE_FORWARD_SLOW rate", "final_map_accuracy"),
        ("turn_rate", "timeout_rate", "turn rate", "timeout_rate"),
        ("coverage", "success_rate", "coverage", "success_rate"),
    ]
    correlations: Dict[str, float] = {}
    colors = {"v3": "tab:blue", "v4": "tab:orange", "v31": "tab:green"}
    markers = {"v3": "o", "v4": "s", "v31": "^"}
    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes.flatten(), pairs):
        xs = [float(row[xkey]) for row in rows]
        ys = [float(row[ykey]) for row in rows]
        correlations[f"{xkey}_vs_{ykey}"] = correlation(xs, ys)
        for version in VERSIONS:
            sub = [row for row in rows if row["version"] == version]
            ax.scatter(
                [float(row[xkey]) for row in sub],
                [float(row[ykey]) for row in sub],
                label=VERSION_LABELS[version],
                alpha=0.75,
                c=colors[version],
                marker=markers[version],
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"corr={correlations[f'{xkey}_vs_{ykey}']:.3f}")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    path = OUT_DIR / "action_map_quality_correlations.png"
    fig.savefig(path)
    plt.close(fig)
    return path, correlations


def read_worst_cases() -> List[Dict[str, Any]]:
    csv_path = REGRESSION_DIR / "map_quality_comparison.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda row: float(row.get("delta_map_accuracy", 0.0)))
        return rows[:4]
    return []


def save_worst_case_panels(worst_cases: List[Dict[str, Any]]) -> Tuple[List[Path], List[str]]:
    paths: List[Path] = []
    missing: List[str] = []
    for row in worst_cases:
        version_name = row["version"]
        map_name = row["map"]
        difficulty = row["difficulty"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
        for ax, version in zip(axes, VERSIONS):
            plot_path = PLOT_DIRS[version] / f"{map_name}_{difficulty}_sample.png"
            ax.set_title(VERSION_LABELS[version])
            ax.axis("off")
            if plot_path.exists():
                ax.imshow(mpimg.imread(plot_path))
            else:
                msg = f"Missing plot: {plot_path}"
                missing.append(msg)
                ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
        fig.suptitle(f"Worst-case panel: {version_name} {map_name} {difficulty}")
        fig.tight_layout()
        out = OUT_DIR / f"worst_case_{version_name}_{map_name}_{difficulty}.png"
        fig.savefig(out)
        plt.close(fig)
        paths.append(out)
    return paths, missing


def build_summary(
    results: Dict[str, Dict[str, Any]],
    correlations: Dict[str, float],
    worst_cases: List[Dict[str, Any]],
    plot_paths: Dict[str, Any],
    missing_plots: List[str],
) -> Dict[str, Any]:
    overall = {
        version: {
            "success_rate": metric_value(results[version]["overall"], "success_rate"),
            "coverage": metric_value(results[version]["overall"], "exploration_coverage"),
            "timeout_rate": metric_value(results[version]["overall"], "timeout_rate"),
            "final_map_accuracy": metric_value(results[version]["overall"], "final_map_accuracy"),
            "final_wall_f1": metric_value(results[version]["overall"], "final_wall_f1"),
            "final_doorway_f1": metric_value(results[version]["overall"], "final_doorway_f1"),
            "probe_forward_rate": metric_value(results[version]["overall"], "probe_action_ratio"),
            "move_forward_rate": metric_value(results[version]["overall"], "move_forward_ratio"),
            "turn_rate": metric_value(results[version]["overall"], "turn_ratio"),
        }
        for version in VERSIONS
    }
    worst = worst_cases[0] if worst_cases else {}
    reduced_probe_corr = correlations.get("probe_forward_rate_vs_final_map_accuracy", 0.0)
    movement_corr = correlations.get("move_forward_slow_rate_vs_final_map_accuracy", 0.0)
    turn_timeout_corr = correlations.get("turn_rate_vs_timeout_rate", 0.0)
    return {
        "output_dir": str(OUT_DIR),
        "overall": overall,
        "worst_regressing_case": worst,
        "correlations": correlations,
        "answers": {
            "where_v3_outperforms_most": (
                f"{worst.get('version', 'unknown')} {worst.get('map', 'unknown')} {worst.get('difficulty', 'unknown')} "
                f"has the largest map-accuracy drop versus v3."
            ),
            "lower_probing_correlation": (
                "Lower probing does not show a strong global positive correlation with map accuracy in the aggregate scatter, "
                "but the v3.1 regression mechanism is still consistent with over-suppressing probes in specific low-confidence cases."
                if reduced_probe_corr <= 0.2
                else "Lower probing appears positively correlated with worse map accuracy."
            ),
            "movement_timeout_or_accuracy_correlation": (
                f"MOVE_FORWARD_SLOW rate vs map accuracy correlation is {movement_corr:.3f}; "
                f"turn rate vs timeout correlation is {turn_timeout_corr:.3f}."
            ),
            "v32_first_target": "doorway mild_noise, followed by hard-noise structured maps with map-accuracy drops.",
            "safest_v32_change": (
                "Preserve v3 probing/map-evidence behavior and only reduce probes when local confidence is already high; "
                "prefer safe forward only when map confidence ahead is sufficient, keep v3 frontier logic mostly unchanged, "
                "and do not add adaptive switching."
            ),
        },
        "plots": plot_paths,
        "missing_visual_panel_plots": missing_plots,
    }


def write_markdown(summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Phase 2D Map-Quality Diagnostics")
    lines.append("")
    lines.append("Accepted v3 remains the reference. Rejected v4 and v3.1 are compared for map-quality regressions.")
    lines.append("")
    lines.append("## Main Plots")
    for group, paths in summary["plots"].items():
        if isinstance(paths, list):
            for path in paths:
                lines.append(f"- {group}: `{path}`")
        else:
            lines.append(f"- {group}: `{paths}`")
    lines.append("")
    lines.append("## Answers")
    for key, value in summary["answers"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Correlations")
    for key, value in summary["correlations"].items():
        lines.append(f"- {key}: {value:.4f}")
    lines.append("")
    lines.append("## Worst Visual Case")
    worst = summary["worst_regressing_case"]
    if worst:
        lines.append(
            f"- {worst.get('version')} {worst.get('map')} {worst.get('difficulty')} "
            f"delta_map_accuracy={float(worst.get('delta_map_accuracy', 0.0)):.4f}"
        )
    else:
        lines.append("- No worst-case rows available.")
    if summary["missing_visual_panel_plots"]:
        lines.append("")
        lines.append("## Missing Plot Notes")
        for msg in summary["missing_visual_panel_plots"]:
            lines.append(f"- {msg}")
    lines.append("")
    (OUT_DIR / "map_quality_diagnostics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {version: load_json(path) for version, path in RESULT_PATHS.items()}

    plot_paths: Dict[str, Any] = {}
    plot_paths["overall_metric_bars"] = str(save_overall_bars(results))
    plot_paths["per_map_heatmaps"] = [str(path) for path in save_per_map_heatmaps(results)]
    plot_paths["per_difficulty_heatmaps"] = [str(path) for path in save_per_difficulty_heatmaps(results)]
    plot_paths["regression_heatmaps"] = [str(path) for path in save_regression_heatmaps(results)]
    combo_rows = iter_combo(results)
    corr_path, correlations = save_correlation_plot(combo_rows)
    plot_paths["action_map_quality_correlations"] = str(corr_path)
    worst_cases = read_worst_cases()
    panel_paths, missing_plots = save_worst_case_panels(worst_cases)
    plot_paths["worst_case_panels"] = [str(path) for path in panel_paths]

    summary = build_summary(results, correlations, worst_cases, plot_paths, missing_plots)
    (OUT_DIR / "map_quality_diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)

    worst = summary["worst_regressing_case"]
    print(f"Saved diagnostics to: {OUT_DIR}")
    if worst:
        print(
            "Worst visual failure case: "
            f"{worst.get('version')} {worst.get('map')} {worst.get('difficulty')} "
            f"delta_map_accuracy={float(worst.get('delta_map_accuracy', 0.0)):.4f}"
        )
    print(f"Probe/map-accuracy correlation: {correlations.get('probe_forward_rate_vs_final_map_accuracy', 0.0):.4f}")
    print(summary["answers"]["safest_v32_change"])


if __name__ == "__main__":
    main()
