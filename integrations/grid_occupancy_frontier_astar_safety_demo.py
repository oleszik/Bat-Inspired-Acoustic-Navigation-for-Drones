"""Integration demo: occupancy mapper + frontier explorer + A* + safety monitor.

Loop:
    sense -> probabilistic occupancy update -> classified map -> frontier
    -> plan -> safety check -> move
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.utils import (  # noqa: E402
    build_safety_state,
    choose_safe_next_action,
    drone_to_pose2d,
    ensure_demo_import_paths,
    occupancy_grid_to_exploration_grid,
    occupancy_grid_to_planner_grid,
    occupancy_grid_to_rows,
    sensor_readings_to_mapper_readings,
)


ensure_demo_import_paths()

from drone_frontier_explorer import FrontierCluster, FrontierDetector, FrontierScorer  # noqa: E402
from drone_grid_sim import Drone, GridWorld, RandomWorldGenerator, SimpleRangeSensor  # noqa: E402
from drone_occupancy_mapper import MappingConfig, OccupancyGrid, OccupancyMapper  # noqa: E402
from drone_path_planner import AStarPlanner  # noqa: E402
from drone_safety_monitor import SafetyDecision, SafetyMonitor, SafetyState  # noqa: E402

try:
    from drone_occupancy_mapper.io import save_occupancy_grid_json
except ImportError:  # pragma: no cover
    save_occupancy_grid_json = None  # type: ignore[assignment]

try:
    from drone_safety_monitor.io import save_decision_log
except ImportError:  # pragma: no cover
    save_decision_log = None  # type: ignore[assignment]


SUMMARY_PATH = Path("integrations/outputs/grid_occupancy_frontier_astar_safety_summary.json")
OCCUPANCY_MAP_PATH = Path("integrations/outputs/final_occupancy_map.json")
SAFETY_LOG_PATH = Path("integrations/outputs/safety_decision_log_occupancy_demo.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run occupancy-map frontier A* safety integration demo.")
    parser.add_argument(
        "--world-type",
        default="single_block",
        choices=[
            "empty_room",
            "corridor",
            "single_block",
            "doorway",
            "cluttered_room",
            "random_obstacles",
        ],
    )
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--target-coverage", type=float, default=60.0)
    parser.add_argument("--max-consecutive-no-path", type=int, default=12)
    parser.add_argument("--early-scan-steps", type=int, default=8)
    return parser.parse_args()


def build_world(args: argparse.Namespace) -> GridWorld:
    generator = RandomWorldGenerator(seed=args.seed)
    return generator.generate(args.world_type, width=args.width, height=args.height)


def find_start(world: GridWorld) -> tuple[int, int]:
    for x, y in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        if world.is_free(x, y):
            return x, y
    for y in range(world.height):
        for x in range(world.width):
            if world.is_free(x, y):
                return x, y
    raise ValueError("World has no free start cell.")


def classified_counts(occupancy_grid: OccupancyGrid) -> dict[str, int]:
    counts = {"occupied_cells": 0, "free_cells": 0, "unknown_cells": 0}
    for row in occupancy_grid_to_rows(occupancy_grid):
        counts["occupied_cells"] += row.count("#")
        counts["free_cells"] += row.count(".")
        counts["unknown_cells"] += row.count("?")
    return counts


def classified_coverage_percent(occupancy_grid: OccupancyGrid) -> float:
    counts = classified_counts(occupancy_grid)
    total = occupancy_grid.width * occupancy_grid.height
    known = counts["occupied_cells"] + counts["free_cells"]
    return 100.0 * known / total if total else 0.0


def sorted_frontier_clusters(
    clusters: list[FrontierCluster],
    scorer: FrontierScorer,
    drone_position: tuple[int, int],
) -> list[FrontierCluster]:
    return sorted(
        clusters,
        key=lambda cluster: (
            scorer.score_cluster(cluster, drone_position),
            cluster.size,
            -abs(cluster.target_cell.x - drone_position[0]) - abs(cluster.target_cell.y - drone_position[1]),
        ),
        reverse=True,
    )


def choose_reachable_frontier_path(
    clusters: list[FrontierCluster],
    scorer: FrontierScorer,
    planner: AStarPlanner,
    start: tuple[int, int],
) -> tuple[tuple[int, int] | None, list[tuple[int, int]], int]:
    _ = scorer.choose_best_frontier(clusters, start)
    for cluster in sorted_frontier_clusters(clusters, scorer, start):
        target = (cluster.target_cell.x, cluster.target_cell.y)
        result = planner.plan(start, target)
        if result.get("found"):
            path = result.get("path", [])
            if isinstance(path, list):
                return target, path, int(result.get("path_length", 0))
    return None, [], 0


def execute_safety_action(
    drone: Drone,
    world: GridWorld,
    planned_action: str | None,
    decision: SafetyDecision,
    step: int,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if decision.action == "CONTINUE":
        if planned_action is None:
            return None, None, None
        return drone.step(planned_action, world), planned_action, None
    if decision.action == "ROTATE_SCAN":
        action = "TURN_LEFT" if step % 2 == 0 else "TURN_RIGHT"
        return drone.step(action, world), action, None
    if decision.action == "REPLAN":
        return None, None, None
    if decision.action == "RETURN_HOME":
        return None, None, "safety_return_home"
    if decision.action == "STOP":
        return None, None, "safety_stop"
    if decision.action == "EMERGENCY_LAND":
        return None, None, "safety_emergency_land"
    return None, None, "safety_unknown_action"


def update_occupancy_from_sensor(
    mapper: OccupancyMapper,
    drone: Drone,
    sensor: SimpleRangeSensor,
    world: GridWorld,
) -> dict[str, int]:
    sensor_readings = sensor.read(drone, world)
    mapper.update_from_range_readings(
        pose=drone_to_pose2d(drone),
        readings=sensor_readings_to_mapper_readings(sensor_readings),
    )
    return sensor_readings


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    world = build_world(args)
    start_x, start_y = find_start(world)
    drone = Drone(start_x, start_y, direction="E")
    sensor = SimpleRangeSensor()
    mapping_config = MappingConfig()
    occupancy_grid = OccupancyGrid(world.width, world.height, config=mapping_config)
    mapper = OccupancyMapper(occupancy_grid, config=mapping_config)
    scorer = FrontierScorer()
    safety_monitor = SafetyMonitor()

    # Seed the probabilistic map at the start pose before the control loop.
    first_readings = update_occupancy_from_sensor(mapper, drone, sensor, world)
    _ = first_readings

    collisions = 0
    replans = 0
    frontier_targets_chosen = 0
    no_path_count = 0
    consecutive_no_path = 0
    termination_reason = "max_steps_reached"
    previous_coverage = classified_coverage_percent(occupancy_grid)
    steps_without_progress = 0
    last_action: str | None = None
    last_frontier_count = 0
    last_path_length = 0
    safety_decision_log: list[dict[str, Any]] = []
    raw_safety_decisions: list[SafetyDecision] = []

    for step in range(1, args.max_steps + 1):
        sensor_readings = update_occupancy_from_sensor(mapper, drone, sensor, world)
        coverage = classified_coverage_percent(occupancy_grid)
        if coverage >= previous_coverage + 0.1:
            steps_without_progress = 0
        else:
            steps_without_progress += 1
        previous_coverage = coverage

        if coverage >= args.target_coverage:
            termination_reason = "target_coverage_reached"
            break

        exploration_grid = occupancy_grid_to_exploration_grid(occupancy_grid)
        detector = FrontierDetector(exploration_grid)
        frontier_cells = detector.find_frontier_cells()
        clusters = detector.cluster_frontiers()
        last_frontier_count = len(frontier_cells)

        planned_action: str | None = None
        if not clusters:
            if step <= args.early_scan_steps:
                planned_action = "TURN_RIGHT" if step % 2 else "TURN_LEFT"
            else:
                termination_reason = "no_frontier_found"
                break
        else:
            planner_grid = occupancy_grid_to_planner_grid(occupancy_grid)
            planner = AStarPlanner(planner_grid, allow_unknown=False)
            x, y, _ = drone.get_state()
            target, path, path_length = choose_reachable_frontier_path(
                clusters=clusters,
                scorer=scorer,
                planner=planner,
                start=(x, y),
            )
            replans += 1
            last_path_length = path_length

            if target is None:
                no_path_count += 1
                consecutive_no_path += 1
                planned_action = "TURN_RIGHT" if step % 2 else "TURN_LEFT"
                if consecutive_no_path >= args.max_consecutive_no_path:
                    termination_reason = "no_path_found_repeatedly"
                    break
            else:
                frontier_targets_chosen += 1
                consecutive_no_path = 0
                planned_action = choose_safe_next_action(drone, path, world)
                if planned_action is None:
                    no_path_count += 1
                    planned_action = "TURN_RIGHT" if step % 2 else "TURN_LEFT"

        safety_state: SafetyState = build_safety_state(
            drone=drone,
            sensor_readings=sensor_readings,
            collision_count=collisions,
            no_path_count=no_path_count,
            steps_without_progress=steps_without_progress,
            coverage_percent=coverage,
            step_count=step,
            max_steps=args.max_steps,
        )
        safety_decision = safety_monitor.evaluate(safety_state)
        raw_safety_decisions.append(safety_decision)

        result, executed_action, safety_termination = execute_safety_action(
            drone=drone,
            world=world,
            planned_action=planned_action,
            decision=safety_decision,
            step=step,
        )
        last_action = executed_action
        if result is not None and result["collision"]:
            collisions += 1

        safety_decision_log.append(
            {
                "step": step,
                "state": asdict(safety_state),
                "decision": asdict(safety_decision),
                "planned_action": planned_action,
                "executed_action": executed_action,
                "coverage": coverage,
            }
        )

        print(
            f"safety action={safety_decision.action:<14} "
            f"severity={safety_decision.severity:<8} reason={safety_decision.reason}"
        )

        if safety_termination is not None:
            termination_reason = safety_termination
            break

        pos_x, pos_y, direction = drone.get_state()
        print(
            f"step={step:03d} action={str(last_action):<10} "
            f"pos=({pos_x},{pos_y},{direction}) coverage={coverage:5.1f}% "
            f"frontiers={last_frontier_count:3d} path_len={last_path_length:3d} "
            f"sensor={sensor_readings}"
        )

    final_coverage = classified_coverage_percent(occupancy_grid)
    if final_coverage >= args.target_coverage:
        termination_reason = "target_coverage_reached"
    counts = classified_counts(occupancy_grid)

    summary = {
        "world_type": args.world_type,
        "width": world.width,
        "height": world.height,
        "seed": args.seed,
        "steps": step if "step" in locals() else 0,
        "final_coverage": final_coverage,
        "collisions": collisions,
        "replans": replans,
        "frontier_targets_chosen": frontier_targets_chosen,
        "no_path_count": no_path_count,
        "safety_decisions_count": len(safety_decision_log),
        "safety_warning_count": sum(1 for item in safety_decision_log if item["decision"]["severity"] == "WARNING"),
        "safety_critical_count": sum(1 for item in safety_decision_log if item["decision"]["severity"] == "CRITICAL"),
        "occupied_cells": counts["occupied_cells"],
        "free_cells": counts["free_cells"],
        "unknown_cells": counts["unknown_cells"],
        "termination_reason": termination_reason,
        "final_position": drone.get_state(),
        "last_frontier_count": last_frontier_count,
        "last_path_length": last_path_length,
        "summary_path": str(SUMMARY_PATH),
        "occupancy_map_path": str(OCCUPANCY_MAP_PATH),
        "safety_decision_log_path": str(SAFETY_LOG_PATH),
    }
    summary["_occupancy_grid"] = occupancy_grid
    summary["_safety_decision_log"] = safety_decision_log
    summary["_raw_safety_decisions"] = raw_safety_decisions
    return summary


def save_outputs(summary: dict[str, Any]) -> dict[str, Any]:
    occupancy_grid = summary.pop("_occupancy_grid")
    safety_decision_log = summary.pop("_safety_decision_log")
    raw_safety_decisions = summary.pop("_raw_safety_decisions")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if save_occupancy_grid_json is not None:
        save_occupancy_grid_json(occupancy_grid, OCCUPANCY_MAP_PATH)
    else:
        payload = {
            "width": occupancy_grid.width,
            "height": occupancy_grid.height,
            "probabilities": occupancy_grid.to_probability_list(),
            "classified_rows": occupancy_grid_to_rows(occupancy_grid),
        }
        OCCUPANCY_MAP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if save_decision_log is not None:
        save_decision_log(raw_safety_decisions, SAFETY_LOG_PATH)
    else:
        SAFETY_LOG_PATH.write_text(json.dumps(safety_decision_log, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = save_outputs(run_demo(args))

    print("\nFinal occupancy integration summary")
    print("-" * 48)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved summary: {SUMMARY_PATH}")
    print(f"Saved occupancy map: {OCCUPANCY_MAP_PATH}")
    print(f"Saved safety log: {SAFETY_LOG_PATH}")


if __name__ == "__main__":
    main()
