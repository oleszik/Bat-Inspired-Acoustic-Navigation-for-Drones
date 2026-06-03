"""Integration demo: grid simulator + frontier explorer + A* path planner.

Loop:
    sense -> update discovered map -> detect frontiers -> score target
    -> plan path -> execute one safe drone action -> repeat
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
    discovered_map_to_exploration_grid,
    discovered_map_to_planner_grid,
    ensure_demo_import_paths,
)


ensure_demo_import_paths()

from drone_frontier_explorer import FrontierCluster, FrontierDetector, FrontierScorer  # noqa: E402
from drone_grid_sim import DiscoveredMap, Drone, GridWorld, RandomWorldGenerator, SimpleRangeSensor  # noqa: E402
from drone_path_planner import AStarPlanner  # noqa: E402
from drone_safety_monitor import SafetyDecision, SafetyMonitor, SafetyState  # noqa: E402

try:
    from drone_safety_monitor.io import save_decision_log
except ImportError:  # pragma: no cover
    save_decision_log = None  # type: ignore[assignment]


OUTPUT_PATH = Path("integrations/outputs/grid_frontier_astar_summary.json")
SAFETY_LOG_PATH = Path("integrations/outputs/safety_decision_log.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grid + frontier + A* integration demo.")
    parser.add_argument("--world-type", default="single_block", choices=[
        "empty_room",
        "corridor",
        "single_block",
        "doorway",
        "cluttered_room",
        "random_obstacles",
    ])
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--target-coverage", type=float, default=75.0)
    parser.add_argument("--max-consecutive-no-path", type=int, default=10)
    parser.add_argument("--allow-unknown-planning", action="store_true")
    return parser.parse_args()


def find_start(world: GridWorld) -> tuple[int, int]:
    """Pick a valid free start near the upper-left interior."""

    preferred = [(1, 1), (2, 1), (1, 2), (2, 2)]
    for x, y in preferred:
        if world.is_free(x, y):
            return x, y

    for y in range(world.height):
        for x in range(world.width):
            if world.is_free(x, y):
                return x, y
    raise ValueError("World has no free start cell.")


def build_world(args: argparse.Namespace) -> GridWorld:
    generator = RandomWorldGenerator(seed=args.seed)
    return generator.generate(args.world_type, width=args.width, height=args.height)


def sorted_frontier_clusters(
    clusters: list[FrontierCluster],
    scorer: FrontierScorer,
    drone_position: tuple[int, int],
) -> list[FrontierCluster]:
    """Sort clusters by the same scoring rule used by ``FrontierScorer``."""

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
    """Choose the best reachable frontier target and return its planned path."""

    _ = scorer.choose_best_frontier(clusters, start)
    for cluster in sorted_frontier_clusters(clusters, scorer, start):
        target = (cluster.target_cell.x, cluster.target_cell.y)
        result = planner.plan(start, target)
        if result.get("found"):
            path = result.get("path", [])
            if isinstance(path, list):
                return target, path, int(result.get("path_length", 0))
    return None, [], 0


def rotate_once(drone: Drone, world: GridWorld) -> dict[str, Any]:
    """Safe fallback action when no path is available."""

    return drone.step("TURN_RIGHT", world)


def execute_safety_action(
    drone: Drone,
    world: GridWorld,
    planned_action: str | None,
    decision: SafetyDecision,
    step: int,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Apply safety decision behavior and return result/action/termination."""

    if decision.action == "CONTINUE":
        if planned_action is None:
            return None, None, None
        return drone.step(planned_action, world), planned_action, None

    if decision.action == "ROTATE_SCAN":
        rotate_action = "TURN_LEFT" if step % 2 == 0 else "TURN_RIGHT"
        return drone.step(rotate_action, world), rotate_action, None

    if decision.action == "REPLAN":
        return None, None, None

    if decision.action == "RETURN_HOME":
        return None, None, "safety_return_home"

    if decision.action == "STOP":
        return None, None, "safety_stop"

    if decision.action == "EMERGENCY_LAND":
        return None, None, "safety_emergency_land"

    return None, None, "safety_unknown_action"


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    world = build_world(args)
    start_x, start_y = find_start(world)
    drone = Drone(start_x, start_y, direction="E")
    sensor = SimpleRangeSensor()
    discovered_map = DiscoveredMap(world.width, world.height)
    scorer = FrontierScorer()
    safety_monitor = SafetyMonitor()

    collisions = 0
    replans = 0
    frontier_targets_chosen = 0
    no_path_count = 0
    consecutive_no_path = 0
    termination_reason = "max_steps_reached"
    last_action: str | None = None
    last_frontier_count = 0
    last_path_length = 0
    previous_coverage = 0.0
    steps_without_progress = 0
    safety_decision_log: list[dict[str, Any]] = []
    raw_safety_decisions: list[SafetyDecision] = []

    for step in range(1, args.max_steps + 1):
        readings = sensor.read(drone, world)
        discovered_map.update(drone, sensor, world)

        coverage = discovered_map.coverage_percent()
        if coverage >= previous_coverage + 0.1:
            steps_without_progress = 0
        else:
            steps_without_progress += 1
        previous_coverage = coverage

        if coverage >= args.target_coverage:
            termination_reason = "target_coverage_reached"
            break

        exploration_grid = discovered_map_to_exploration_grid(discovered_map)
        detector = FrontierDetector(exploration_grid)
        frontier_cells = detector.find_frontier_cells()
        clusters = detector.cluster_frontiers()
        last_frontier_count = len(frontier_cells)

        if not clusters:
            termination_reason = "no_frontier_found"
            break

        planner_grid = discovered_map_to_planner_grid(
            discovered_map,
            allow_unknown=args.allow_unknown_planning,
        )
        planner = AStarPlanner(planner_grid, allow_unknown=args.allow_unknown_planning)
        x, y, _ = drone.get_state()
        target, path, path_length = choose_reachable_frontier_path(
            clusters=clusters,
            scorer=scorer,
            planner=planner,
            start=(x, y),
        )
        replans += 1
        last_path_length = path_length

        planned_action: str | None = None
        if target is None:
            no_path_count += 1
            consecutive_no_path += 1
            if consecutive_no_path >= args.max_consecutive_no_path:
                termination_reason = "no_path_found_repeatedly"
                break
            planned_action = "TURN_RIGHT"
        else:
            frontier_targets_chosen += 1
            consecutive_no_path = 0
            planned_action = choose_safe_next_action(drone, path, world)
            if planned_action is None:
                no_path_count += 1
                planned_action = "TURN_RIGHT"

        safety_state: SafetyState = build_safety_state(
            drone=drone,
            sensor_readings=readings,
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
            f"sensor={readings}"
        )

    final_coverage = discovered_map.coverage_percent()
    if final_coverage >= args.target_coverage:
        termination_reason = "target_coverage_reached"

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
        "termination_reason": termination_reason,
        "final_position": drone.get_state(),
        "last_frontier_count": last_frontier_count,
        "last_path_length": last_path_length,
        "safety_decision_log_path": str(SAFETY_LOG_PATH),
    }
    summary["_safety_decision_log"] = safety_decision_log
    summary["_raw_safety_decisions"] = raw_safety_decisions
    return summary


def main() -> None:
    args = parse_args()
    summary = run_demo(args)
    safety_decision_log = summary.pop("_safety_decision_log")
    raw_safety_decisions = summary.pop("_raw_safety_decisions")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if save_decision_log is not None:
        save_decision_log(raw_safety_decisions, SAFETY_LOG_PATH)
    else:
        SAFETY_LOG_PATH.write_text(json.dumps(safety_decision_log, indent=2), encoding="utf-8")

    print("\nFinal summary")
    print("-" * 40)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved summary: {OUTPUT_PATH}")
    print(f"Saved safety log: {SAFETY_LOG_PATH}")


if __name__ == "__main__":
    main()
