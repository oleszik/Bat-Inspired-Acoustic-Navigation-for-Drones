"""Small adapters between the grid simulator, frontier explorer, and A* planner.

The three projects intentionally keep separate grid classes. This file is the
thin compatibility layer for the first ecosystem integration demo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def ensure_demo_import_paths() -> None:
    """Make sibling repos importable when running from the umbrella repo root."""

    umbrella_root = Path(__file__).resolve().parents[1]
    sibling_root = umbrella_root.parent
    for repo_name in (
        "drone-grid-simulator",
        "drone-path-planner",
        "drone-frontier-explorer",
        "drone-safety-monitor",
        "drone-occupancy-mapper",
    ):
        repo_path = sibling_root / repo_name
        if repo_path.exists():
            repo_path_text = str(repo_path)
            if repo_path_text not in sys.path:
                sys.path.insert(0, repo_path_text)


ensure_demo_import_paths()

from drone_frontier_explorer import ExplorationGrid  # noqa: E402
from drone_occupancy_mapper import OccupancyGrid, Pose2D, RangeReadings  # noqa: E402
from drone_path_planner import PlannerGrid  # noqa: E402
from drone_safety_monitor import SafetyState  # noqa: E402


DIRECTION_VECTORS: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}

TURN_LEFT: dict[str, str] = {
    "N": "W",
    "W": "S",
    "S": "E",
    "E": "N",
}

TURN_RIGHT: dict[str, str] = {
    "N": "E",
    "E": "S",
    "S": "W",
    "W": "N",
}


def _rows_from_discovered_map(discovered_map: Any) -> list[str]:
    """Return discovered map rows as strings using '#', '.', and '?'."""

    rows: list[str] = []
    for row in discovered_map.grid:
        rows.append("".join(str(cell) for cell in row))
    return rows


def discovered_map_to_exploration_grid(discovered_map: Any) -> ExplorationGrid:
    """Convert a simulator ``DiscoveredMap`` into a frontier ``ExplorationGrid``."""

    return ExplorationGrid.from_rows(_rows_from_discovered_map(discovered_map))


def discovered_map_to_planner_grid(discovered_map: Any, allow_unknown: bool = False) -> PlannerGrid:
    """Convert a simulator ``DiscoveredMap`` into a planner ``PlannerGrid``.

    Cell rules:
    - ``#`` remains obstacle.
    - ``.`` remains free.
    - ``?`` remains unknown.

    Unknown cells are represented as ``?`` in the grid. They are blocked by
    default when ``AStarPlanner(..., allow_unknown=False)`` is used.
    """

    _ = allow_unknown
    return PlannerGrid.from_rows(_rows_from_discovered_map(discovered_map))


def _direction_for_step(current: tuple[int, int], nxt: tuple[int, int]) -> str | None:
    dx = nxt[0] - current[0]
    dy = nxt[1] - current[1]
    for direction, vector in DIRECTION_VECTORS.items():
        if vector == (dx, dy):
            return direction
    return None


def _turn_actions_to_face(current_direction: str, target_direction: str) -> list[str]:
    if current_direction == target_direction:
        return []
    if TURN_LEFT[current_direction] == target_direction:
        return ["TURN_LEFT"]
    if TURN_RIGHT[current_direction] == target_direction:
        return ["TURN_RIGHT"]
    return ["TURN_RIGHT", "TURN_RIGHT"]


def path_to_drone_actions(drone: Any, path: list[tuple[int, int]]) -> list[str]:
    """Convert a grid path into drone turn/forward actions."""

    if len(path) < 2:
        return []

    _, _, current_direction = drone.get_state()
    actions: list[str] = []
    for current, nxt in zip(path, path[1:]):
        target_direction = _direction_for_step(current, nxt)
        if target_direction is None:
            continue
        turn_actions = _turn_actions_to_face(current_direction, target_direction)
        actions.extend(turn_actions)
        actions.append("FORWARD")
        current_direction = target_direction
    return actions


def choose_safe_next_action(drone: Any, path: list[tuple[int, int]], world: Any) -> str | None:
    """Choose only the next immediate safe action for following ``path``."""

    if len(path) < 2:
        return None

    x, y, direction = drone.get_state()
    next_cell = path[1]
    target_direction = _direction_for_step((x, y), next_cell)
    if target_direction is None:
        return None

    if direction != target_direction:
        turns = _turn_actions_to_face(direction, target_direction)
        return turns[0] if turns else None

    dx, dy = DIRECTION_VECTORS[direction]
    forward_cell = (x + dx, y + dy)
    if forward_cell != next_cell:
        return None
    if not world.is_free(*forward_cell):
        return None
    return "FORWARD"


def build_safety_state(
    drone: Any,
    sensor_readings: dict[str, int],
    collision_count: int,
    no_path_count: int,
    steps_without_progress: int,
    coverage_percent: float,
    step_count: int,
    max_steps: int,
) -> SafetyState:
    """Build a ``SafetyState`` snapshot for ``SafetyMonitor.evaluate``."""

    x, y, direction = drone.get_state()
    return SafetyState(
        position=(int(x), int(y)),
        direction=str(direction),
        front_distance=int(sensor_readings.get("front", 0)),
        left_distance=int(sensor_readings.get("left", 0)),
        right_distance=int(sensor_readings.get("right", 0)),
        collision_count=int(collision_count),
        no_path_count=int(no_path_count),
        steps_without_progress=int(steps_without_progress),
        coverage_percent=float(coverage_percent),
        step_count=int(step_count),
        max_steps=int(max_steps),
    )


def occupancy_grid_to_rows(occupancy_grid: OccupancyGrid) -> list[str]:
    """Convert classified occupancy cells into '#', '.', '?' rows."""

    rows: list[str] = []
    for y in range(occupancy_grid.height):
        row = []
        for x in range(occupancy_grid.width):
            row.append(occupancy_grid.classify_cell(x, y))
        rows.append("".join(row))
    return rows


def occupancy_grid_to_exploration_grid(occupancy_grid: OccupancyGrid) -> ExplorationGrid:
    """Convert an ``OccupancyGrid`` into a frontier ``ExplorationGrid``."""

    return ExplorationGrid.from_rows(occupancy_grid_to_rows(occupancy_grid))


def occupancy_grid_to_planner_grid(occupancy_grid: OccupancyGrid) -> PlannerGrid:
    """Convert an ``OccupancyGrid`` into a planner ``PlannerGrid``.

    Unknown cells remain ``?`` and are blocked by default by ``AStarPlanner``.
    """

    return PlannerGrid.from_rows(occupancy_grid_to_rows(occupancy_grid))


def sensor_readings_to_mapper_readings(sensor_readings: dict[str, int]) -> RangeReadings:
    """Convert simulator sensor readings into occupancy-mapper readings."""

    return RangeReadings(
        front_distance=int(sensor_readings.get("front", 0)),
        left_distance=int(sensor_readings.get("left", 0)),
        right_distance=int(sensor_readings.get("right", 0)),
    )


def drone_to_pose2d(drone: Any) -> Pose2D:
    """Convert simulator drone state into an occupancy-mapper pose."""

    x, y, direction = drone.get_state()
    return Pose2D(x=int(x), y=int(y), direction=str(direction))
