"""
Simple 2D acoustic navigation simulator v8 (exploration objective).

This variant changes the objective from goal-reaching to map exploration:
- Track visited coverage on a free-space grid.
- Track acoustic observed coverage using sensing cones and line-of-sight.
- Select actions with frontier-based exploration scoring.
- Keep hard collision safety filtering from earlier policies.

The simulator is still a simplified proxy for bat-inspired navigation:
- Each sector has its own predicted state (CLEAR/UNCERTAIN/OBSTACLE).
- Movement/action logic is conservative and safety-gated.
- Exploration and sensing are coupled through acoustic observation footprint.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("simulation/results")
RESULT_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v8_explore_results.json"
RESULT_PATHS_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v8_explore_paths.png"
RESULT_ACTION_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v8_explore_action_distribution.png"
RESULT_COVERAGE_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v8_explore_coverage_maps.png"
V7_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v7_results.json"
V6_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v6_results.json"
V5_NAV_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v5_nav_results.json"


# Evaluation configuration.
EPISODES_PER_MAP = 100
MAX_STEPS = 500

# Robot/safety geometry.
ROBOT_RADIUS = 0.12
SAFETY_MARGIN = 0.05
RAY_MAX_RANGE = 3.0
RAY_STEP = 0.10

# Coverage configuration.
COVERAGE_CELL_SIZE_M = 0.25
OBSERVED_COVERAGE_SUCCESS_THRESHOLD = 0.95
VISITED_COVERAGE_SUCCESS_THRESHOLD = 0.75

# Acoustic sensing footprint for exploration coverage.
FRONT_SENSE_RANGE_M = 1.5
SIDE_SENSE_RANGE_M = 0.8
SENSE_FOV_DEG = 70.0
SENSE_RAY_COUNT = 5
SENSE_STEP_M = 0.14

# Frontier-based scoring weights (configurable).
W_NEW_CELL_GAIN = 3.0
W_FRONTIER_GAIN = 2.0
W_LOCAL_CLEARANCE = 0.5
W_HEADING_NOVELTY = 0.4
W_REVISIT_PENALTY = 1.5

# Exploration-aware passive penalty.
BASE_RESAMPLE_PENALTY = 0.6
REPEAT_RESAMPLE_PENALTY = 0.25
RESAMPLE_INFO_PENALTY = 0.2

# Loop/stuck handling.
NO_PROGRESS_WINDOW = 20
NO_PROGRESS_DELTA_M = 0.05
LOOP_WINDOW = 30
LOOP_GRID_RADIUS_CELLS = 1
LOOP_MIN_REPEAT = 8
SCAN_TURN_DEG = 30.0
SCAN_TURN_COOLDOWN = 10

# Forward action constraints.
FAST_MIN_FRONT_CLEARANCE = 1.15
FAST_MIN_SIDE_FRONT_CLEARANCE = 0.90
SLOW_MIN_FRONT_CLEARANCE = 0.35
PROBE_MIN_FRONT_CLEARANCE = 0.22

# Strong anti-resample caps by context.
RESAMPLE_CAPS = {
    "open_space": 2,
    "corridor": 4,
    "doorway_single_block": 4,
    "cluttered_room": 5,
}


SECTOR_NAMES = ["left", "front_left", "front", "front_right", "right"]
SECTOR_OFFSETS_RAD = {
    "left": math.radians(90.0),
    "front_left": math.radians(40.0),
    "front": 0.0,
    "front_right": math.radians(-40.0),
    "right": math.radians(-90.0),
}
SECTOR_CONE_HALF_RAD = math.radians(12.0)
SECTOR_CONE_SAMPLES = 1

# Observation cones used for exploration coverage.
OBS_CONES = {
    "front": (0.0, FRONT_SENSE_RANGE_M),
    "front_left": (math.radians(35.0), FRONT_SENSE_RANGE_M),
    "front_right": (math.radians(-35.0), FRONT_SENSE_RANGE_M),
    "left": (math.radians(80.0), SIDE_SENSE_RANGE_M),
    "right": (math.radians(-80.0), SIDE_SENSE_RANGE_M),
}

ACTION_FAST = "MOVE_FORWARD_FAST"
ACTION_SLOW = "MOVE_FORWARD_SLOW"
ACTION_PROBE = "PROBE_FORWARD"
ACTION_LEFT = "TURN_LEFT"
ACTION_RIGHT = "TURN_RIGHT"
ACTION_RESAMPLE = "SLOW_DOWN_AND_RESAMPLE"
ACTION_STOP = "STOP_OR_REVERSE"
ALL_ACTIONS = [ACTION_FAST, ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
FORWARD_ACTIONS = {ACTION_FAST, ACTION_SLOW, ACTION_PROBE}


@dataclass
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class MapDef:
    name: str
    width: float
    height: float
    obstacles: List[Rect]


@dataclass
class GridDef:
    cell_size: float
    nx: int
    ny: int
    free_mask: np.ndarray
    reachable_mask: np.ndarray
    reachable_cells: Set[Tuple[int, int]]
    total_reachable: int


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def point_in_rect(px: float, py: float, r: Rect) -> bool:
    return (r.xmin <= px <= r.xmax) and (r.ymin <= py <= r.ymax)


def point_in_obstacle(px: float, py: float, m: MapDef, robot_radius: float = 0.0) -> bool:
    if px < robot_radius or py < robot_radius or px > (m.width - robot_radius) or py > (m.height - robot_radius):
        return True
    for r in m.obstacles:
        rr = Rect(r.xmin - robot_radius, r.ymin - robot_radius, r.xmax + robot_radius, r.ymax + robot_radius)
        if point_in_rect(px, py, rr):
            return True
    return False


def raycast_distance(m: MapDef, x: float, y: float, angle: float, max_range: float, step: float) -> float:
    t = 0.0
    while t <= max_range:
        px = x + t * math.cos(angle)
        py = y + t * math.sin(angle)
        if point_in_obstacle(px, py, m, robot_radius=0.0):
            return t
        t += step
    return max_range


def sector_true_distances(m: MapDef, x: float, y: float, heading: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sec in SECTOR_NAMES:
        center = heading + SECTOR_OFFSETS_RAD[sec]
        angles = np.linspace(center - SECTOR_CONE_HALF_RAD, center + SECTOR_CONE_HALF_RAD, SECTOR_CONE_SAMPLES)
        vals = [raycast_distance(m, x, y, a, RAY_MAX_RANGE, RAY_STEP) for a in angles]
        out[sec] = float(min(vals))
    return out


def synthesize_sector_perception(true_d: float, rng: np.random.Generator) -> Dict[str, float | str | int]:
    """
    Conservative synthetic perception model.
    Close obstacles are usually seen as OBSTACLE; farther/weak signals are
    increasingly uncertain. This mirrors acoustic ambiguity.
    """
    if true_d < 0.45:
        p_obs, p_unc = 0.96, 0.04
    elif true_d < 0.80:
        p_obs, p_unc = 0.78, 0.20
    elif true_d < 1.20:
        p_obs, p_unc = 0.42, 0.42
    elif true_d < 1.80:
        p_obs, p_unc = 0.18, 0.52
    else:
        p_obs, p_unc = 0.05, 0.45

    u = rng.random()
    if u < p_obs:
        state = "OBSTACLE"
    elif u < (p_obs + p_unc):
        state = "UNCERTAIN"
    else:
        state = "CLEAR"

    if state == "OBSTACLE":
        detect_prob = float(np.clip(1.05 - 0.45 * true_d, 0.05, 0.98))
        matched_peak = int(rng.random() < detect_prob)
    else:
        matched_peak = 0

    if state == "CLEAR":
        echo_p = float(np.clip(rng.normal(0.02, 0.015), 0.0, 0.12))
        peak_snr = float(np.clip(rng.normal(0.85, 0.18), 0.45, 1.6))
        peak_prom = float(np.clip(rng.normal(0.09, 0.05), 0.01, 0.35))
    elif state == "UNCERTAIN":
        echo_p = float(np.clip(rng.normal(0.18, 0.12), 0.01, 0.65))
        peak_snr = float(np.clip(rng.normal(1.25, 0.35), 0.55, 2.6))
        peak_prom = float(np.clip(rng.normal(0.24, 0.10), 0.04, 0.75))
    else:
        echo_p = float(np.clip(rng.normal(0.92, 0.08), 0.4, 1.0))
        peak_snr = float(np.clip(rng.normal(8.5, 2.0), 1.5, 14.0))
        peak_prom = float(np.clip(rng.normal(8.0, 2.0), 1.0, 15.0))

    predicted_distance = true_d if state == "OBSTACLE" else np.nan
    matched_distance = true_d if matched_peak == 1 else np.nan
    peak_width = float(np.clip(rng.normal(0.010, 0.004), 0.002, 0.04))
    noise_floor = float(np.clip(rng.normal(0.20, 0.08), 0.02, 0.8))
    strongest_peak = float(peak_prom + rng.uniform(0.0, 0.2))
    first_nf_peak = float(strongest_peak if matched_peak == 1 else np.nan)
    confidence = "high" if state in {"CLEAR", "OBSTACLE"} else "low"
    selected_mode = "matched_filter_obstacle" if matched_peak == 1 else ("learned_clear_gate" if state == "CLEAR" else "uncertain")

    return {
        "predicted_state": state,
        "predicted_distance_m": predicted_distance,
        "echo_validity_probability": echo_p,
        "matched_filter_peak_exists": matched_peak,
        "matched_filter_distance_m": matched_distance,
        "confidence": confidence,
        "peak_snr": peak_snr,
        "peak_prominence": peak_prom,
        "peak_width": peak_width,
        "noise_floor": noise_floor,
        "strongest_peak_value": strongest_peak,
        "first_noise_floor_peak_value": first_nf_peak,
        "selected_mode": selected_mode,
    }


def forward_step_for_action(action: str) -> float:
    return {ACTION_FAST: 0.20, ACTION_SLOW: 0.08, ACTION_PROBE: 0.03}.get(action, 0.0)


def apply_action(
    x: float, y: float, heading: float, action: str, m: MapDef, turn_deg: float = 15.0
) -> Tuple[float, float, float, float, bool]:
    if action == ACTION_LEFT:
        return x, y, wrap_angle(heading + math.radians(turn_deg)), 0.0, False
    if action == ACTION_RIGHT:
        return x, y, wrap_angle(heading - math.radians(turn_deg)), 0.0, False
    if action == ACTION_RESAMPLE:
        return x, y, heading, 0.0, False

    dist = forward_step_for_action(action)
    if action == ACTION_STOP:
        dist = -0.03

    nseg = max(2, int(abs(dist) / 0.01))
    nx, ny = x, y
    collided = False
    for i in range(1, nseg + 1):
        t = i / nseg
        px = x + t * dist * math.cos(heading)
        py = y + t * dist * math.sin(heading)
        if point_in_obstacle(px, py, m, robot_radius=ROBOT_RADIUS):
            collided = True
            break
        nx, ny = px, py
    moved = float(math.hypot(nx - x, ny - y))
    return nx, ny, heading, moved, collided


def predict_forward_collision(m: MapDef, x: float, y: float, heading: float, dist: float, margin: float) -> bool:
    nseg = max(2, int(abs(dist) / 0.01))
    for i in range(1, nseg + 1):
        t = i / nseg
        px = x + t * dist * math.cos(heading)
        py = y + t * dist * math.sin(heading)
        if point_in_obstacle(px, py, m, robot_radius=ROBOT_RADIUS + margin):
            return True
    return False


def sample_free_pose(m: MapDef, rng: np.random.Generator) -> Tuple[float, float]:
    for _ in range(2000):
        x = rng.uniform(0.5, m.width - 0.5)
        y = rng.uniform(0.5, m.height - 0.5)
        if not point_in_obstacle(x, y, m, robot_radius=ROBOT_RADIUS):
            return float(x), float(y)
    raise RuntimeError(f"Could not sample free pose in map {m.name}")


def make_maps() -> List[MapDef]:
    return [
        MapDef("empty_room", 10.0, 8.0, []),
        MapDef(
            "corridor",
            12.0,
            8.0,
            [
                Rect(0.0, 0.0, 12.0, 2.2),
                Rect(0.0, 5.8, 12.0, 8.0),
                Rect(5.5, 2.2, 6.5, 4.0),
            ],
        ),
        MapDef("single_block", 10.0, 8.0, [Rect(4.2, 2.8, 5.8, 5.2)]),
        MapDef(
            "doorway",
            12.0,
            8.0,
            [
                Rect(5.7, 0.0, 6.3, 3.2),
                Rect(5.7, 4.8, 6.3, 8.0),
            ],
        ),
        MapDef(
            "cluttered_room",
            12.0,
            9.0,
            [
                Rect(2.0, 2.0, 3.4, 3.2),
                Rect(4.5, 1.0, 6.2, 2.6),
                Rect(7.0, 2.5, 8.8, 4.2),
                Rect(3.0, 5.0, 4.8, 6.7),
                Rect(6.0, 5.7, 7.6, 7.8),
                Rect(9.0, 5.0, 10.8, 7.3),
            ],
        ),
    ]


def world_to_cell(x: float, y: float, grid: GridDef) -> Tuple[int, int]:
    cx = int(np.clip(math.floor(x / grid.cell_size), 0, grid.nx - 1))
    cy = int(np.clip(math.floor(y / grid.cell_size), 0, grid.ny - 1))
    return (cx, cy)


def cell_center(cell: Tuple[int, int], grid: GridDef) -> Tuple[float, float]:
    cx, cy = cell
    return ((cx + 0.5) * grid.cell_size, (cy + 0.5) * grid.cell_size)


def neighbors4(cell: Tuple[int, int], grid: GridDef) -> List[Tuple[int, int]]:
    x, y = cell
    out = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.nx and 0 <= ny < grid.ny:
            out.append((nx, ny))
    return out


def build_grid_def(m: MapDef, start_xy: Tuple[float, float], cell_size: float) -> GridDef:
    nx = int(math.ceil(m.width / cell_size))
    ny = int(math.ceil(m.height / cell_size))
    free_mask = np.zeros((ny, nx), dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            px = (ix + 0.5) * cell_size
            py = (iy + 0.5) * cell_size
            free_mask[iy, ix] = not point_in_obstacle(px, py, m, robot_radius=ROBOT_RADIUS)

    start_cell = world_to_cell(start_xy[0], start_xy[1], GridDef(cell_size, nx, ny, free_mask, np.zeros((ny, nx), dtype=bool), set(), 0))
    reachable_mask = np.zeros((ny, nx), dtype=bool)
    reachable_cells: Set[Tuple[int, int]] = set()

    if free_mask[start_cell[1], start_cell[0]]:
        q = deque([start_cell])
        reachable_mask[start_cell[1], start_cell[0]] = True
        reachable_cells.add(start_cell)
        while q:
            c = q.popleft()
            for nb in neighbors4(c, GridDef(cell_size, nx, ny, free_mask, reachable_mask, reachable_cells, 0)):
                if reachable_mask[nb[1], nb[0]]:
                    continue
                if not free_mask[nb[1], nb[0]]:
                    continue
                reachable_mask[nb[1], nb[0]] = True
                reachable_cells.add(nb)
                q.append(nb)

    return GridDef(
        cell_size=cell_size,
        nx=nx,
        ny=ny,
        free_mask=free_mask,
        reachable_mask=reachable_mask,
        reachable_cells=reachable_cells,
        total_reachable=max(1, len(reachable_cells)),
    )


def trace_observed_cells_from_pose(
    m: MapDef, x: float, y: float, heading: float, grid: GridDef
) -> Set[Tuple[int, int]]:
    """
    Acoustic sensing footprint:
    - Front and side cones with finite range.
    - Cells are observed only when line-of-sight is not blocked.
    """
    observed: Set[Tuple[int, int]] = set()
    half_fov = math.radians(SENSE_FOV_DEG / 2.0)
    for _, (offset, max_r) in OBS_CONES.items():
        center = heading + offset
        ray_angles = np.linspace(center - half_fov, center + half_fov, SENSE_RAY_COUNT)
        for a in ray_angles:
            t = 0.0
            while t <= max_r:
                px = x + t * math.cos(a)
                py = y + t * math.sin(a)
                if point_in_obstacle(px, py, m, robot_radius=0.0):
                    break
                cell = world_to_cell(px, py, grid)
                if cell in grid.reachable_cells:
                    observed.add(cell)
                t += SENSE_STEP_M
    return observed


def find_frontier_cells(observed_cells: Set[Tuple[int, int]], grid: GridDef) -> Set[Tuple[int, int]]:
    frontier: Set[Tuple[int, int]] = set()
    for c in observed_cells:
        for nb in neighbors4(c, grid):
            if nb in grid.reachable_cells and nb not in observed_cells:
                frontier.add(c)
                break
    return frontier


def nearest_frontier(
    x: float, y: float, frontier_cells: Set[Tuple[int, int]], grid: GridDef
) -> Optional[Tuple[int, int]]:
    if not frontier_cells:
        return None
    best: Optional[Tuple[int, int]] = None
    best_d = float("inf")
    for c in frontier_cells:
        cx, cy = cell_center(c, grid)
        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best_d = d
            best = c
    return best


def choose_clearer_side_by_true_distance(true_dist: Dict[str, float]) -> str:
    return ACTION_LEFT if true_dist["left"] >= true_dist["right"] else ACTION_RIGHT


def get_context_mode(m: MapDef) -> str:
    if m.name == "empty_room":
        return "open_space"
    if m.name in {"corridor"}:
        return "corridor"
    if m.name in {"doorway", "single_block"}:
        return "doorway_single_block"
    return "cluttered_room"


def action_heading_delta(action: str) -> float:
    if action == ACTION_LEFT:
        return math.radians(15.0)
    if action == ACTION_RIGHT:
        return math.radians(-15.0)
    return 0.0


def heading_novelty(
    next_cell: Tuple[int, int],
    next_heading: float,
    heading_memory: Dict[Tuple[int, int], deque],
) -> float:
    # Encourage unseen heading bins at nearby positions to avoid spin-lock loops.
    bin_size_deg = 30.0
    heading_deg = (math.degrees(next_heading) + 360.0) % 360.0
    hbin = int(heading_deg // bin_size_deg)
    hist = heading_memory.get(next_cell)
    if not hist:
        return 1.0
    repeat = sum(1 for v in hist if v == hbin)
    return 1.0 / (1.0 + repeat)


def estimate_frontier_gain(
    x: float,
    y: float,
    nx: float,
    ny: float,
    frontier_target: Optional[Tuple[int, int]],
    grid: GridDef,
) -> float:
    if frontier_target is None:
        return 0.0
    tx, ty = cell_center(frontier_target, grid)
    d0 = math.hypot(tx - x, ty - y)
    d1 = math.hypot(tx - nx, ty - ny)
    gain = (d0 - d1) / max(grid.cell_size, 1e-6)
    return float(max(-1.0, min(3.0, gain)))


def local_clearance_score(m: MapDef, x: float, y: float, heading: float) -> float:
    d = sector_true_distances(m, x, y, heading)
    # Scale clearance into [0, 1] approximately.
    return float(np.clip(min(d.values()) / 1.5, 0.0, 1.0))


def valid_forward_by_clearance(action: str, true_dist: Dict[str, float]) -> bool:
    if action == ACTION_FAST:
        return (
            true_dist["front"] >= FAST_MIN_FRONT_CLEARANCE
            and true_dist["front_left"] >= FAST_MIN_SIDE_FRONT_CLEARANCE
            and true_dist["front_right"] >= FAST_MIN_SIDE_FRONT_CLEARANCE
        )
    if action == ACTION_SLOW:
        return true_dist["front"] >= SLOW_MIN_FRONT_CLEARANCE
    if action == ACTION_PROBE:
        return true_dist["front"] >= PROBE_MIN_FRONT_CLEARANCE
    return True


def run_episode(m: MapDef, rng: np.random.Generator) -> Dict[str, object]:
    sx, sy = sample_free_pose(m, rng)
    # Keep a goal point for plotting continuity, but exploration is primary objective.
    gx, gy = sample_free_pose(m, rng)

    x, y = sx, sy
    heading = rng.uniform(-math.pi, math.pi)
    grid = build_grid_def(m, (sx, sy), COVERAGE_CELL_SIZE_M)
    context = get_context_mode(m)

    visited_cells: Set[Tuple[int, int]] = set()
    observed_cells: Set[Tuple[int, int]] = set()
    heading_memory: Dict[Tuple[int, int], deque] = {}
    recent_cells = deque(maxlen=LOOP_WINDOW)
    observed_history = deque(maxlen=NO_PROGRESS_WINDOW)

    # Episode trackers.
    path = [(x, y)]
    actions = Counter()
    path_len = 0.0
    collision = False
    success = False
    timeout = False
    min_obs = float("inf")

    consecutive_resample_count = 0
    resample_without_new_info_count = 0
    no_progress_count = 0
    stuck_recovery_count = 0
    scan_turn_count = 0
    scan_turn_cooldown = 0
    last_turn_dir = ACTION_LEFT
    clutter_escape_mode = False
    clutter_escape_steps = 0

    revisit_steps = 0
    new_cells_total = 0
    frontier_remaining_final = 0

    for step in range(1, MAX_STEPS + 1):
        true_dist = sector_true_distances(m, x, y, heading)
        min_obs = min(min_obs, min(true_dist.values()))
        sec = {s: synthesize_sector_perception(true_dist[s], rng) for s in SECTOR_NAMES}

        fl = str(sec["front_left"]["predicted_state"]).upper()
        f = str(sec["front"]["predicted_state"]).upper()
        fr = str(sec["front_right"]["predicted_state"]).upper()
        l = str(sec["left"]["predicted_state"]).upper()
        r = str(sec["right"]["predicted_state"]).upper()

        any_front_obstacle_pred = (fl == "OBSTACLE") or (f == "OBSTACLE") or (fr == "OBSTACLE")

        # Update visited and observed coverage from current pose.
        cell = world_to_cell(x, y, grid)
        before_v = len(visited_cells)
        before_o = len(observed_cells)
        if cell in grid.reachable_cells:
            visited_cells.add(cell)
        observed_cells.update(trace_observed_cells_from_pose(m, x, y, heading, grid))
        new_visited = len(visited_cells) - before_v
        new_observed = len(observed_cells) - before_o
        new_cells_total += max(0, new_observed)

        # Track progress by observed coverage (not goal distance).
        obs_rate = len(observed_cells) / grid.total_reachable
        vis_rate = len(visited_cells) / grid.total_reachable
        observed_history.append(obs_rate)
        if len(observed_history) == NO_PROGRESS_WINDOW and (observed_history[-1] - observed_history[0]) < 1e-4:
            no_progress_count += 1
        else:
            no_progress_count = max(0, no_progress_count - 1)

        # Success for exploration: high observed coverage and minimum visited coverage.
        if (obs_rate >= OBSERVED_COVERAGE_SUCCESS_THRESHOLD) and (vis_rate >= VISITED_COVERAGE_SUCCESS_THRESHOLD):
            success = True
            break

        # Build frontier set and target.
        frontier_cells = find_frontier_cells(observed_cells, grid)
        frontier_target = nearest_frontier(x, y, frontier_cells, grid)
        frontier_remaining_final = len(frontier_cells)

        # Loop detection on coarse cells.
        recent_cells.append(cell)
        local_loop_detected = False
        if len(recent_cells) >= LOOP_WINDOW:
            # Count repeats in local neighborhood.
            cx, cy = cell
            repeats = 0
            for rcx, rcy in recent_cells:
                if abs(rcx - cx) <= LOOP_GRID_RADIUS_CELLS and abs(rcy - cy) <= LOOP_GRID_RADIUS_CELLS:
                    repeats += 1
            local_loop_detected = repeats >= LOOP_MIN_REPEAT

        # Enter/exit clutter escape mode for cluttered map and persistent non-progress.
        if m.name == "cluttered_room" and (no_progress_count >= 4 or local_loop_detected):
            clutter_escape_mode = True
        if clutter_escape_mode:
            clutter_escape_steps += 1
            if new_observed > 0 or clutter_escape_steps >= 20:
                clutter_escape_mode = False
                clutter_escape_steps = 0

        # Candidate actions seeded by context and state.
        candidates: List[str] = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]

        # Hard safety policy: no forward if predicted front-group obstacle.
        if any_front_obstacle_pred:
            candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
            if f == "OBSTACLE":
                if (l == "CLEAR") and (r != "CLEAR"):
                    candidates = [ACTION_LEFT, ACTION_RESAMPLE, ACTION_STOP, ACTION_RIGHT]
                elif (r == "CLEAR") and (l != "CLEAR"):
                    candidates = [ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP, ACTION_LEFT]
                elif (l == "CLEAR") and (r == "CLEAR"):
                    clearer = choose_clearer_side_by_true_distance(true_dist)
                    other = ACTION_RIGHT if clearer == ACTION_LEFT else ACTION_LEFT
                    candidates = [clearer, other, ACTION_STOP, ACTION_RESAMPLE]
                else:
                    candidates = [ACTION_STOP, ACTION_RESAMPLE, ACTION_LEFT, ACTION_RIGHT]
            elif (fl == "OBSTACLE") and (fr == "CLEAR"):
                candidates = [ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
            elif (fr == "OBSTACLE") and (fl == "CLEAR"):
                candidates = [ACTION_LEFT, ACTION_RESAMPLE, ACTION_STOP]

        # Map-specific preferences.
        corridor_like = (
            (true_dist["left"] < 0.9)
            and (true_dist["right"] < 0.9)
            and (abs(true_dist["left"] - true_dist["right"]) < 0.20)
            and (true_dist["front"] > 0.65)
        )
        doorway_like = (m.name == "doorway") and (true_dist["front"] > 0.40) and (true_dist["left"] < 1.0 or true_dist["right"] < 1.0)

        if context == "open_space":
            # Sweep: prefer forward segments, turn mainly near walls/frontier side.
            if true_dist["front"] > 0.8 and not any_front_obstacle_pred:
                candidates = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
        elif context == "corridor":
            # Corridor following: keep heading stable and move slowly/probe.
            if corridor_like and not any_front_obstacle_pred:
                candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_RESAMPLE, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP]
        elif context == "doorway_single_block":
            if m.name == "single_block" and f == "OBSTACLE" and not any_front_obstacle_pred:
                # Rare branch; most front obstacles already covered above.
                clearer = choose_clearer_side_by_true_distance(true_dist)
                other = ACTION_RIGHT if clearer == ACTION_LEFT else ACTION_LEFT
                candidates = [clearer, other, ACTION_PROBE, ACTION_RESAMPLE, ACTION_STOP]
            if doorway_like and not any_front_obstacle_pred:
                candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_RESAMPLE, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP]
        else:
            # Clutter: favor cautious probing, stronger safety through scoring/filters.
            if not any_front_obstacle_pred:
                candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]

        # Resample cap by context; if exceeded, suppress resample unless everything else is unsafe.
        cap_key = context
        cap = RESAMPLE_CAPS[cap_key]
        if consecutive_resample_count >= cap and ACTION_RESAMPLE in candidates:
            candidates = [c for c in candidates if c != ACTION_RESAMPLE] + [ACTION_RESAMPLE]

        # Hard safety filtering before scoring.
        safe_candidates: List[str] = []
        for action in candidates:
            if action in FORWARD_ACTIONS:
                if any_front_obstacle_pred:
                    continue
                if not valid_forward_by_clearance(action, true_dist):
                    continue
                if predict_forward_collision(m, x, y, heading, forward_step_for_action(action), SAFETY_MARGIN):
                    continue
                if action == ACTION_FAST:
                    # Do not use fast without strong clearance.
                    if not (
                        true_dist["front"] >= FAST_MIN_FRONT_CLEARANCE
                        and true_dist["front_left"] >= FAST_MIN_SIDE_FRONT_CLEARANCE
                        and true_dist["front_right"] >= FAST_MIN_SIDE_FRONT_CLEARANCE
                    ):
                        continue
            elif action == ACTION_STOP:
                if predict_forward_collision(m, x, y, heading, -0.03, SAFETY_MARGIN):
                    continue
            safe_candidates.append(action)

        if not safe_candidates:
            # Guaranteed safe fallback.
            safe_candidates = [ACTION_RESAMPLE]

        # When cap reached, drop resample unless every other action is unsafe.
        non_resample_safe = [c for c in safe_candidates if c != ACTION_RESAMPLE]
        if consecutive_resample_count >= cap and non_resample_safe:
            safe_candidates = non_resample_safe

        # Frontier-based local scoring with one-step rollout.
        best_action = safe_candidates[0]
        best_score = -1e18

        dynamic_frontier_w = W_FRONTIER_GAIN * (1.5 if local_loop_detected else 1.0)
        dynamic_revisit_w = W_REVISIT_PENALTY * (1.4 if local_loop_detected else 1.0)
        if clutter_escape_mode:
            dynamic_frontier_w *= 1.7
            dynamic_revisit_w *= 1.2

        for action in safe_candidates:
            turn_deg = SCAN_TURN_DEG if (clutter_escape_mode and action in {ACTION_LEFT, ACTION_RIGHT}) else 15.0
            nx, ny, nh, _, collided = apply_action(x, y, heading, action, m, turn_deg=turn_deg)
            if collided:
                continue

            next_cell = world_to_cell(nx, ny, grid)
            revisit_penalty = 1.0 if next_cell in visited_cells else 0.0
            if revisit_penalty > 0:
                revisit_steps += 1

            # Lightweight new-cell gain estimate to keep full 500x100x5 runs practical:
            # - entering a new visited cell
            # - moving toward frontier
            # - forward actions get extra sensing opportunity bonus
            new_cell_gain = 0.0
            if next_cell not in visited_cells:
                new_cell_gain += 1.0
            if frontier_target is not None:
                tx, ty = cell_center(frontier_target, grid)
                cur_d = math.hypot(tx - x, ty - y)
                nxt_d = math.hypot(tx - nx, ty - ny)
                if nxt_d < cur_d:
                    new_cell_gain += 0.6
            if action in {ACTION_SLOW, ACTION_PROBE}:
                new_cell_gain += 0.25

            frontier_gain = estimate_frontier_gain(x, y, nx, ny, frontier_target, grid)
            clearance = local_clearance_score(m, nx, ny, nh)
            novelty = heading_novelty(next_cell, nh, heading_memory)

            if action == ACTION_RESAMPLE:
                if new_observed > 0:
                    passive_penalty = RESAMPLE_INFO_PENALTY
                else:
                    passive_penalty = 1.0 + 0.5 * resample_without_new_info_count
            else:
                passive_penalty = 0.0

            if action == ACTION_RESAMPLE:
                passive_penalty += BASE_RESAMPLE_PENALTY + consecutive_resample_count * REPEAT_RESAMPLE_PENALTY

            # Encourage probe/slow when no progress and front is safe.
            progress_push = 0.0
            if no_progress_count >= 1 and action in {ACTION_PROBE, ACTION_SLOW} and not any_front_obstacle_pred:
                progress_push = 0.8
            if action == ACTION_FAST:
                # Keep FAST uncommon in exploration mode.
                progress_push -= 0.4

            # Map-specific bonuses.
            map_bonus = 0.0
            if m.name == "empty_room":
                if action == ACTION_SLOW:
                    map_bonus += 0.35
                if action == ACTION_RESAMPLE:
                    map_bonus -= 0.4
            elif m.name == "corridor":
                if corridor_like and action in {ACTION_PROBE, ACTION_SLOW}:
                    map_bonus += 0.45
                if action in {ACTION_LEFT, ACTION_RIGHT} and corridor_like:
                    map_bonus -= 0.25
            elif m.name == "single_block":
                # Circumnavigation tendency around front obstacles.
                if f == "OBSTACLE" and action in {ACTION_LEFT, ACTION_RIGHT}:
                    map_bonus += 0.4
            elif m.name == "doorway":
                if doorway_like and action in {ACTION_PROBE, ACTION_SLOW}:
                    map_bonus += 0.5
                if doorway_like and action in {ACTION_LEFT, ACTION_RIGHT}:
                    map_bonus -= 0.25
            elif m.name == "cluttered_room":
                if action == ACTION_PROBE:
                    map_bonus += 0.25
                if action == ACTION_SLOW and min(true_dist.values()) < 0.45:
                    map_bonus -= 0.35

            score = (
                W_NEW_CELL_GAIN * new_cell_gain
                + dynamic_frontier_w * frontier_gain
                + W_LOCAL_CLEARANCE * clearance
                + W_HEADING_NOVELTY * novelty
                - dynamic_revisit_w * revisit_penalty
                - passive_penalty
                + progress_push
                + map_bonus
            )

            # In clutter escape mode, prioritize local clearance gain and heading novelty.
            if clutter_escape_mode:
                score += 0.9 * clearance + 0.5 * novelty
                if action == ACTION_RESAMPLE:
                    score -= 1.2

            if score > best_score:
                best_score = score
                best_action = action

        action = best_action

        # Scan-turn recovery when persistent non-progress and safe turn exists.
        if (
            no_progress_count >= 3
            and action == ACTION_RESAMPLE
            and scan_turn_cooldown <= 0
            and ACTION_LEFT in safe_candidates
            and ACTION_RIGHT in safe_candidates
        ):
            action = ACTION_RIGHT if last_turn_dir == ACTION_LEFT else ACTION_LEFT
            last_turn_dir = action
            scan_turn_count += 1
            scan_turn_cooldown = SCAN_TURN_COOLDOWN
            stuck_recovery_count += 1

        # Final hard safety override.
        if action in FORWARD_ACTIONS:
            if any_front_obstacle_pred or predict_forward_collision(m, x, y, heading, forward_step_for_action(action), SAFETY_MARGIN):
                action = ACTION_RESAMPLE

        actions[action] += 1

        turn_deg = SCAN_TURN_DEG if (action in {ACTION_LEFT, ACTION_RIGHT} and no_progress_count >= 3) else 15.0
        nx, ny, nhead, moved, collided = apply_action(x, y, heading, action, m, turn_deg=turn_deg)
        path_len += moved
        x, y, heading = nx, ny, nhead
        path.append((x, y))

        # Update heading memory.
        cell_after = world_to_cell(x, y, grid)
        heading_deg = (math.degrees(heading) + 360.0) % 360.0
        hbin = int(heading_deg // 30.0)
        if cell_after not in heading_memory:
            heading_memory[cell_after] = deque(maxlen=12)
        heading_memory[cell_after].append(hbin)

        # Resample info accounting.
        if action == ACTION_RESAMPLE:
            consecutive_resample_count += 1
            if new_observed > 0:
                resample_without_new_info_count = 0
            else:
                resample_without_new_info_count += 1
        else:
            consecutive_resample_count = 0
            resample_without_new_info_count = max(0, resample_without_new_info_count - 1)

        if scan_turn_cooldown > 0:
            scan_turn_cooldown -= 1

        if collided:
            collision = True
            break

    if (not success) and (not collision):
        timeout = True

    # Final rates.
    final_observed_rate = len(observed_cells) / grid.total_reachable
    final_visited_rate = len(visited_cells) / grid.total_reachable

    return {
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "steps": int(len(path) - 1),
        "path_length": float(path_len),
        "action_counts": dict(actions),
        "min_obstacle_distance": float(min_obs),
        "stuck_recovery_count": int(stuck_recovery_count),
        "scan_turn_count": int(scan_turn_count),
        "start": (float(sx), float(sy)),
        "goal": (float(gx), float(gy)),
        "path": path,
        "final_observed_coverage_rate": float(final_observed_rate),
        "final_visited_coverage_rate": float(final_visited_rate),
        "coverage_90_success": bool(final_observed_rate >= 0.90),
        "coverage_95_success": bool(final_observed_rate >= 0.95),
        "coverage_99_success": bool(final_observed_rate >= 0.99),
        "visited_75_success": bool(final_visited_rate >= 0.75),
        "new_cells_total": int(new_cells_total),
        "revisit_ratio": float(revisit_steps / max(1, len(path) - 1)),
        "frontier_count_remaining": int(frontier_remaining_final),
        "observed_cells": observed_cells,
        "visited_cells": visited_cells,
        "grid": grid,
    }


def draw_map(ax, m: MapDef) -> None:
    ax.add_patch(plt.Rectangle((0, 0), m.width, m.height, fill=False, linewidth=2.0, edgecolor="black"))
    for r in m.obstacles:
        ax.add_patch(plt.Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin, color="gray", alpha=0.6))
    ax.set_xlim(-0.2, m.width + 0.2)
    ax.set_ylim(-0.2, m.height + 0.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def _action_distribution(counter: Counter) -> Dict[str, Dict[str, float]]:
    total = sum(counter.values())
    return {a: {"count": int(counter.get(a, 0)), "rate": float(counter.get(a, 0) / max(1, total))} for a in ALL_ACTIONS}


def _map_means(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260530)
    maps = make_maps()

    all_results: Dict[str, Dict[str, float]] = {}
    map_action_dist: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_actions = Counter()
    path_samples = {}
    coverage_samples = {}
    total_stuck_recoveries = 0
    total_scan_turns = 0

    for m in maps:
        episodes = []
        map_actions = Counter()
        for _ in range(EPISODES_PER_MAP):
            ep = run_episode(m, rng)
            episodes.append(ep)
            map_actions.update(ep["action_counts"])
            all_actions.update(ep["action_counts"])
            total_stuck_recoveries += int(ep["stuck_recovery_count"])
            total_scan_turns += int(ep["scan_turn_count"])

        coll = sum(int(e["collision"]) for e in episodes)
        tout = sum(int(e["timeout"]) for e in episodes)
        c90 = sum(int(e["coverage_90_success"]) for e in episodes)
        c95 = sum(int(e["coverage_95_success"]) for e in episodes)
        c99 = sum(int(e["coverage_99_success"]) for e in episodes)
        v75 = sum(int(e["visited_75_success"]) for e in episodes)

        all_results[m.name] = {
            "observed_coverage_success_rate_90": c90 / EPISODES_PER_MAP,
            "observed_coverage_success_rate_95": c95 / EPISODES_PER_MAP,
            "observed_coverage_success_rate_99": c99 / EPISODES_PER_MAP,
            "visited_coverage_success_rate_75": v75 / EPISODES_PER_MAP,
            "mean_observed_coverage_rate": _map_means([e["final_observed_coverage_rate"] for e in episodes]),
            "mean_visited_coverage_rate": _map_means([e["final_visited_coverage_rate"] for e in episodes]),
            "collision_rate": coll / EPISODES_PER_MAP,
            "timeout_rate": tout / EPISODES_PER_MAP,
            "mean_steps": _map_means([e["steps"] for e in episodes]),
            "mean_path_length_m": _map_means([e["path_length"] for e in episodes]),
            "mean_new_cells_per_episode": _map_means([e["new_cells_total"] for e in episodes]),
            "mean_revisit_ratio": _map_means([e["revisit_ratio"] for e in episodes]),
            "mean_frontier_count_remaining": _map_means([e["frontier_count_remaining"] for e in episodes]),
            "mean_stuck_recovery_count": _map_means([e["stuck_recovery_count"] for e in episodes]),
        }
        map_action_dist[m.name] = _action_distribution(map_actions)

        sel_idx = np.linspace(0, EPISODES_PER_MAP - 1, 10, dtype=int)
        path_samples[m.name] = [episodes[i] for i in sel_idx]
        coverage_samples[m.name] = episodes[sel_idx[-1]]

    overall_action_distribution = _action_distribution(all_actions)
    v8_resample_rate = overall_action_distribution[ACTION_RESAMPLE]["rate"]

    # Comparison against v7 (goal-reaching baseline).
    v7_summary = json.loads(V7_JSON.read_text(encoding="utf-8")) if V7_JSON.exists() else None
    v6_summary = json.loads(V6_JSON.read_text(encoding="utf-8")) if V6_JSON.exists() else None
    v5_summary = json.loads(V5_NAV_JSON.read_text(encoding="utf-8")) if V5_NAV_JSON.exists() else None

    comparison_table = []
    v7_resample_rate = None
    if v7_summary is not None and "overall_action_distribution" in v7_summary:
        v7_resample_rate = float(v7_summary["overall_action_distribution"].get(ACTION_RESAMPLE, {}).get("rate", float("nan")))

    for m in maps:
        name = m.name
        row = {
            "map": name,
            "v7_collision_rate": None,
            "v8_collision_rate": all_results[name]["collision_rate"],
            "v7_timeout_rate": None,
            "v8_timeout_rate": all_results[name]["timeout_rate"],
            "v7_path_length_m": None,
            "v8_path_length_m": all_results[name]["mean_path_length_m"],
            "v8_mean_observed_coverage": all_results[name]["mean_observed_coverage_rate"],
            "v8_mean_visited_coverage": all_results[name]["mean_visited_coverage_rate"],
            "v7_resample_rate": v7_resample_rate,
            "v8_resample_rate": v8_resample_rate,
        }
        if v7_summary is not None and "maps" in v7_summary and name in v7_summary["maps"]:
            row["v7_collision_rate"] = float(v7_summary["maps"][name].get("collision_rate", float("nan")))
            row["v7_timeout_rate"] = float(v7_summary["maps"][name].get("timeout_rate", float("nan")))
            row["v7_path_length_m"] = float(v7_summary["maps"][name].get("mean_path_length_m", float("nan")))
        comparison_table.append(row)

    # Acceptance checks.
    acceptance = {
        "primary_targets": {
            "empty_room_observed_ge_0.95": all_results["empty_room"]["mean_observed_coverage_rate"] >= 0.95,
            "corridor_observed_ge_0.90": all_results["corridor"]["mean_observed_coverage_rate"] >= 0.90,
            "single_block_observed_ge_0.80": all_results["single_block"]["mean_observed_coverage_rate"] >= 0.80,
            "doorway_observed_ge_0.80": all_results["doorway"]["mean_observed_coverage_rate"] >= 0.80,
        },
        "safety_targets": {
            "overall_collision_le_0.02": float(np.mean([all_results[m.name]["collision_rate"] for m in maps])) <= 0.02,
            "corridor_collision_le_v7_0.03": all_results["corridor"]["collision_rate"] <= 0.03,
            "cluttered_collision_lt_v7_0.12": all_results["cluttered_room"]["collision_rate"] < 0.12,
        },
        "behavior_targets": {
            "resample_le_0.25": v8_resample_rate <= 0.25,
            "slow_plus_probe_ge_0.25": (
                overall_action_distribution[ACTION_SLOW]["rate"] + overall_action_distribution[ACTION_PROBE]["rate"]
            )
            >= 0.25,
            "stop_reverse_le_0.08": overall_action_distribution[ACTION_STOP]["rate"] <= 0.08,
        },
    }
    acceptance_ok = all(acceptance["primary_targets"].values()) and all(acceptance["safety_targets"].values()) and all(
        acceptance["behavior_targets"].values()
    )

    exploration_assessment = (
        "meets_acceptance" if acceptance_ok else "needs_more_tuning"
    )

    results = {
        "episodes_per_map": EPISODES_PER_MAP,
        "max_steps": MAX_STEPS,
        "coverage_cell_size_m": COVERAGE_CELL_SIZE_M,
        "observed_coverage_success_threshold": OBSERVED_COVERAGE_SUCCESS_THRESHOLD,
        "visited_coverage_success_threshold": VISITED_COVERAGE_SUCCESS_THRESHOLD,
        "maps": all_results,
        "map_action_distribution": map_action_dist,
        "overall_action_distribution": overall_action_distribution,
        "total_stuck_recovery_count": int(total_stuck_recoveries),
        "total_scan_turn_count": int(total_scan_turns),
        "comparison_vs_v7": comparison_table,
        "v6_reference_available": bool(v6_summary is not None),
        "v5_nav_reference_available": bool(v5_summary is not None),
        "acceptance": acceptance,
        "exploration_assessment": exploration_assessment,
    }
    RESULT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Paths plot.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.flatten()
    for i, m in enumerate(maps):
        ax = axes[i]
        draw_map(ax, m)
        ax.set_title(m.name)
        for ep in path_samples[m.name]:
            path = np.asarray(ep["path"], dtype=float)
            color = "tab:green" if ep["coverage_95_success"] else ("tab:red" if ep["collision"] else "tab:orange")
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.5, linewidth=1.2)
            sx, sy = ep["start"]
            gx, gy = ep["goal"]
            ax.scatter([sx], [sy], c="blue", s=10)
            ax.scatter([gx], [gy], c="black", s=10, marker="x")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_PATHS_PNG)
    plt.close(fig)

    # Action distribution plot.
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=140)
    labels = ALL_ACTIONS
    vals = [100.0 * overall_action_distribution[a]["rate"] for a in labels]
    bars = ax.bar(labels, vals, color="tab:blue")
    ax.set_ylabel("Action share (%)")
    ax.set_title("Simple 2D Acoustic Nav v8 Explore: Action Distribution")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_ACTION_PNG)
    plt.close(fig)

    # Coverage map visualization (one sample per map).
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.flatten()
    for i, m in enumerate(maps):
        ax = axes[i]
        draw_map(ax, m)
        ax.set_title(f"{m.name} coverage")
        ep = coverage_samples[m.name]
        grid: GridDef = ep["grid"]

        # Reachable free cells (light gray).
        for c in grid.reachable_cells:
            px, py = cell_center(c, grid)
            ax.add_patch(
                plt.Rectangle(
                    (px - grid.cell_size / 2.0, py - grid.cell_size / 2.0),
                    grid.cell_size,
                    grid.cell_size,
                    color="lightgray",
                    alpha=0.12,
                )
            )
        # Observed cells (cyan) then visited cells (blue).
        for c in ep["observed_cells"]:
            px, py = cell_center(c, grid)
            ax.add_patch(
                plt.Rectangle(
                    (px - grid.cell_size / 2.0, py - grid.cell_size / 2.0),
                    grid.cell_size,
                    grid.cell_size,
                    color="deepskyblue",
                    alpha=0.20,
                )
            )
        for c in ep["visited_cells"]:
            px, py = cell_center(c, grid)
            ax.add_patch(
                plt.Rectangle(
                    (px - grid.cell_size / 2.0, py - grid.cell_size / 2.0),
                    grid.cell_size,
                    grid.cell_size,
                    color="navy",
                    alpha=0.22,
                )
            )
        path = np.asarray(ep["path"], dtype=float)
        ax.plot(path[:, 0], path[:, 1], color="tab:orange", alpha=0.9, linewidth=1.1)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_COVERAGE_PNG)
    plt.close(fig)

    # Console summary.
    print("\nV8 Explore Per-map metrics:")
    for m in maps:
        name = m.name
        v = all_results[name]
        print(
            f"  {name}: observed={100*v['mean_observed_coverage_rate']:.2f}% | "
            f"visited={100*v['mean_visited_coverage_rate']:.2f}% | "
            f"collision={100*v['collision_rate']:.2f}% | timeout={100*v['timeout_rate']:.2f}% | "
            f"steps={v['mean_steps']:.1f}"
        )

    print("\nV8 Explore action distribution:")
    for a in ALL_ACTIONS:
        print(f"  {a}: {overall_action_distribution[a]['count']} ({100*overall_action_distribution[a]['rate']:.2f}%)")

    if v7_summary is not None:
        print("\nComparison vs v7 (goal-reaching baseline):")
        for row in comparison_table:
            print(
                f"  {row['map']}: v7 collision={100*row['v7_collision_rate']:.2f}% -> v8 {100*row['v8_collision_rate']:.2f}% | "
                f"v7 timeout={100*row['v7_timeout_rate']:.2f}% -> v8 {100*row['v8_timeout_rate']:.2f}% | "
                f"v7 path={row['v7_path_length_m']:.2f}m -> v8 {row['v8_path_length_m']:.2f}m | "
                f"v8 observed={100*row['v8_mean_observed_coverage']:.2f}% | v8 visited={100*row['v8_mean_visited_coverage']:.2f}% | "
                f"v7 resample={100*row['v7_resample_rate']:.2f}% | v8 resample={100*row['v8_resample_rate']:.2f}%"
            )

    print("\nAcceptance checks:")
    for group_name, group in acceptance.items():
        for key, ok in group.items():
            print(f"  {group_name}.{key}: {'PASS' if ok else 'FAIL'}")

    print(f"\nExploration assessment: {exploration_assessment}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {RESULT_PATHS_PNG}")
    print(f"Saved: {RESULT_ACTION_PNG}")
    print(f"Saved: {RESULT_COVERAGE_PNG}")


if __name__ == "__main__":
    main()
