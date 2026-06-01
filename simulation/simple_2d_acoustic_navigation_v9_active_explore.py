"""
Simple 2D acoustic navigation simulator v9 (active exploration).

v9 keeps hard collision safety but replaces passive stop-heavy behavior with:
- global frontier targeting + commitment
- anti-stop penalties/restrictions
- active recovery ordering
- map-specific exploration modes
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
RESULT_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v9_active_explore_results.json"
RESULT_PATHS_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v9_active_explore_paths.png"
RESULT_ACTION_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v9_active_explore_action_distribution.png"
RESULT_COVERAGE_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v9_active_explore_coverage_maps.png"
V8_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v8_explore_results.json"

EPISODES_PER_MAP = 100
MAX_STEPS = 500

ROBOT_RADIUS = 0.12
SAFETY_MARGIN_DEFAULT = 0.05
SAFETY_MARGIN_CLUTTER = 0.08
RAY_MAX_RANGE = 3.0
RAY_STEP = 0.10

COVERAGE_CELL_SIZE_M = 0.25
OBSERVED_COVERAGE_SUCCESS_THRESHOLD = 0.95
VISITED_COVERAGE_SUCCESS_THRESHOLD = 0.75

FRONT_SENSE_RANGE_M = 2.0
SIDE_SENSE_RANGE_M = 1.0
SENSE_FOV_DEG = 90.0
SENSE_RAY_COUNT = 7
SENSE_STEP_M = 0.12

# v9 scoring weights
W_NEW_OBS_GAIN = 4.0
W_FRONTIER_ALIGN = 3.0
W_FRONTIER_DIST_REDUCE = 2.0
W_HEADING_NOVELTY = 1.0
W_LOCAL_CLEARANCE = 0.8
W_REVISIT_PENALTY = 2.0

BASE_STOP_PENALTY = 2.0
REPEAT_STOP_PENALTY = 1.0
BASE_RESAMPLE_PENALTY = 0.8
REPEAT_RESAMPLE_PENALTY = 0.4
RESAMPLE_INFO_PENALTY = 0.2

STOP_SUPPRESS_AFTER = 2
FRONTIER_COMMIT_STEPS = 20
FRONTIER_STALL_STEPS = 25
BLOCKED_FRONTIER_COOLDOWN = 40
RECENT_CELL_WINDOW = 60

FAST_MIN_FRONT_CLEARANCE = 1.30
FAST_MIN_SIDE_FRONT_CLEARANCE = 1.00
SLOW_MIN_FRONT_CLEARANCE = 0.35
PROBE_MIN_FRONT_CLEARANCE = 0.20

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


def apply_action(x: float, y: float, heading: float, action: str, m: MapDef, turn_deg: float = 15.0) -> Tuple[float, float, float, float, bool]:
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
            [Rect(0.0, 0.0, 12.0, 2.2), Rect(0.0, 5.8, 12.0, 8.0), Rect(5.5, 2.2, 6.5, 4.0)],
        ),
        MapDef("single_block", 10.0, 8.0, [Rect(4.2, 2.8, 5.8, 5.2)]),
        MapDef("doorway", 12.0, 8.0, [Rect(5.7, 0.0, 6.3, 3.2), Rect(5.7, 4.8, 6.3, 8.0)]),
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

    dummy = GridDef(cell_size, nx, ny, free_mask, np.zeros((ny, nx), dtype=bool), set(), 0)
    start_cell = world_to_cell(start_xy[0], start_xy[1], dummy)
    reachable_mask = np.zeros((ny, nx), dtype=bool)
    reachable_cells: Set[Tuple[int, int]] = set()
    grid = GridDef(cell_size, nx, ny, free_mask, reachable_mask, reachable_cells, 0)

    if free_mask[start_cell[1], start_cell[0]]:
        q = deque([start_cell])
        reachable_mask[start_cell[1], start_cell[0]] = True
        reachable_cells.add(start_cell)
        while q:
            c = q.popleft()
            for nb in neighbors4(c, grid):
                if reachable_mask[nb[1], nb[0]] or (not free_mask[nb[1], nb[0]]):
                    continue
                reachable_mask[nb[1], nb[0]] = True
                reachable_cells.add(nb)
                q.append(nb)

    grid.total_reachable = max(1, len(reachable_cells))
    return grid


def trace_observed_cells_from_pose(m: MapDef, x: float, y: float, heading: float, grid: GridDef) -> Set[Tuple[int, int]]:
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
                c = world_to_cell(px, py, grid)
                if c in grid.reachable_cells:
                    observed.add(c)
                t += SENSE_STEP_M
    return observed


def find_frontier_cells(observed_cells: Set[Tuple[int, int]], grid: GridDef) -> Set[Tuple[int, int]]:
    frontier: Set[Tuple[int, int]] = set()
    for c in observed_cells:
        if c not in grid.reachable_cells:
            continue
        for nb in neighbors4(c, grid):
            if nb in grid.reachable_cells and nb not in observed_cells:
                frontier.add(c)
                break
    return frontier


def unobserved_neighbor_count(cell: Tuple[int, int], observed_cells: Set[Tuple[int, int]], grid: GridDef) -> int:
    return sum(1 for nb in neighbors4(cell, grid) if nb in grid.reachable_cells and nb not in observed_cells)


def cluster_frontier_cells(frontier_cells: Set[Tuple[int, int]], grid: GridDef) -> List[List[Tuple[int, int]]]:
    clusters = []
    rem = set(frontier_cells)
    while rem:
        start = rem.pop()
        comp = [start]
        q = deque([start])
        while q:
            c = q.popleft()
            for nb in neighbors4(c, grid):
                if nb in rem:
                    rem.remove(nb)
                    comp.append(nb)
                    q.append(nb)
        clusters.append(comp)
    return clusters


def select_global_frontier_target(
    x: float,
    y: float,
    observed_cells: Set[Tuple[int, int]],
    frontier_cells: Set[Tuple[int, int]],
    grid: GridDef,
    recent_path_cells: List[Tuple[int, int]],
    blocked_targets: Dict[Tuple[int, int], int],
    prefer_far_from_recent: bool,
) -> Optional[Tuple[int, int]]:
    if not frontier_cells:
        return None

    clusters = cluster_frontier_cells(frontier_cells, grid)
    best_target = None
    best_score = -1e18
    recent_centers = [cell_center(c, grid) for c in recent_path_cells]

    for comp in clusters:
        rep = max(comp, key=lambda c: unobserved_neighbor_count(c, observed_cells, grid))
        if blocked_targets.get(rep, 0) > 0:
            continue
        rx, ry = cell_center(rep, grid)
        unobs = float(sum(unobserved_neighbor_count(c, observed_cells, grid) for c in comp) / max(1, len(comp)))
        d_agent = math.hypot(rx - x, ry - y)
        if recent_centers:
            d_recent = min(math.hypot(rx - cx, ry - cy) for (cx, cy) in recent_centers)
        else:
            d_recent = 0.0
        score = 2.0 * unobs + 1.0 * d_recent - 0.4 * d_agent
        if prefer_far_from_recent:
            score += 0.8 * d_recent
        if score > best_score:
            best_score = score
            best_target = rep
    return best_target


def heading_novelty(next_cell: Tuple[int, int], next_heading: float, heading_memory: Dict[Tuple[int, int], deque]) -> float:
    heading_deg = (math.degrees(next_heading) + 360.0) % 360.0
    hbin = int(heading_deg // 30.0)
    hist = heading_memory.get(next_cell)
    if not hist:
        return 1.0
    repeats = sum(1 for v in hist if v == hbin)
    return 1.0 / (1.0 + repeats)


def local_clearance_score(m: MapDef, x: float, y: float, heading: float) -> float:
    d = sector_true_distances(m, x, y, heading)
    return float(np.clip(min(d.values()) / 1.5, 0.0, 1.0))


def get_context_mode(m: MapDef) -> str:
    if m.name == "empty_room":
        return "open_space"
    if m.name == "corridor":
        return "corridor"
    if m.name in {"doorway", "single_block"}:
        return "doorway_single_block"
    return "cluttered_room"


def choose_clearer_side_by_true_distance(true_dist: Dict[str, float]) -> str:
    return ACTION_LEFT if true_dist["left"] >= true_dist["right"] else ACTION_RIGHT


def safe_margin_for_map(m: MapDef) -> float:
    return SAFETY_MARGIN_CLUTTER if m.name == "cluttered_room" else SAFETY_MARGIN_DEFAULT


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


def run_episode(m: MapDef, rng: np.random.Generator) -> Dict[str, object]:
    sx, sy = sample_free_pose(m, rng)
    gx, gy = sample_free_pose(m, rng)  # plotted only
    x, y = sx, sy
    heading = rng.uniform(-math.pi, math.pi)
    grid = build_grid_def(m, (sx, sy), COVERAGE_CELL_SIZE_M)
    context = get_context_mode(m)
    margin = safe_margin_for_map(m)

    visited_cells: Set[Tuple[int, int]] = set()
    observed_cells: Set[Tuple[int, int]] = set()
    path = [(x, y)]
    actions = Counter()
    path_len = 0.0
    collision = False
    success = False

    heading_memory: Dict[Tuple[int, int], deque] = {}
    recent_cells = deque(maxlen=RECENT_CELL_WINDOW)
    recent_path_cells: List[Tuple[int, int]] = []

    frontier_target: Optional[Tuple[int, int]] = None
    frontier_commit_left = 0
    frontier_stall_count = 0
    frontier_prev_dist: Optional[float] = None
    blocked_targets: Dict[Tuple[int, int], int] = {}

    sweep_heading: Optional[float] = None
    sweep_turn_pending = 0
    doorway_lock_steps = 0
    circumnav_side: Optional[str] = None
    circumnav_steps = 0

    consecutive_resample_count = 0
    consecutive_stop_reverse_count = 0
    resample_without_new_info_count = 0

    revisit_steps = 0
    new_cells_total = 0
    frontier_count_remaining = 0
    stuck_recovery_count = 0
    scan_turn_count = 0

    for _step in range(1, MAX_STEPS + 1):
        # decrement blocked frontier cooldown
        for k in list(blocked_targets.keys()):
            blocked_targets[k] -= 1
            if blocked_targets[k] <= 0:
                del blocked_targets[k]

        true_dist = sector_true_distances(m, x, y, heading)
        sec = {s: synthesize_sector_perception(true_dist[s], rng) for s in SECTOR_NAMES}
        fl = str(sec["front_left"]["predicted_state"]).upper()
        f = str(sec["front"]["predicted_state"]).upper()
        fr = str(sec["front_right"]["predicted_state"]).upper()
        l = str(sec["left"]["predicted_state"]).upper()
        r = str(sec["right"]["predicted_state"]).upper()
        any_front_obstacle_pred = (fl == "OBSTACLE") or (f == "OBSTACLE") or (fr == "OBSTACLE")

        # observe current pose
        obs_before = len(observed_cells)
        cur_cell = world_to_cell(x, y, grid)
        if cur_cell in grid.reachable_cells:
            visited_cells.add(cur_cell)
        observed_cells.update(trace_observed_cells_from_pose(m, x, y, heading, grid))
        newly_observed_this_step = len(observed_cells) - obs_before
        new_cells_total += max(0, newly_observed_this_step)

        obs_rate = len(observed_cells) / grid.total_reachable
        vis_rate = len(visited_cells) / grid.total_reachable
        if (obs_rate >= OBSERVED_COVERAGE_SUCCESS_THRESHOLD) and (vis_rate >= VISITED_COVERAGE_SUCCESS_THRESHOLD):
            success = True
            break

        frontier_cells = find_frontier_cells(observed_cells, grid)
        frontier_count_remaining = len(frontier_cells)

        recent_cells.append(cur_cell)
        recent_path_cells.append(cur_cell)
        if len(recent_path_cells) > RECENT_CELL_WINDOW:
            recent_path_cells = recent_path_cells[-RECENT_CELL_WINDOW:]
        if len(recent_cells) >= 20:
            revisit_ratio_recent = 1.0 - (len(set(recent_cells)) / max(1, len(recent_cells)))
        else:
            revisit_ratio_recent = 0.0
        local_loop = revisit_ratio_recent > 0.5

        need_new_target = frontier_target is None or frontier_commit_left <= 0 or frontier_target not in frontier_cells
        if frontier_target is not None:
            tx, ty = cell_center(frontier_target, grid)
            dcur = math.hypot(tx - x, ty - y)
            if frontier_prev_dist is not None and (frontier_prev_dist - dcur) < 0.01:
                frontier_stall_count += 1
            else:
                frontier_stall_count = 0
            frontier_prev_dist = dcur
            if frontier_stall_count >= FRONTIER_STALL_STEPS:
                blocked_targets[frontier_target] = BLOCKED_FRONTIER_COOLDOWN
                need_new_target = True
                frontier_stall_count = 0
        if need_new_target:
            frontier_target = select_global_frontier_target(
                x, y, observed_cells, frontier_cells, grid, recent_path_cells, blocked_targets, local_loop
            )
            frontier_commit_left = FRONTIER_COMMIT_STEPS if frontier_target is not None else 0
            frontier_prev_dist = None
        else:
            frontier_commit_left -= 1

        heading_to_frontier = None
        frontier_dist_cur = None
        if frontier_target is not None:
            tx, ty = cell_center(frontier_target, grid)
            heading_to_frontier = math.atan2(ty - y, tx - x)
            frontier_dist_cur = math.hypot(tx - x, ty - y)

        corridor_mode = (
            (true_dist["left"] < 0.9)
            and (true_dist["right"] < 0.9)
            and (abs(true_dist["left"] - true_dist["right"]) < 0.2)
            and (true_dist["front"] > 0.55)
        )
        doorway_mode = (
            (m.name == "doorway")
            and (true_dist["front"] > 0.55)
            and ((true_dist["front_left"] < 0.8) or (true_dist["front_right"] < 0.8))
            and ((true_dist["left"] > 0.9) or (true_dist["right"] > 0.9))
        )

        candidates: List[str] = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
        if context == "open_space":
            if sweep_heading is None:
                sweep_heading = heading
            if true_dist["front"] < 0.55 or any_front_obstacle_pred:
                sweep_turn_pending = 1
                sweep_heading = wrap_angle(sweep_heading + math.radians(90.0))
            if sweep_turn_pending > 0:
                td = ACTION_LEFT if wrap_angle(sweep_heading - heading) > 0 else ACTION_RIGHT
                candidates = [td, ACTION_PROBE, ACTION_SLOW, ACTION_RESAMPLE, ACTION_STOP]
            else:
                candidates = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
        elif context == "corridor":
            if corridor_mode and not any_front_obstacle_pred:
                candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_RESAMPLE, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP]
        elif m.name == "doorway":
            if doorway_mode:
                doorway_lock_steps = max(doorway_lock_steps, 6)
            if doorway_lock_steps > 0 and not any_front_obstacle_pred:
                candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_RESAMPLE, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP]
                doorway_lock_steps -= 1
        elif m.name == "single_block":
            if f == "OBSTACLE" and circumnav_side is None:
                circumnav_side = choose_clearer_side_by_true_distance(true_dist)
                circumnav_steps = 20
            if circumnav_side is not None and circumnav_steps > 0:
                other = ACTION_RIGHT if circumnav_side == ACTION_LEFT else ACTION_LEFT
                candidates = [circumnav_side, ACTION_PROBE, ACTION_SLOW, other, ACTION_RESAMPLE, ACTION_STOP]
                circumnav_steps -= 1
            elif circumnav_steps <= 0:
                circumnav_side = None
        else:
            candidates = [ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_SLOW, ACTION_RESAMPLE, ACTION_STOP]

        if any_front_obstacle_pred:
            if f == "OBSTACLE":
                if (l == "CLEAR") and (r != "CLEAR"):
                    candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
                elif (r == "CLEAR") and (l != "CLEAR"):
                    candidates = [ACTION_RIGHT, ACTION_LEFT, ACTION_RESAMPLE, ACTION_STOP]
                elif (l == "CLEAR") and (r == "CLEAR"):
                    clearer = choose_clearer_side_by_true_distance(true_dist)
                    other = ACTION_RIGHT if clearer == ACTION_LEFT else ACTION_LEFT
                    candidates = [clearer, other, ACTION_RESAMPLE, ACTION_STOP]
                else:
                    candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
            elif (fl == "OBSTACLE") and (fr == "CLEAR"):
                candidates = [ACTION_RIGHT, ACTION_LEFT, ACTION_RESAMPLE, ACTION_STOP]
            elif (fr == "OBSTACLE") and (fl == "CLEAR"):
                candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
            else:
                candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]

        safe_candidates: List[str] = []
        for a in candidates:
            if a in FORWARD_ACTIONS:
                if any_front_obstacle_pred:
                    continue
                if not valid_forward_by_clearance(a, true_dist):
                    continue
                if predict_forward_collision(m, x, y, heading, forward_step_for_action(a), margin):
                    continue
            elif a == ACTION_STOP:
                if predict_forward_collision(m, x, y, heading, -0.03, margin):
                    continue
            safe_candidates.append(a)
        if not safe_candidates:
            safe_candidates = [ACTION_RESAMPLE]

        cap = RESAMPLE_CAPS[context]
        if consecutive_resample_count >= cap:
            non_resample = [a for a in safe_candidates if a != ACTION_RESAMPLE]
            if non_resample:
                safe_candidates = non_resample

        non_stop_safe = [a for a in safe_candidates if a != ACTION_STOP]
        if consecutive_stop_reverse_count >= STOP_SUPPRESS_AFTER and non_stop_safe:
            safe_candidates = non_stop_safe

        best_action = safe_candidates[0]
        best_score = -1e18
        dyn_revisit = W_REVISIT_PENALTY * (1.4 if local_loop else 1.0)
        dyn_frontier_align = W_FRONTIER_ALIGN * (1.5 if local_loop else 1.0)

        for a in safe_candidates:
            nx, ny, nh, _, collided = apply_action(x, y, heading, a, m, turn_deg=15.0)
            if collided:
                continue
            nc = world_to_cell(nx, ny, grid)
            revisit_penalty = 1.0 if nc in visited_cells else 0.0

            new_obs_gain = 0.0
            if nc not in observed_cells:
                new_obs_gain += 1.0
            new_obs_gain += 0.3 * unobserved_neighbor_count(nc, observed_cells, grid)
            if a in {ACTION_SLOW, ACTION_PROBE}:
                new_obs_gain += 0.4
            if a in {ACTION_LEFT, ACTION_RIGHT}:
                new_obs_gain += 0.15

            align_gain = 0.0
            dist_reduce = 0.0
            if frontier_target is not None and heading_to_frontier is not None and frontier_dist_cur is not None:
                err = abs(wrap_angle(heading_to_frontier - nh))
                align_gain = math.cos(err)
                tx, ty = cell_center(frontier_target, grid)
                dnext = math.hypot(tx - nx, ty - ny)
                dist_reduce = (frontier_dist_cur - dnext) / max(0.25, grid.cell_size)

            novelty = heading_novelty(nc, nh, heading_memory)
            clearance = local_clearance_score(m, nx, ny, nh)

            stop_penalty = 0.0
            if a == ACTION_STOP:
                stop_penalty = BASE_STOP_PENALTY + consecutive_stop_reverse_count * REPEAT_STOP_PENALTY
                if non_stop_safe:
                    stop_penalty += 4.0
            passive_penalty = 0.0
            if a == ACTION_RESAMPLE:
                if newly_observed_this_step > 0:
                    passive_penalty = RESAMPLE_INFO_PENALTY
                else:
                    passive_penalty = 1.0 + 0.5 * resample_without_new_info_count
                passive_penalty += BASE_RESAMPLE_PENALTY + consecutive_resample_count * REPEAT_RESAMPLE_PENALTY

            map_bonus = 0.0
            if context == "open_space":
                if a == ACTION_SLOW and true_dist["front"] > 0.9:
                    map_bonus += 0.8
                if a == ACTION_PROBE and true_dist["front"] > 0.9:
                    map_bonus -= 0.2
                if a == ACTION_STOP:
                    map_bonus -= 1.0
            elif context == "corridor":
                if corridor_mode and a in {ACTION_PROBE, ACTION_SLOW}:
                    map_bonus += 0.7
                if corridor_mode and a in {ACTION_LEFT, ACTION_RIGHT}:
                    map_bonus -= 0.5
            elif m.name == "doorway":
                if doorway_lock_steps > 0 and a in {ACTION_PROBE, ACTION_SLOW}:
                    map_bonus += 0.8
                if doorway_lock_steps > 0 and a in {ACTION_LEFT, ACTION_RIGHT, ACTION_STOP}:
                    map_bonus -= 0.8
            elif m.name == "single_block":
                if circumnav_side is not None and a == circumnav_side:
                    map_bonus += 0.6
            elif context == "cluttered_room":
                if a == ACTION_PROBE:
                    map_bonus += 0.5
                if a == ACTION_SLOW and min(true_dist.values()) < 0.5:
                    map_bonus -= 0.4

            score = (
                W_NEW_OBS_GAIN * new_obs_gain
                + dyn_frontier_align * align_gain
                + W_FRONTIER_DIST_REDUCE * dist_reduce
                + W_HEADING_NOVELTY * novelty
                + W_LOCAL_CLEARANCE * clearance
                - dyn_revisit * revisit_penalty
                - stop_penalty
                - passive_penalty
                + map_bonus
            )
            if score > best_score:
                best_score = score
                best_action = a

        action = best_action
        if action == ACTION_STOP and non_stop_safe:
            # Active recovery order: turn-to-frontier, probe, slow, other safe.
            turn_opts = [a for a in non_stop_safe if a in {ACTION_LEFT, ACTION_RIGHT}]
            if turn_opts and frontier_target is not None and heading_to_frontier is not None:
                left_err = abs(wrap_angle(heading_to_frontier - wrap_angle(heading + math.radians(15.0))))
                right_err = abs(wrap_angle(heading_to_frontier - wrap_angle(heading - math.radians(15.0))))
                action = ACTION_LEFT if left_err <= right_err else ACTION_RIGHT
                if action not in turn_opts:
                    action = turn_opts[0]
            elif ACTION_PROBE in non_stop_safe:
                action = ACTION_PROBE
            elif ACTION_SLOW in non_stop_safe:
                action = ACTION_SLOW
            else:
                action = non_stop_safe[0]

        if sweep_turn_pending > 0 and action in {ACTION_LEFT, ACTION_RIGHT}:
            sweep_turn_pending -= 1
            stuck_recovery_count += 1
            scan_turn_count += 1

        if action in FORWARD_ACTIONS:
            if any_front_obstacle_pred or predict_forward_collision(m, x, y, heading, forward_step_for_action(action), margin):
                action = ACTION_RESAMPLE

        actions[action] += 1
        nx, ny, nhead, moved, collided = apply_action(x, y, heading, action, m, turn_deg=15.0)
        path_len += moved

        obs_before_act = len(observed_cells)
        x, y, heading = nx, ny, nhead
        path.append((x, y))
        observed_cells.update(trace_observed_cells_from_pose(m, x, y, heading, grid))
        newly_observed_from_action = len(observed_cells) - obs_before_act
        new_cells_total += max(0, newly_observed_from_action)

        next_cell = world_to_cell(x, y, grid)
        if next_cell in visited_cells:
            revisit_steps += 1
        if next_cell in grid.reachable_cells:
            visited_cells.add(next_cell)

        hdeg = (math.degrees(heading) + 360.0) % 360.0
        hbin = int(hdeg // 30.0)
        if next_cell not in heading_memory:
            heading_memory[next_cell] = deque(maxlen=12)
        heading_memory[next_cell].append(hbin)

        if action == ACTION_RESAMPLE:
            consecutive_resample_count += 1
            if newly_observed_this_step > 0 or newly_observed_from_action > 0:
                resample_without_new_info_count = 0
            else:
                resample_without_new_info_count += 1
        else:
            consecutive_resample_count = 0
            resample_without_new_info_count = max(0, resample_without_new_info_count - 1)

        if action == ACTION_STOP:
            consecutive_stop_reverse_count += 1
        else:
            consecutive_stop_reverse_count = 0

        if collided:
            collision = True
            break

    timeout = (not success) and (not collision)
    final_observed_rate = len(observed_cells) / grid.total_reachable
    final_visited_rate = len(visited_cells) / grid.total_reachable
    steps = max(1, len(path) - 1)
    return {
        "success": bool(success),
        "collision": bool(collision),
        "timeout": bool(timeout),
        "steps": int(len(path) - 1),
        "path_length": float(path_len),
        "action_counts": dict(actions),
        "start": (float(sx), float(sy)),
        "goal": (float(gx), float(gy)),
        "path": path,
        "final_observed_coverage_rate": float(final_observed_rate),
        "final_visited_coverage_rate": float(final_visited_rate),
        "observed_coverage_success_rate_50": bool(final_observed_rate >= 0.50),
        "observed_coverage_success_rate_75": bool(final_observed_rate >= 0.75),
        "observed_coverage_success_rate_90": bool(final_observed_rate >= 0.90),
        "observed_coverage_success_rate_95": bool(final_observed_rate >= 0.95),
        "new_cells_total": int(new_cells_total),
        "revisit_ratio": float(revisit_steps / steps),
        "frontier_count_remaining": int(frontier_count_remaining),
        "stuck_recovery_count": int(stuck_recovery_count),
        "scan_turn_count": int(scan_turn_count),
        "stop_reverse_rate": float(actions.get(ACTION_STOP, 0) / steps),
        "probe_plus_slow_rate": float((actions.get(ACTION_PROBE, 0) + actions.get(ACTION_SLOW, 0)) / steps),
        "observed_cells": observed_cells,
        "visited_cells": visited_cells,
        "grid": grid,
    }


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
        c50 = sum(int(e["observed_coverage_success_rate_50"]) for e in episodes)
        c75 = sum(int(e["observed_coverage_success_rate_75"]) for e in episodes)
        c90 = sum(int(e["observed_coverage_success_rate_90"]) for e in episodes)
        c95 = sum(int(e["observed_coverage_success_rate_95"]) for e in episodes)

        map_dist = _action_distribution(map_actions)
        map_action_dist[m.name] = map_dist
        all_results[m.name] = {
            "mean_observed_coverage_rate": float(np.mean([e["final_observed_coverage_rate"] for e in episodes])),
            "mean_visited_coverage_rate": float(np.mean([e["final_visited_coverage_rate"] for e in episodes])),
            "observed_coverage_success_rate_50": c50 / EPISODES_PER_MAP,
            "observed_coverage_success_rate_75": c75 / EPISODES_PER_MAP,
            "observed_coverage_success_rate_90": c90 / EPISODES_PER_MAP,
            "observed_coverage_success_rate_95": c95 / EPISODES_PER_MAP,
            "collision_rate": coll / EPISODES_PER_MAP,
            "timeout_rate": tout / EPISODES_PER_MAP,
            "mean_steps": float(np.mean([e["steps"] for e in episodes])),
            "mean_path_length_m": float(np.mean([e["path_length"] for e in episodes])),
            "mean_new_cells_per_episode": float(np.mean([e["new_cells_total"] for e in episodes])),
            "mean_revisit_ratio": float(np.mean([e["revisit_ratio"] for e in episodes])),
            "mean_frontier_count_remaining": float(np.mean([e["frontier_count_remaining"] for e in episodes])),
            "stop_reverse_rate": float(map_dist[ACTION_STOP]["rate"]),
            "probe_plus_slow_rate": float(map_dist[ACTION_PROBE]["rate"] + map_dist[ACTION_SLOW]["rate"]),
            "mean_stuck_recovery_count": float(np.mean([e["stuck_recovery_count"] for e in episodes])),
        }

        sel_idx = np.linspace(0, EPISODES_PER_MAP - 1, 10, dtype=int)
        path_samples[m.name] = [episodes[i] for i in sel_idx]
        coverage_samples[m.name] = episodes[sel_idx[-1]]

    overall_action_distribution = _action_distribution(all_actions)

    v8_summary = json.loads(V8_JSON.read_text(encoding="utf-8")) if V8_JSON.exists() else None
    comparison_vs_v8 = []
    for m in maps:
        name = m.name
        row = {
            "map": name,
            "v8_observed_coverage": None,
            "v9_observed_coverage": all_results[name]["mean_observed_coverage_rate"],
            "v8_visited_coverage": None,
            "v9_visited_coverage": all_results[name]["mean_visited_coverage_rate"],
            "v8_collision_rate": None,
            "v9_collision_rate": all_results[name]["collision_rate"],
            "v8_stop_reverse_rate": None,
            "v9_stop_reverse_rate": all_results[name]["stop_reverse_rate"],
            "v8_path_length": None,
            "v9_path_length": all_results[name]["mean_path_length_m"],
        }
        if v8_summary is not None:
            if "maps" in v8_summary and name in v8_summary["maps"]:
                row["v8_observed_coverage"] = float(v8_summary["maps"][name].get("mean_observed_coverage_rate", float("nan")))
                row["v8_visited_coverage"] = float(v8_summary["maps"][name].get("mean_visited_coverage_rate", float("nan")))
                row["v8_collision_rate"] = float(v8_summary["maps"][name].get("collision_rate", float("nan")))
                row["v8_path_length"] = float(v8_summary["maps"][name].get("mean_path_length_m", float("nan")))
            if "map_action_distribution" in v8_summary and name in v8_summary["map_action_distribution"]:
                row["v8_stop_reverse_rate"] = float(v8_summary["map_action_distribution"][name].get(ACTION_STOP, {}).get("rate", float("nan")))
        comparison_vs_v8.append(row)

    overall_collision_rate = float(np.mean([all_results[m.name]["collision_rate"] for m in maps]))
    overall_stop_rate = float(overall_action_distribution[ACTION_STOP]["rate"])
    overall_probe_slow_rate = float(overall_action_distribution[ACTION_PROBE]["rate"] + overall_action_distribution[ACTION_SLOW]["rate"])
    overall_resample_rate = float(overall_action_distribution[ACTION_RESAMPLE]["rate"])

    acceptance = {
        "primary_behavior": {
            "stop_reverse_le_0.10": overall_stop_rate <= 0.10,
            "probe_plus_slow_ge_0.35": overall_probe_slow_rate >= 0.35,
            "resample_le_0.10": overall_resample_rate <= 0.10,
            "overall_collision_le_0.02": overall_collision_rate <= 0.02,
        },
        "coverage": {
            "empty_room_observed_ge_0.80": all_results["empty_room"]["mean_observed_coverage_rate"] >= 0.80,
            "corridor_observed_ge_0.80": all_results["corridor"]["mean_observed_coverage_rate"] >= 0.80,
            "single_block_observed_ge_0.70": all_results["single_block"]["mean_observed_coverage_rate"] >= 0.70,
            "doorway_observed_ge_0.60": all_results["doorway"]["mean_observed_coverage_rate"] >= 0.60,
            "cluttered_observed_ge_0.30": all_results["cluttered_room"]["mean_observed_coverage_rate"] >= 0.30,
        },
        "safety": {
            "cluttered_collision_le_0.02": all_results["cluttered_room"]["collision_rate"] <= 0.02,
            "corridor_collision_le_0.02": all_results["corridor"]["collision_rate"] <= 0.02,
        },
    }
    accepted = all(acceptance["primary_behavior"].values()) and all(acceptance["coverage"].values()) and all(acceptance["safety"].values())
    final_assessment = "accepted" if accepted else "needs_more_tuning"

    results = {
        "episodes_per_map": EPISODES_PER_MAP,
        "max_steps": MAX_STEPS,
        "coverage_cell_size_m": COVERAGE_CELL_SIZE_M,
        "maps": all_results,
        "map_action_distribution": map_action_dist,
        "overall_action_distribution": overall_action_distribution,
        "total_stuck_recovery_count": int(total_stuck_recoveries),
        "total_scan_turn_count": int(total_scan_turns),
        "comparison_vs_v8": comparison_vs_v8,
        "acceptance": acceptance,
        "final_assessment": final_assessment,
    }
    RESULT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.flatten()
    for i, m in enumerate(maps):
        ax = axes[i]
        draw_map(ax, m)
        ax.set_title(m.name)
        for ep in path_samples[m.name]:
            path = np.asarray(ep["path"], dtype=float)
            color = "tab:green" if ep["observed_coverage_success_rate_75"] else ("tab:red" if ep["collision"] else "tab:orange")
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.5, linewidth=1.1)
            sx, sy = ep["start"]
            gx, gy = ep["goal"]
            ax.scatter([sx], [sy], c="blue", s=10)
            ax.scatter([gx], [gy], c="black", s=10, marker="x")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_PATHS_PNG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=140)
    vals = [100.0 * overall_action_distribution[a]["rate"] for a in ALL_ACTIONS]
    bars = ax.bar(ALL_ACTIONS, vals, color="tab:blue")
    ax.set_ylabel("Action share (%)")
    ax.set_title("Simple 2D Acoustic Nav v9 Active Explore: Action Distribution")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_ACTION_PNG)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.flatten()
    for i, m in enumerate(maps):
        ax = axes[i]
        draw_map(ax, m)
        ax.set_title(f"{m.name} coverage")
        ep = coverage_samples[m.name]
        grid: GridDef = ep["grid"]
        for c in grid.reachable_cells:
            px, py = cell_center(c, grid)
            ax.add_patch(plt.Rectangle((px - grid.cell_size / 2.0, py - grid.cell_size / 2.0), grid.cell_size, grid.cell_size, color="lightgray", alpha=0.12))
        for c in ep["observed_cells"]:
            px, py = cell_center(c, grid)
            ax.add_patch(plt.Rectangle((px - grid.cell_size / 2.0, py - grid.cell_size / 2.0), grid.cell_size, grid.cell_size, color="deepskyblue", alpha=0.20))
        for c in ep["visited_cells"]:
            px, py = cell_center(c, grid)
            ax.add_patch(plt.Rectangle((px - grid.cell_size / 2.0, py - grid.cell_size / 2.0), grid.cell_size, grid.cell_size, color="navy", alpha=0.22))
        p = np.asarray(ep["path"], dtype=float)
        ax.plot(p[:, 0], p[:, 1], color="tab:orange", alpha=0.9, linewidth=1.1)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_COVERAGE_PNG)
    plt.close(fig)

    print("\nV9 per-map coverage/collision/stop:")
    for m in maps:
        v = all_results[m.name]
        print(
            f"  {m.name}: observed={100*v['mean_observed_coverage_rate']:.2f}% | visited={100*v['mean_visited_coverage_rate']:.2f}% | "
            f"collision={100*v['collision_rate']:.2f}% | timeout={100*v['timeout_rate']:.2f}% | stop={100*v['stop_reverse_rate']:.2f}%"
        )

    print("\nV9 overall action distribution:")
    for a in ALL_ACTIONS:
        print(f"  {a}: {overall_action_distribution[a]['count']} ({100*overall_action_distribution[a]['rate']:.2f}%)")

    print("\nAcceptance checks:")
    for gname, group in acceptance.items():
        for key, ok in group.items():
            print(f"  {gname}.{key}: {'PASS' if ok else 'FAIL'}")
    print(f"\nFinal assessment: {final_assessment}")
    print(f"Saved: {RESULT_JSON}")
    print(f"Saved: {RESULT_PATHS_PNG}")
    print(f"Saved: {RESULT_ACTION_PNG}")
    print(f"Saved: {RESULT_COVERAGE_PNG}")


if __name__ == "__main__":
    main()

