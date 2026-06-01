"""
Simple 2D acoustic navigation simulator v11 (doorway + clutter tuning).

This variant keeps v10 safety behavior and curriculum interface, and applies
targeted improvements for doorway crossing and cluttered-room recovery.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("simulation/results")
RESULT_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v11_doorway_clutter_tuning_results.json"
RESULT_ACTION_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v11_doorway_clutter_tuning_action_distribution.png"
RESULT_PATHS_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v11_doorway_clutter_tuning_paths.png"
RESULT_COVERAGE_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v11_doorway_clutter_tuning_coverage_maps.png"
RESULT_DIFF_SUMMARY_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v11_doorway_clutter_tuning_difficulty_summary.png"
V9_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v9_active_explore_results.json"

EPISODES_PER_MAP = 100
MAX_STEPS = 700

ROBOT_RADIUS = 0.12
SAFETY_MARGIN_DEFAULT = 0.05
SAFETY_MARGIN_CLUTTER = 0.08

RAY_MAX_RANGE = 3.0
RAY_STEP = 0.10

COVERAGE_CELL_SIZE_M = 0.25
OBSERVED_COVERAGE_SUCCESS_THRESHOLD = 0.95
VISITED_COVERAGE_SUCCESS_THRESHOLD = 0.75

# v10/v11 sensing defaults (clean/mild/medium).
FRONT_SENSE_RANGE_DEFAULT = 2.5
SIDE_SENSE_RANGE_DEFAULT = 1.2
SENSE_FOV_DEG_DEFAULT = 100.0
SENSE_RAY_COUNT_DEFAULT = 7
SENSE_STEP_DEFAULT = 0.12

# Hard mode may intentionally reduce effective sensing.
FRONT_SENSE_RANGE_HARD = 2.0
SIDE_SENSE_RANGE_HARD = 1.0
SENSE_FOV_DEG_HARD = 90.0
SENSE_RAY_COUNT_HARD = 7
SENSE_STEP_HARD = 0.12

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

ACTION_FAST = "MOVE_FORWARD_FAST"
ACTION_SLOW = "MOVE_FORWARD_SLOW"
ACTION_PROBE = "PROBE_FORWARD"
ACTION_LEFT = "TURN_LEFT"
ACTION_RIGHT = "TURN_RIGHT"
ACTION_RESAMPLE = "SLOW_DOWN_AND_RESAMPLE"
ACTION_STOP = "STOP_OR_REVERSE"
ALL_ACTIONS = [ACTION_FAST, ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
FORWARD_ACTIONS = {ACTION_FAST, ACTION_SLOW, ACTION_PROBE}

# v10/v11 scoring.
W_NEW_OBS_GAIN = 5.0
W_FRONTIER_CLUSTER_ALIGN = 4.0
W_FRONTIER_DIST_REDUCE = 3.0
W_SWEEP_PROGRESS = 2.0
W_HEADING_NOVELTY = 1.5
W_LOCAL_CLEARANCE = 1.0
W_REVISIT_PENALTY = 3.0
W_STOP_PENALTY = 5.0
W_PASSIVE_PENALTY = 2.0

# Anti-stop/passive settings.
BASE_STOP_PENALTY = 2.0
REPEAT_STOP_PENALTY = 1.0
BASE_RESAMPLE_PENALTY = 0.8
REPEAT_RESAMPLE_PENALTY = 0.5
RESAMPLE_INFO_PENALTY = 0.2
STOP_SUPPRESS_AFTER = 2

# Frontier and loop memory.
FRONTIER_COMMIT_STEPS = 30
FRONTIER_STALL_STEPS = 30
BLOCKED_FRONTIER_COOLDOWN = 45
RECENT_CELL_WINDOW = 80
LOOP_RATIO_THRESHOLD = 0.55

# Safety gating.
FAST_MIN_FRONT_CLEARANCE = 1.35
FAST_MIN_SIDE_FRONT_CLEARANCE = 1.05
SLOW_MIN_FRONT_CLEARANCE = 0.35
PROBE_MIN_FRONT_CLEARANCE = 0.20

RESAMPLE_CAPS = {
    "open_space": 2,
    "corridor": 3,
    "doorway_single_block": 3,
    "cluttered_room": 6,
}

# v11 targeted tuning knobs.
DOORWAY_COMMIT_STEPS = 14
DOORWAY_ALIGN_ERR_RAD = math.radians(25.0)
DOORWAY_OSC_WINDOW = 8
DOORWAY_OSC_TURN_THRESHOLD = 5

CLUTTER_STAGNATION_STEPS = 24
CLUTTER_RECOVERY_STEPS = 12

# Selectable acoustic difficulty presets.
DIFFICULTY_PRESETS: Dict[str, Dict[str, float]] = {
    "clean": {
        "distance_noise_std": 0.0,
        "missed_echo_prob": 0.0,
        "false_echo_prob": 0.0,
        "echo_jitter_std": 0.0,
        "max_range_dropout_prob": 0.0,
        "obstacle_confidence_noise": 0.0,
    },
    "mild_noise": {
        "distance_noise_std": 0.03,
        "missed_echo_prob": 0.03,
        "false_echo_prob": 0.01,
        "echo_jitter_std": 0.02,
        "max_range_dropout_prob": 0.01,
        "obstacle_confidence_noise": 0.03,
    },
    "medium_noise": {
        "distance_noise_std": 0.08,
        "missed_echo_prob": 0.08,
        "false_echo_prob": 0.03,
        "echo_jitter_std": 0.05,
        "max_range_dropout_prob": 0.04,
        "obstacle_confidence_noise": 0.08,
    },
    # Hard uses v9-like strongest behavior plus extra dropouts/jitter.
    "hard_noise": {
        "distance_noise_std": 0.13,
        "missed_echo_prob": 0.12,
        "false_echo_prob": 0.06,
        "echo_jitter_std": 0.08,
        "max_range_dropout_prob": 0.08,
        "obstacle_confidence_noise": 0.12,
    },
}


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


def apply_perception_noise(
    true_d: float,
    rng: np.random.Generator,
    preset: Dict[str, float],
) -> float:
    d = true_d + rng.normal(0.0, preset["distance_noise_std"])
    d += rng.normal(0.0, preset["echo_jitter_std"])
    d = float(np.clip(d, 0.03, RAY_MAX_RANGE))
    if rng.random() < preset["max_range_dropout_prob"]:
        d = RAY_MAX_RANGE
    return d


def synthesize_sector_perception(
    true_d: float,
    rng: np.random.Generator,
    preset: Dict[str, float],
) -> Dict[str, float | str | int]:
    d = apply_perception_noise(true_d, rng, preset)

    if d < 0.45:
        p_obs, p_unc = 0.96, 0.04
    elif d < 0.80:
        p_obs, p_unc = 0.78, 0.20
    elif d < 1.20:
        p_obs, p_unc = 0.42, 0.42
    elif d < 1.80:
        p_obs, p_unc = 0.18, 0.52
    else:
        p_obs, p_unc = 0.05, 0.45

    # Difficulty-controlled missed and false echoes.
    p_obs = max(0.0, p_obs * (1.0 - preset["missed_echo_prob"]))
    p_false = preset["false_echo_prob"]

    # Confidence noise perturbs probabilities.
    conf_noise = rng.normal(0.0, preset["obstacle_confidence_noise"])
    p_obs = float(np.clip(p_obs + conf_noise, 0.0, 1.0))
    p_unc = float(np.clip(p_unc - 0.5 * conf_noise, 0.0, 1.0))

    u = rng.random()
    if u < p_obs:
        state = "OBSTACLE"
    elif u < min(1.0, p_obs + p_unc):
        state = "UNCERTAIN"
    else:
        state = "CLEAR"

    if state != "OBSTACLE" and rng.random() < p_false:
        state = "UNCERTAIN"

    if state == "OBSTACLE":
        detect_prob = float(np.clip(1.05 - 0.45 * d, 0.05, 0.98)) * (1.0 - preset["missed_echo_prob"])
        matched_peak = int(rng.random() < detect_prob)
    else:
        matched_peak = int(rng.random() < (0.2 * p_false))

    if state == "CLEAR":
        echo_p = float(np.clip(rng.normal(0.02, 0.015 + preset["obstacle_confidence_noise"]), 0.0, 0.20))
        peak_snr = float(np.clip(rng.normal(0.85, 0.20 + preset["obstacle_confidence_noise"]), 0.30, 2.0))
        peak_prom = float(np.clip(rng.normal(0.09, 0.06 + preset["obstacle_confidence_noise"]), 0.01, 0.60))
    elif state == "UNCERTAIN":
        echo_p = float(np.clip(rng.normal(0.18, 0.12 + preset["obstacle_confidence_noise"]), 0.01, 0.75))
        peak_snr = float(np.clip(rng.normal(1.25, 0.40 + preset["obstacle_confidence_noise"]), 0.40, 3.0))
        peak_prom = float(np.clip(rng.normal(0.24, 0.12 + preset["obstacle_confidence_noise"]), 0.03, 1.0))
    else:
        echo_p = float(np.clip(rng.normal(0.92, 0.10 + preset["obstacle_confidence_noise"]), 0.20, 1.0))
        peak_snr = float(np.clip(rng.normal(8.5, 2.2 + 8.0 * preset["obstacle_confidence_noise"]), 1.0, 15.0))
        peak_prom = float(np.clip(rng.normal(8.0, 2.2 + 8.0 * preset["obstacle_confidence_noise"]), 0.8, 16.0))

    predicted_distance = d if state == "OBSTACLE" else np.nan
    matched_distance = d if matched_peak == 1 else np.nan
    peak_width = float(np.clip(rng.normal(0.010, 0.004 + preset["echo_jitter_std"]), 0.002, 0.05))
    noise_floor = float(np.clip(rng.normal(0.20, 0.09 + preset["obstacle_confidence_noise"]), 0.02, 0.9))
    strongest_peak = float(peak_prom + rng.uniform(0.0, 0.3))
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
        MapDef("corridor", 12.0, 8.0, [Rect(0.0, 0.0, 12.0, 2.2), Rect(0.0, 5.8, 12.0, 8.0), Rect(5.5, 2.2, 6.5, 4.0)]),
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
    return ((cell[0] + 0.5) * grid.cell_size, (cell[1] + 0.5) * grid.cell_size)


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


def get_sense_params(difficulty: str) -> Tuple[float, float, float, int, float]:
    if difficulty == "hard_noise":
        return FRONT_SENSE_RANGE_HARD, SIDE_SENSE_RANGE_HARD, SENSE_FOV_DEG_HARD, SENSE_RAY_COUNT_HARD, SENSE_STEP_HARD
    return FRONT_SENSE_RANGE_DEFAULT, SIDE_SENSE_RANGE_DEFAULT, SENSE_FOV_DEG_DEFAULT, SENSE_RAY_COUNT_DEFAULT, SENSE_STEP_DEFAULT


def trace_observed_cells_from_pose(
    m: MapDef,
    x: float,
    y: float,
    heading: float,
    grid: GridDef,
    difficulty: str,
) -> Set[Tuple[int, int]]:
    front_range, side_range, fov_deg, ray_count, ray_step = get_sense_params(difficulty)
    cones = {
        "front": (0.0, front_range),
        "front_left": (math.radians(35.0), front_range),
        "front_right": (math.radians(-35.0), front_range),
        "left": (math.radians(80.0), side_range),
        "right": (math.radians(-80.0), side_range),
    }
    half_fov = math.radians(fov_deg / 2.0)
    observed: Set[Tuple[int, int]] = set()
    for _, (offset, max_r) in cones.items():
        center = heading + offset
        ray_angles = np.linspace(center - half_fov, center + half_fov, ray_count)
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
                t += ray_step
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


def cluster_frontier_cells(frontier_cells: Set[Tuple[int, int]], grid: GridDef) -> Dict[int, List[Tuple[int, int]]]:
    clusters: Dict[int, List[Tuple[int, int]]] = {}
    rem = set(frontier_cells)
    cid = 0
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
        clusters[cid] = comp
        cid += 1
    return clusters


def select_frontier_cluster(
    x: float,
    y: float,
    observed_cells: Set[Tuple[int, int]],
    frontier_cells: Set[Tuple[int, int]],
    grid: GridDef,
    recent_path_cells: List[Tuple[int, int]],
    blocked_clusters: Dict[int, int],
    context: str,
    local_loop: bool,
) -> Tuple[Optional[int], Optional[Tuple[int, int]], float]:
    clusters = cluster_frontier_cells(frontier_cells, grid)
    if not clusters:
        return None, None, 0.0
    recent_centers = [cell_center(c, grid) for c in recent_path_cells]
    best_cid = None
    best_target = None
    best_score = -1e18
    best_dist = 0.0
    for cid, comp in clusters.items():
        rep = max(comp, key=lambda c: unobserved_neighbor_count(c, observed_cells, grid))
        rcx, rcy = cell_center(rep, grid)
        d_agent = math.hypot(rcx - x, rcy - y)
        if recent_centers:
            d_recent = min(math.hypot(rcx - px, rcy - py) for (px, py) in recent_centers)
        else:
            d_recent = 0.0
        cluster_unobs = float(sum(unobserved_neighbor_count(c, observed_cells, grid) for c in comp))
        cluster_size = float(len(comp))
        blocked_penalty = 10.0 if blocked_clusters.get(cid, 0) > 0 else 0.0
        # v11: stronger clutter weighting for under-covered regions.
        clutter_bonus = 0.0
        if context == "cluttered_room":
            clutter_bonus += 2.0 * cluster_unobs + 0.8 * cluster_size
            if local_loop:
                clutter_bonus += 1.2 * d_recent
        score = 4.0 * cluster_unobs + 2.0 * cluster_size + 1.5 * d_recent - 0.6 * d_agent - blocked_penalty + clutter_bonus
        if score > best_score:
            best_score = score
            best_cid = cid
            best_target = rep
            best_dist = d_agent
    return best_cid, best_target, best_dist


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


def run_episode(
    m: MapDef,
    rng: np.random.Generator,
    difficulty: str,
    preset: Dict[str, float],
    max_steps: int,
) -> Dict[str, object]:
    sx, sy = sample_free_pose(m, rng)
    gx, gy = sample_free_pose(m, rng)  # plotting reference only
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

    selected_frontier_cluster_id: Optional[int] = None
    selected_frontier_target: Optional[Tuple[int, int]] = None
    frontier_commit_left = 0
    frontier_prev_dist: Optional[float] = None
    frontier_stall = 0
    blocked_clusters: Dict[int, int] = {}
    frontier_switch_count = 0
    blocked_frontier_count = 0
    frontier_dist_accum = 0.0
    frontier_dist_n = 0

    # structured mode memory
    sweep_heading: Optional[float] = None
    sweep_turn_pending = 0
    corridor_forward_bias_steps = 0
    doorway_lock_steps = 0
    doorway_side_explored = False
    doorway_commit_steps = 0
    doorway_commit_count = 0
    doorway_turn_history = deque(maxlen=DOORWAY_OSC_WINDOW)
    circumnav_side: Optional[str] = None
    circumnav_steps = 0

    consecutive_resample_count = 0
    consecutive_stop_reverse_count = 0
    resample_without_new_info_count = 0
    clutter_stagnation_steps = 0
    clutter_recovery_mode_steps = 0
    clutter_recovery_count = 0

    revisit_steps = 0
    loop_detected_count = 0
    loop_recovery_steps = 0
    in_loop_recovery = False

    new_cells_total = 0
    frontier_count_remaining = 0
    failure_reason = "timeout"

    for _step in range(1, max_steps + 1):
        for k in list(blocked_clusters.keys()):
            blocked_clusters[k] -= 1
            if blocked_clusters[k] <= 0:
                del blocked_clusters[k]

        true_dist = sector_true_distances(m, x, y, heading)
        sec = {s: synthesize_sector_perception(true_dist[s], rng, preset) for s in SECTOR_NAMES}
        fl = str(sec["front_left"]["predicted_state"]).upper()
        f = str(sec["front"]["predicted_state"]).upper()
        fr = str(sec["front_right"]["predicted_state"]).upper()
        l = str(sec["left"]["predicted_state"]).upper()
        r = str(sec["right"]["predicted_state"]).upper()
        any_front_obstacle_pred = (fl == "OBSTACLE") or (f == "OBSTACLE") or (fr == "OBSTACLE")

        before_obs = len(observed_cells)
        cur_cell = world_to_cell(x, y, grid)
        if cur_cell in grid.reachable_cells:
            visited_cells.add(cur_cell)
        observed_cells.update(trace_observed_cells_from_pose(m, x, y, heading, grid, difficulty))
        newly_observed_cells_this_step = len(observed_cells) - before_obs
        new_cells_total += max(0, newly_observed_cells_this_step)
        if context == "cluttered_room":
            if newly_observed_cells_this_step <= 0:
                clutter_stagnation_steps += 1
            else:
                clutter_stagnation_steps = max(0, clutter_stagnation_steps - 2)
            if clutter_stagnation_steps >= CLUTTER_STAGNATION_STEPS:
                clutter_recovery_mode_steps = CLUTTER_RECOVERY_STEPS
                clutter_stagnation_steps = 0
                clutter_recovery_count += 1

        obs_rate = len(observed_cells) / grid.total_reachable
        vis_rate = len(visited_cells) / grid.total_reachable
        if (obs_rate >= OBSERVED_COVERAGE_SUCCESS_THRESHOLD) and (vis_rate >= VISITED_COVERAGE_SUCCESS_THRESHOLD):
            success = True
            failure_reason = "covered_target"
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
        loop_detected = revisit_ratio_recent > LOOP_RATIO_THRESHOLD
        if loop_detected:
            loop_detected_count += 1
            in_loop_recovery = True
        if in_loop_recovery:
            loop_recovery_steps += 1
            if revisit_ratio_recent < 0.35:
                in_loop_recovery = False

        need_frontier_refresh = (
            selected_frontier_target is None
            or frontier_commit_left <= 0
            or selected_frontier_target not in frontier_cells
        )

        if selected_frontier_target is not None:
            tx, ty = cell_center(selected_frontier_target, grid)
            dcur = math.hypot(tx - x, ty - y)
            frontier_dist_accum += dcur
            frontier_dist_n += 1
            if frontier_prev_dist is not None and (frontier_prev_dist - dcur) < 0.01:
                frontier_stall += 1
            else:
                frontier_stall = 0
            frontier_prev_dist = dcur
            if frontier_stall >= FRONTIER_STALL_STEPS:
                if selected_frontier_cluster_id is not None:
                    blocked_clusters[selected_frontier_cluster_id] = BLOCKED_FRONTIER_COOLDOWN
                    blocked_frontier_count += 1
                need_frontier_refresh = True
                frontier_stall = 0

        if need_frontier_refresh:
            prev_cluster = selected_frontier_cluster_id
            cid, target, _d = select_frontier_cluster(
                x,
                y,
                observed_cells,
                frontier_cells,
                grid,
                recent_path_cells,
                blocked_clusters,
                context,
                loop_detected,
            )
            selected_frontier_cluster_id = cid
            selected_frontier_target = target
            frontier_commit_left = FRONTIER_COMMIT_STEPS if target is not None else 0
            frontier_prev_dist = None
            if prev_cluster is not None and cid is not None and cid != prev_cluster:
                frontier_switch_count += 1
        else:
            frontier_commit_left -= 1

        heading_to_frontier = None
        frontier_dist_cur = None
        if selected_frontier_target is not None:
            tx, ty = cell_center(selected_frontier_target, grid)
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
        doorway_aligned = False
        if heading_to_frontier is not None:
            doorway_aligned = abs(wrap_angle(heading_to_frontier - heading)) <= DOORWAY_ALIGN_ERR_RAD
        doorway_oscillating = False
        if len(doorway_turn_history) >= DOORWAY_OSC_WINDOW:
            doorway_turns = sum(1 for a in doorway_turn_history if a in {ACTION_LEFT, ACTION_RIGHT})
            doorway_oscillating = doorway_turns >= DOORWAY_OSC_TURN_THRESHOLD

        clutter_recovery_active = context == "cluttered_room" and clutter_recovery_mode_steps > 0
        candidates: List[str] = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
        sweep_progress_gain_hint = 0.0

        # A. open-space sweep
        if context == "open_space":
            if sweep_heading is None:
                sweep_heading = heading
            if true_dist["front"] < 0.55 or any_front_obstacle_pred:
                sweep_turn_pending = 1
                sweep_heading = wrap_angle(sweep_heading + math.radians(90.0))
            if sweep_turn_pending > 0:
                td = ACTION_LEFT if wrap_angle(sweep_heading - heading) > 0 else ACTION_RIGHT
                candidates = [td, ACTION_SLOW, ACTION_PROBE, ACTION_RESAMPLE, ACTION_STOP]
            else:
                candidates = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
                if true_dist["front"] > 0.9:
                    sweep_progress_gain_hint = 1.0

        # B. corridor end-to-end sweep
        elif context == "corridor":
            if corridor_mode and not any_front_obstacle_pred:
                corridor_forward_bias_steps = max(corridor_forward_bias_steps, 8)
            if corridor_forward_bias_steps > 0:
                candidates = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
                corridor_forward_bias_steps -= 1

        # D. doorway two-region exploration
        elif m.name == "doorway":
            if doorway_mode:
                doorway_lock_steps = max(doorway_lock_steps, 8)
                if doorway_aligned and doorway_commit_steps <= 0 and not any_front_obstacle_pred:
                    doorway_commit_steps = DOORWAY_COMMIT_STEPS
                    doorway_commit_count += 1
            # Cancel doorway commitment if it becomes unsafe.
            if doorway_commit_steps > 0:
                probe_unsafe = predict_forward_collision(m, x, y, heading, forward_step_for_action(ACTION_PROBE), margin)
                slow_unsafe = predict_forward_collision(m, x, y, heading, forward_step_for_action(ACTION_SLOW), margin)
                if any_front_obstacle_pred or (probe_unsafe and slow_unsafe):
                    doorway_commit_steps = 0
            if doorway_lock_steps > 0 and not any_front_obstacle_pred:
                candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
                doorway_lock_steps -= 1
                doorway_side_explored = True
            elif doorway_side_explored and selected_frontier_target is not None:
                candidates = [ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
            if doorway_commit_steps > 0:
                candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
                doorway_commit_steps -= 1

        # C. single-block circumnavigation
        elif m.name == "single_block":
            if f == "OBSTACLE" and circumnav_side is None:
                circumnav_side = choose_clearer_side_by_true_distance(true_dist)
                circumnav_steps = 30
            if circumnav_side is not None and circumnav_steps > 0:
                other = ACTION_RIGHT if circumnav_side == ACTION_LEFT else ACTION_LEFT
                candidates = [circumnav_side, ACTION_PROBE, ACTION_SLOW, other, ACTION_RESAMPLE, ACTION_STOP]
                circumnav_steps -= 1
            elif circumnav_steps <= 0:
                circumnav_side = None

        # E. clutter cautious expansion
        elif context == "cluttered_room":
            # v11: if locally stagnant, force controlled recovery instead of endless probing.
            if clutter_recovery_active:
                candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_PROBE, ACTION_SLOW, ACTION_STOP]
                clutter_recovery_mode_steps -= 1
            else:
                candidates = [ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_SLOW, ACTION_RESAMPLE, ACTION_STOP]

        if any_front_obstacle_pred:
            if f == "OBSTACLE":
                if (l == "CLEAR") and (r != "CLEAR"):
                    candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_PROBE, ACTION_RESAMPLE, ACTION_STOP]
                elif (r == "CLEAR") and (l != "CLEAR"):
                    candidates = [ACTION_RIGHT, ACTION_LEFT, ACTION_PROBE, ACTION_RESAMPLE, ACTION_STOP]
                elif (l == "CLEAR") and (r == "CLEAR"):
                    clearer = choose_clearer_side_by_true_distance(true_dist)
                    other = ACTION_RIGHT if clearer == ACTION_LEFT else ACTION_LEFT
                    candidates = [clearer, other, ACTION_PROBE, ACTION_RESAMPLE, ACTION_STOP]
                else:
                    candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
            elif (fl == "OBSTACLE") and (fr == "CLEAR"):
                candidates = [ACTION_RIGHT, ACTION_LEFT, ACTION_PROBE, ACTION_RESAMPLE, ACTION_STOP]
            elif (fr == "OBSTACLE") and (fl == "CLEAR"):
                candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_PROBE, ACTION_RESAMPLE, ACTION_STOP]
            else:
                candidates = [ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]

        safe_candidates: List[str] = []
        for a in candidates:
            # MOVE_FORWARD_FAST remains disabled by design.
            if a == ACTION_FAST:
                continue
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
        revisit_w = W_REVISIT_PENALTY * (1.6 if loop_detected else 1.0)
        frontier_align_w = W_FRONTIER_CLUSTER_ALIGN * (1.4 if loop_detected else 1.0)
        if context == "cluttered_room" and (clutter_recovery_active or loop_detected):
            revisit_w *= 1.25
            frontier_align_w *= 1.35

        for a in safe_candidates:
            nx, ny, nh, _, collided = apply_action(x, y, heading, a, m, turn_deg=15.0)
            if collided:
                continue
            nc = world_to_cell(nx, ny, grid)

            revisit_penalty = 1.0 if nc in visited_cells else 0.0
            if loop_detected and nc in recent_cells:
                revisit_penalty += 0.8

            # Approximate likely new observed gain.
            new_observed_cell_gain = 0.0
            if nc not in observed_cells:
                new_observed_cell_gain += 1.0
            new_observed_cell_gain += 0.35 * unobserved_neighbor_count(nc, observed_cells, grid)
            if a in {ACTION_SLOW, ACTION_PROBE}:
                new_observed_cell_gain += 0.45
            if a in {ACTION_LEFT, ACTION_RIGHT}:
                new_observed_cell_gain += 0.15

            frontier_cluster_alignment = 0.0
            frontier_distance_reduction = 0.0
            if selected_frontier_target is not None and heading_to_frontier is not None and frontier_dist_cur is not None:
                err = abs(wrap_angle(heading_to_frontier - nh))
                frontier_cluster_alignment = math.cos(err)
                tx, ty = cell_center(selected_frontier_target, grid)
                dnext = math.hypot(tx - nx, ty - ny)
                frontier_distance_reduction = (frontier_dist_cur - dnext) / max(0.25, grid.cell_size)

            heading_nov = heading_novelty(nc, nh, heading_memory)
            local_clear = local_clearance_score(m, nx, ny, nh)
            sweep_progress_gain = sweep_progress_gain_hint if a == ACTION_SLOW else 0.0

            stop_penalty = 0.0
            if a == ACTION_STOP:
                stop_penalty = BASE_STOP_PENALTY + consecutive_stop_reverse_count * REPEAT_STOP_PENALTY
                if non_stop_safe:
                    stop_penalty += 4.0

            passive_penalty = 0.0
            if a == ACTION_RESAMPLE:
                if newly_observed_cells_this_step > 0:
                    passive_penalty = RESAMPLE_INFO_PENALTY
                else:
                    passive_penalty = 1.0 + 0.5 * resample_without_new_info_count
                passive_penalty += BASE_RESAMPLE_PENALTY + consecutive_resample_count * REPEAT_RESAMPLE_PENALTY

            # directional shaping by map mode
            map_bonus = 0.0
            if context == "open_space":
                if a == ACTION_SLOW and true_dist["front"] > 1.0:
                    map_bonus += 0.8
                if a in {ACTION_LEFT, ACTION_RIGHT} and sweep_turn_pending == 0:
                    map_bonus -= 0.2
            elif context == "corridor":
                if corridor_mode and a in {ACTION_SLOW, ACTION_PROBE}:
                    map_bonus += 0.7
                if corridor_mode and a in {ACTION_LEFT, ACTION_RIGHT}:
                    map_bonus -= 0.45
            elif m.name == "single_block":
                if circumnav_side is not None and a == circumnav_side:
                    map_bonus += 0.6
            elif m.name == "doorway":
                if doorway_lock_steps > 0 and a in {ACTION_PROBE, ACTION_SLOW}:
                    map_bonus += 0.8
                if doorway_lock_steps > 0 and a in {ACTION_LEFT, ACTION_RIGHT, ACTION_STOP}:
                    map_bonus -= 0.8
                if doorway_commit_steps > 0 and a in {ACTION_PROBE, ACTION_SLOW}:
                    map_bonus += 1.0
                if doorway_commit_steps > 0 and a in {ACTION_LEFT, ACTION_RIGHT}:
                    map_bonus -= 0.9
                if doorway_oscillating and a in {ACTION_LEFT, ACTION_RIGHT}:
                    map_bonus -= 1.1
            elif context == "cluttered_room":
                if a == ACTION_PROBE:
                    map_bonus += 0.6
                if a == ACTION_SLOW and min(true_dist.values()) < 0.55:
                    map_bonus -= 0.5
                if clutter_recovery_active:
                    if a in {ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE}:
                        map_bonus += 0.7
                    if a == ACTION_PROBE:
                        map_bonus += 0.4
                if a == ACTION_RESAMPLE and newly_observed_cells_this_step > 0:
                    map_bonus += 0.35

            score = (
                W_NEW_OBS_GAIN * new_observed_cell_gain
                + frontier_align_w * frontier_cluster_alignment
                + W_FRONTIER_DIST_REDUCE * frontier_distance_reduction
                + W_SWEEP_PROGRESS * sweep_progress_gain
                + W_HEADING_NOVELTY * heading_nov
                + W_LOCAL_CLEARANCE * local_clear
                - revisit_w * revisit_penalty
                - W_STOP_PENALTY * stop_penalty
                - W_PASSIVE_PENALTY * passive_penalty
                + map_bonus
            )
            if score > best_score:
                best_score = score
                best_action = a

        action = best_action

        # active recovery order when STOP selected but safe alternatives exist
        if action == ACTION_STOP and non_stop_safe:
            turn_opts = [a for a in non_stop_safe if a in {ACTION_LEFT, ACTION_RIGHT}]
            if turn_opts and selected_frontier_target is not None and heading_to_frontier is not None:
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

        if action in FORWARD_ACTIONS:
            if any_front_obstacle_pred or predict_forward_collision(m, x, y, heading, forward_step_for_action(action), margin):
                action = ACTION_RESAMPLE

        actions[action] += 1
        nx, ny, nhead, moved, collided = apply_action(x, y, heading, action, m, turn_deg=15.0)
        path_len += moved

        before_act_obs = len(observed_cells)
        x, y, heading = nx, ny, nhead
        path.append((x, y))
        observed_cells.update(trace_observed_cells_from_pose(m, x, y, heading, grid, difficulty))
        newly_observed_cells_from_action = len(observed_cells) - before_act_obs
        new_cells_total += max(0, newly_observed_cells_from_action)

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
            if newly_observed_cells_this_step > 0 or newly_observed_cells_from_action > 0:
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

        if m.name == "doorway":
            doorway_turn_history.append(action)

        if collided:
            collision = True
            failure_reason = "collision"
            break

    final_obs = len(observed_cells) / grid.total_reachable
    final_vis = len(visited_cells) / grid.total_reachable
    timeout = (not success) and (not collision)
    if timeout:
        if final_obs >= 0.95:
            failure_reason = "timeout_high_coverage"
        elif final_obs >= 0.75:
            failure_reason = "timeout_mid_coverage"
        elif final_obs >= 0.50:
            failure_reason = "timeout_low_coverage"
        else:
            failure_reason = "timeout_very_low_coverage"
    steps = max(1, len(path) - 1)
    mean_frontier_distance = frontier_dist_accum / max(1, frontier_dist_n)
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
        "final_observed_coverage_rate": float(final_obs),
        "final_visited_coverage_rate": float(final_vis),
        "observed_coverage_success_rate_50": bool(final_obs >= 0.50),
        "observed_coverage_success_rate_75": bool(final_obs >= 0.75),
        "observed_coverage_success_rate_90": bool(final_obs >= 0.90),
        "observed_coverage_success_rate_95": bool(final_obs >= 0.95),
        "new_cells_total": int(new_cells_total),
        "revisit_ratio": float(revisit_steps / steps),
        "mean_frontier_distance": float(mean_frontier_distance),
        "frontier_count_remaining": int(frontier_count_remaining),
        "selected_frontier_cluster_id": int(selected_frontier_cluster_id) if selected_frontier_cluster_id is not None else None,
        "frontier_switch_count": int(frontier_switch_count),
        "blocked_frontier_count": int(blocked_frontier_count),
        "doorway_commit_count": int(doorway_commit_count),
        "clutter_recovery_count": int(clutter_recovery_count),
        "loop_detected_count": int(loop_detected_count),
        "loop_recovery_steps": int(loop_recovery_steps),
        "failure_reason": str(failure_reason),
        "stop_reverse_rate": float(actions.get(ACTION_STOP, 0) / steps),
        "probe_plus_slow_rate": float((actions.get(ACTION_SLOW, 0) + actions.get(ACTION_PROBE, 0)) / steps),
        "observed_cells": observed_cells,
        "visited_cells": visited_cells,
        "grid": grid,
    }


def draw_path_grid_panel(ax, m: MapDef, ep: Dict[str, object]) -> None:
    draw_map(ax, m)
    grid: GridDef = ep["grid"]  # type: ignore[assignment]
    for c in grid.reachable_cells:
        px, py = cell_center(c, grid)
        ax.add_patch(
            plt.Rectangle(
                (px - grid.cell_size / 2.0, py - grid.cell_size / 2.0),
                grid.cell_size,
                grid.cell_size,
                color="lightgray",
                alpha=0.10,
            )
        )
    for c in ep["observed_cells"]:  # type: ignore[index]
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
    for c in ep["visited_cells"]:  # type: ignore[index]
        px, py = cell_center(c, grid)
        ax.add_patch(
            plt.Rectangle(
                (px - grid.cell_size / 2.0, py - grid.cell_size / 2.0),
                grid.cell_size,
                grid.cell_size,
                color="navy",
                alpha=0.20,
            )
        )
    p = np.asarray(ep["path"], dtype=float)  # type: ignore[arg-type]
    ax.plot(p[:, 0], p[:, 1], color="tab:orange", linewidth=1.1, alpha=0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v11 doorway/clutter tuning curriculum.")
    parser.add_argument("--episodes-per-map", type=int, default=EPISODES_PER_MAP)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--difficulties",
        type=str,
        default="clean,mild_noise,medium_noise,hard_noise",
        help="Comma-separated difficulties from: clean,mild_noise,medium_noise,hard_noise",
    )
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    for d in difficulties:
        if d not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {d}")

    maps = make_maps()
    results_by_diff: Dict[str, Dict[str, object]] = {}

    # For plotting, keep one representative path/coverage episode per map/difficulty.
    sample_eps: Dict[Tuple[str, str], Dict[str, object]] = {}

    for diff in difficulties:
        preset = DIFFICULTY_PRESETS[diff]
        diff_map_results: Dict[str, Dict[str, object]] = {}
        diff_actions = Counter()
        total_stuck_recovery = 0
        total_scan_turns = 0

        for m in maps:
            episodes = []
            map_actions = Counter()
            for _ in range(args.episodes_per_map):
                ep = run_episode(m, rng, diff, preset, args.max_steps)
                episodes.append(ep)
                map_actions.update(ep["action_counts"])  # type: ignore[arg-type]
                diff_actions.update(ep["action_counts"])  # type: ignore[arg-type]
                total_stuck_recovery += int(ep["loop_recovery_steps"])
                total_scan_turns += int(ep["loop_detected_count"])

            coll = sum(int(e["collision"]) for e in episodes)
            tout = sum(int(e["timeout"]) for e in episodes)
            c50 = sum(int(e["observed_coverage_success_rate_50"]) for e in episodes)
            c75 = sum(int(e["observed_coverage_success_rate_75"]) for e in episodes)
            c90 = sum(int(e["observed_coverage_success_rate_90"]) for e in episodes)
            c95 = sum(int(e["observed_coverage_success_rate_95"]) for e in episodes)
            map_action_dist = _action_distribution(map_actions)
            failure_reason_counts = Counter([str(e["failure_reason"]) for e in episodes])
            coverage_success_rate = c95 / args.episodes_per_map

            diff_map_results[m.name] = {
                "mean_observed_coverage_rate": float(np.mean([e["final_observed_coverage_rate"] for e in episodes])),
                "mean_visited_coverage_rate": float(np.mean([e["final_visited_coverage_rate"] for e in episodes])),
                "coverage_success_rate": coverage_success_rate,
                "observed_coverage_success_rate_50": c50 / args.episodes_per_map,
                "observed_coverage_success_rate_75": c75 / args.episodes_per_map,
                "observed_coverage_success_rate_90": c90 / args.episodes_per_map,
                "observed_coverage_success_rate_95": c95 / args.episodes_per_map,
                "collision_rate": coll / args.episodes_per_map,
                "timeout_rate": tout / args.episodes_per_map,
                "mean_steps": float(np.mean([e["steps"] for e in episodes])),
                "mean_path_length_m": float(np.mean([e["path_length"] for e in episodes])),
                "mean_new_cells_per_episode": float(np.mean([e["new_cells_total"] for e in episodes])),
                "mean_revisit_ratio": float(np.mean([e["revisit_ratio"] for e in episodes])),
                "mean_frontier_count_remaining": float(np.mean([e["frontier_count_remaining"] for e in episodes])),
                "frontier_switch_count": float(np.mean([e["frontier_switch_count"] for e in episodes])),
                "blocked_frontier_count": float(np.mean([e["blocked_frontier_count"] for e in episodes])),
                "doorway_commit_count": float(np.mean([e["doorway_commit_count"] for e in episodes])),
                "clutter_recovery_count": float(np.mean([e["clutter_recovery_count"] for e in episodes])),
                "loop_detected_count": float(np.mean([e["loop_detected_count"] for e in episodes])),
                "mean_loop_recovery_steps": float(np.mean([e["loop_recovery_steps"] for e in episodes])),
                "mean_frontier_distance": float(np.mean([e["mean_frontier_distance"] for e in episodes])),
                "stop_reverse_rate": float(map_action_dist[ACTION_STOP]["rate"]),
                "probe_plus_slow_rate": float(map_action_dist[ACTION_SLOW]["rate"] + map_action_dist[ACTION_PROBE]["rate"]),
                "action_distribution": map_action_dist,
                "failure_reason_counts": dict(failure_reason_counts),
            }

            sample_eps[(diff, m.name)] = episodes[-1]

        diff_action_dist = _action_distribution(diff_actions)
        diff_agg = {
            "mean_observed_coverage_all_maps": float(np.mean([diff_map_results[m.name]["mean_observed_coverage_rate"] for m in maps])),
            "mean_collision_all_maps": float(np.mean([diff_map_results[m.name]["collision_rate"] for m in maps])),
            "mean_timeout_all_maps": float(np.mean([diff_map_results[m.name]["timeout_rate"] for m in maps])),
            "mean_stop_reverse_rate_all_maps": float(diff_action_dist[ACTION_STOP]["rate"]),
            "mean_probe_plus_slow_rate_all_maps": float(diff_action_dist[ACTION_SLOW]["rate"] + diff_action_dist[ACTION_PROBE]["rate"]),
        }

        results_by_diff[diff] = {
            "maps": diff_map_results,
            "overall_action_distribution": diff_action_dist,
            "aggregate": diff_agg,
            "total_stuck_recovery_count": int(total_stuck_recovery),
            "total_scan_turn_count": int(total_scan_turns),
        }

    # v9 comparison for clean mode.
    comparison_vs_v9 = []
    v9_summary = json.loads(V9_JSON.read_text(encoding="utf-8")) if V9_JSON.exists() else None
    if "clean" in results_by_diff:
        clean_maps = results_by_diff["clean"]["maps"]  # type: ignore[index]
        for m in maps:
            row = {
                "map": m.name,
                "v9_observed_coverage": None,
                "v11_clean_observed_coverage": clean_maps[m.name]["mean_observed_coverage_rate"],
                "v9_visited_coverage": None,
                "v11_clean_visited_coverage": clean_maps[m.name]["mean_visited_coverage_rate"],
                "v9_collision_rate": None,
                "v11_clean_collision_rate": clean_maps[m.name]["collision_rate"],
                "v9_timeout_rate": None,
                "v11_clean_timeout_rate": clean_maps[m.name]["timeout_rate"],
                "v9_revisit_ratio": None,
                "v11_clean_revisit_ratio": clean_maps[m.name]["mean_revisit_ratio"],
                "v9_path_length": None,
                "v11_clean_path_length": clean_maps[m.name]["mean_path_length_m"],
            }
            if v9_summary is not None and "maps" in v9_summary and m.name in v9_summary["maps"]:
                v9m = v9_summary["maps"][m.name]
                row["v9_observed_coverage"] = v9m.get("mean_observed_coverage_rate")
                row["v9_visited_coverage"] = v9m.get("mean_visited_coverage_rate")
                row["v9_collision_rate"] = v9m.get("collision_rate")
                row["v9_timeout_rate"] = v9m.get("timeout_rate")
                row["v9_revisit_ratio"] = v9m.get("mean_revisit_ratio")
                row["v9_path_length"] = v9m.get("mean_path_length_m")
            comparison_vs_v9.append(row)

    # Acceptance checks.
    def _accept_map(diff: str, map_name: str, threshold: float) -> bool:
        return (results_by_diff[diff]["maps"][map_name]["mean_observed_coverage_rate"] >= threshold)  # type: ignore[index]

    accepted_clean = False
    if "clean" in results_by_diff:
        clean_ok = (
            _accept_map("clean", "empty_room", 0.95)
            and _accept_map("clean", "corridor", 0.95)
            and _accept_map("clean", "single_block", 0.90)
            and _accept_map("clean", "doorway", 0.85)
            and _accept_map("clean", "cluttered_room", 0.50)
        )
        clean_collision = results_by_diff["clean"]["aggregate"]["mean_collision_all_maps"] <= 0.01  # type: ignore[index]
        accepted_clean = bool(clean_ok and clean_collision)

    accepted_mild = False
    if "mild_noise" in results_by_diff:
        mild_ok = (
            _accept_map("mild_noise", "empty_room", 0.90)
            and _accept_map("mild_noise", "corridor", 0.90)
            and _accept_map("mild_noise", "single_block", 0.80)
            and _accept_map("mild_noise", "doorway", 0.75)
            and _accept_map("mild_noise", "cluttered_room", 0.40)
        )
        mild_collision = results_by_diff["mild_noise"]["aggregate"]["mean_collision_all_maps"] <= 0.02  # type: ignore[index]
        accepted_mild = bool(mild_ok and mild_collision)

    accepted_medium = False
    if "medium_noise" in results_by_diff:
        medium_ok = (
            _accept_map("medium_noise", "empty_room", 0.88)
            and _accept_map("medium_noise", "corridor", 0.85)
            and _accept_map("medium_noise", "single_block", 0.78)
            and _accept_map("medium_noise", "doorway", 0.70)
            and _accept_map("medium_noise", "cluttered_room", 0.35)
        )
        medium_collision = results_by_diff["medium_noise"]["aggregate"]["mean_collision_all_maps"] <= 0.02  # type: ignore[index]
        accepted_medium = bool(medium_ok and medium_collision)

    accepted_hard = False
    if "hard_noise" in results_by_diff:
        hard_ok = (
            _accept_map("hard_noise", "empty_room", 0.85)
            and _accept_map("hard_noise", "corridor", 0.80)
            and _accept_map("hard_noise", "single_block", 0.72)
            and _accept_map("hard_noise", "doorway", 0.65)
            and _accept_map("hard_noise", "cluttered_room", 0.30)
        )
        hard_collision = results_by_diff["hard_noise"]["aggregate"]["mean_collision_all_maps"] <= 0.02  # type: ignore[index]
        accepted_hard = bool(hard_ok and hard_collision)

    # behavior acceptance from full curriculum average
    all_diff_action_rates = []
    for d in difficulties:
        da = results_by_diff[d]["overall_action_distribution"]  # type: ignore[index]
        all_diff_action_rates.append(
            (
                da[ACTION_STOP]["rate"],
                da[ACTION_SLOW]["rate"] + da[ACTION_PROBE]["rate"],
                da[ACTION_RESAMPLE]["rate"],
            )
        )
    mean_stop = float(np.mean([x[0] for x in all_diff_action_rates])) if all_diff_action_rates else 0.0
    mean_probe_slow = float(np.mean([x[1] for x in all_diff_action_rates])) if all_diff_action_rates else 0.0
    mean_resample = float(np.mean([x[2] for x in all_diff_action_rates])) if all_diff_action_rates else 0.0
    behavior_acceptance = {
        "stop_reverse_le_0.05": mean_stop <= 0.05,
        "probe_plus_slow_ge_0.45": mean_probe_slow >= 0.45,
        "resample_le_0.05": mean_resample <= 0.05,
    }

    accepted_overall = bool(accepted_clean and accepted_mild and accepted_medium and accepted_hard and all(behavior_acceptance.values()))
    needs_more_tuning = not accepted_overall
    final_status = "accepted_overall" if accepted_overall else "needs_more_tuning"

    results = {
        "episodes_per_map": args.episodes_per_map,
        "max_steps": args.max_steps,
        "coverage_cell_size_m": COVERAGE_CELL_SIZE_M,
        "difficulty_presets": DIFFICULTY_PRESETS,
        "difficulties": difficulties,
        "results_by_difficulty": results_by_diff,
        "comparison_vs_v9": comparison_vs_v9,
        "acceptance": {
            "accepted_clean": bool(accepted_clean),
            "accepted_mild": bool(accepted_mild),
            "accepted_medium": bool(accepted_medium),
            "accepted_hard": bool(accepted_hard),
            "accepted_overall": bool(accepted_overall),
            "needs_more_tuning": bool(needs_more_tuning),
            "behavior_acceptance": behavior_acceptance,
            "final_status": final_status,
        },
    }
    RESULT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Action distribution plot by difficulty.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=140)
    axes = axes.flatten()
    for i, d in enumerate(difficulties[:4]):
        ax = axes[i]
        dist = results_by_diff[d]["overall_action_distribution"]  # type: ignore[index]
        vals = [100.0 * dist[a]["rate"] for a in ALL_ACTIONS]
        ax.bar(ALL_ACTIONS, vals, color="tab:blue")
        ax.set_title(d)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("Action share (%)")
    for j in range(len(difficulties), 4):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_ACTION_PNG)
    plt.close(fig)

    # Paths plot (clean mode only if present, else first difficulty).
    path_diff = "clean" if "clean" in difficulties else difficulties[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.flatten()
    for i, m in enumerate(maps):
        ax = axes[i]
        draw_map(ax, m)
        ax.set_title(f"{path_diff}: {m.name}")
        ep = sample_eps[(path_diff, m.name)]
        p = np.asarray(ep["path"], dtype=float)  # type: ignore[arg-type]
        ax.plot(p[:, 0], p[:, 1], color="tab:orange", linewidth=1.1, alpha=0.9)
        sx, sy = ep["start"]  # type: ignore[index]
        gx, gy = ep["goal"]  # type: ignore[index]
        ax.scatter([sx], [sy], c="blue", s=10)
        ax.scatter([gx], [gy], c="black", s=10, marker="x")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_PATHS_PNG)
    plt.close(fig)

    # Coverage map plot for clean mode.
    cov_diff = "clean" if "clean" in difficulties else difficulties[0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.flatten()
    for i, m in enumerate(maps):
        ax = axes[i]
        ax.set_title(f"{cov_diff}: {m.name}")
        draw_path_grid_panel(ax, m, sample_eps[(cov_diff, m.name)])
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_COVERAGE_PNG)
    plt.close(fig)

    # Difficulty coverage/collision summary.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    diffs = difficulties
    mean_cov = [results_by_diff[d]["aggregate"]["mean_observed_coverage_all_maps"] for d in diffs]  # type: ignore[index]
    mean_col = [results_by_diff[d]["aggregate"]["mean_collision_all_maps"] for d in diffs]  # type: ignore[index]
    axes[0].bar(diffs, [100.0 * x for x in mean_cov], color="tab:green")
    axes[0].set_title("Mean Observed Coverage (All Maps)")
    axes[0].set_ylabel("%")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(diffs, [100.0 * x for x in mean_col], color="tab:red")
    axes[1].set_title("Mean Collision Rate (All Maps)")
    axes[1].set_ylabel("%")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULT_DIFF_SUMMARY_PNG)
    plt.close(fig)

    # Console summary.
    for d in difficulties:
        print(f"\n{d} summary:")
        for m in maps:
            r = results_by_diff[d]["maps"][m.name]  # type: ignore[index]
            print(
                f"  {m.name}: observed={100*r['mean_observed_coverage_rate']:.2f}% | "
                f"visited={100*r['mean_visited_coverage_rate']:.2f}% | "
                f"collision={100*r['collision_rate']:.2f}% | timeout={100*r['timeout_rate']:.2f}% | "
                f"stop={100*r['stop_reverse_rate']:.2f}% | probe+slow={100*r['probe_plus_slow_rate']:.2f}%"
            )

    if comparison_vs_v9:
        print("\nv9 vs v11 clean comparison:")
        for row in comparison_vs_v9:
            v9_obs = row["v9_observed_coverage"]
            v9_vis = row["v9_visited_coverage"]
            v9_col = row["v9_collision_rate"]
            v9_t = row["v9_timeout_rate"]
            print(
                f"  {row['map']}: obs {100*v9_obs if v9_obs is not None else float('nan'):.2f}% -> "
                f"{100*row['v11_clean_observed_coverage']:.2f}% | "
                f"vis {100*v9_vis if v9_vis is not None else float('nan'):.2f}% -> "
                f"{100*row['v11_clean_visited_coverage']:.2f}% | "
                f"col {100*v9_col if v9_col is not None else float('nan'):.2f}% -> "
                f"{100*row['v11_clean_collision_rate']:.2f}% | "
                f"timeout {100*v9_t if v9_t is not None else float('nan'):.2f}% -> "
                f"{100*row['v11_clean_timeout_rate']:.2f}%"
            )

    print("\nAcceptance:")
    print(f"  accepted_clean: {accepted_clean}")
    print(f"  accepted_mild: {accepted_mild}")
    print(f"  accepted_medium: {accepted_medium}")
    print(f"  accepted_hard: {accepted_hard}")
    print(f"  accepted_overall: {accepted_overall}")
    print(f"  needs_more_tuning: {needs_more_tuning}")
    print(f"  behavior_acceptance: {behavior_acceptance}")
    print(f"  final_status: {final_status}")

    print(f"\nSaved: {RESULT_JSON}")
    print(f"Saved: {RESULT_ACTION_PNG}")
    print(f"Saved: {RESULT_PATHS_PNG}")
    print(f"Saved: {RESULT_COVERAGE_PNG}")
    print(f"Saved: {RESULT_DIFF_SUMMARY_PNG}")


if __name__ == "__main__":
    main()
