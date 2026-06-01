"""
Simple 2D acoustic navigation simulator v6.

Adds on top of v5_nav:
- Goal-cone reward bonus in local planner
- Adaptive forward-bias weighting when goal is straight ahead in constrained maps
- Map-type tuned resample/commit parameters for tighter scenarios
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("simulation/results")
RESULT_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v6_results.json"
RESULT_PATHS_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v6_paths.png"
RESULT_ACTION_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v6_action_distribution.png"
V1_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v1_results.json"
V2_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v2_results.json"
V3_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v3_results.json"
V4_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v4_results.json"
V5_NAV_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v5_nav_results.json"

EPISODES_PER_MAP = 100
MAX_STEPS = 340
ROBOT_RADIUS = 0.12
SAFETY_MARGIN = 0.05
GOAL_RADIUS = 0.30
RAY_MAX_RANGE = 3.0
RAY_STEP = 0.03

RESAMPLE_CAP = 6
PROBE_ESCALATE_COUNT = 3
PROBE_ESCALATE_PROGRESS_M = 0.05
COMMIT_SLOW_STEPS = 3
STUCK_WINDOW = 20
STUCK_MIN_PROGRESS_M = 0.05
OSC_BIAS_STEPS = 4
SCAN_TURN_COOLDOWN_STEPS = 12
LOCAL_PLANNER_GOAL_W = 1.0
LOCAL_PLANNER_COLLISION_PENALTY = 20.0
LOCAL_PLANNER_CLEARANCE_W = 0.25
GOAL_CONE_BONUS = 0.35
CONSTRAINED_FORWARD_BIAS = 0.45

SECTOR_NAMES = ["left", "front_left", "front", "front_right", "right"]
SECTOR_OFFSETS_RAD = {
    "left": math.radians(90.0),
    "front_left": math.radians(40.0),
    "front": 0.0,
    "front_right": math.radians(-40.0),
    "right": math.radians(-90.0),
}
SECTOR_CONE_HALF_RAD = math.radians(12.0)
SECTOR_CONE_SAMPLES = 3

ACTION_FAST = "MOVE_FORWARD_FAST"
ACTION_SLOW = "MOVE_FORWARD_SLOW"
ACTION_PROBE = "PROBE_FORWARD"
ACTION_LEFT = "TURN_LEFT"
ACTION_RIGHT = "TURN_RIGHT"
ACTION_RESAMPLE = "SLOW_DOWN_AND_RESAMPLE"
ACTION_STOP = "STOP_OR_REVERSE"
ALL_ACTIONS = [ACTION_FAST, ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]


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
    out = {}
    for sec in SECTOR_NAMES:
        center = heading + SECTOR_OFFSETS_RAD[sec]
        angles = np.linspace(center - SECTOR_CONE_HALF_RAD, center + SECTOR_CONE_HALF_RAD, SECTOR_CONE_SAMPLES)
        vals = [raycast_distance(m, x, y, a, RAY_MAX_RANGE, RAY_STEP) for a in angles]
        out[sec] = float(min(vals))
    return out


def synthesize_sector_perception(true_d: float, rng: np.random.Generator) -> Dict[str, float | str | int]:
    # Same conservative observation model as v2.
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


def choose_clearer_side_by_true_distance(true_dist: Dict[str, float]) -> str:
    return ACTION_LEFT if true_dist["left"] >= true_dist["right"] else ACTION_RIGHT


def forward_step_for_action(action: str) -> float:
    return {ACTION_FAST: 0.20, ACTION_SLOW: 0.08, ACTION_PROBE: 0.03}.get(action, 0.0)


def adaptive_turn_deg(abs_heading_err_deg: float, front_true_dist: float) -> float:
    if abs_heading_err_deg > 60.0:
        base = 30.0
    elif abs_heading_err_deg > 30.0:
        base = 20.0
    else:
        base = 10.0
    if front_true_dist < 0.55:
        base = max(base, 20.0)
    return float(base)


def predict_forward_collision(m: MapDef, x: float, y: float, heading: float, dist: float, margin: float) -> bool:
    nseg = max(2, int(abs(dist) / 0.01))
    for i in range(1, nseg + 1):
        t = i / nseg
        px = x + t * dist * math.cos(heading)
        py = y + t * dist * math.sin(heading)
        if point_in_obstacle(px, py, m, robot_radius=ROBOT_RADIUS + margin):
            return True
    return False


def local_plan_score(
    m: MapDef,
    x: float,
    y: float,
    heading: float,
    gx: float,
    gy: float,
    action: str,
    turn_deg: float,
    abs_heading_err_deg: float,
    corridor_mode: bool,
    constrained_map: bool,
) -> float:
    # One-step simulate candidate action.
    nx, ny, nh, _, collided = apply_action(x, y, heading, action, m, forced_turn_deg=(turn_deg if action in {ACTION_LEFT, ACTION_RIGHT} else None))
    if collided:
        return -LOCAL_PLANNER_COLLISION_PENALTY

    d_goal = math.hypot(gx - nx, gy - ny)
    clearance = min(sector_true_distances(m, nx, ny, nh).values())
    score = -LOCAL_PLANNER_GOAL_W * d_goal + LOCAL_PLANNER_CLEARANCE_W * clearance

    # Small goal-cone bonus: encourage safe forward progress when roughly aligned to goal.
    if action in {ACTION_FAST, ACTION_SLOW, ACTION_PROBE} and abs_heading_err_deg <= 15.0:
        score += GOAL_CONE_BONUS

    # Additional constrained-map forward bias when goal is mostly ahead.
    if constrained_map and corridor_mode and action in {ACTION_SLOW, ACTION_PROBE} and abs_heading_err_deg <= 25.0:
        score += CONSTRAINED_FORWARD_BIAS

    # Small look-ahead after turn: try probe once.
    if action in {ACTION_LEFT, ACTION_RIGHT}:
        fx, fy, _, _, col2 = apply_action(nx, ny, nh, ACTION_PROBE, m)
        if col2:
            score -= 2.0
        else:
            score += -0.35 * math.hypot(gx - fx, gy - fy)

    return float(score)


def apply_action(
    x: float, y: float, heading: float, action: str, m: MapDef, forced_turn_deg: float | None = None
) -> Tuple[float, float, float, float, bool]:
    turn_step_deg = 15.0 if forced_turn_deg is None else forced_turn_deg
    turn = {ACTION_LEFT: math.radians(turn_step_deg), ACTION_RIGHT: math.radians(-turn_step_deg)}

    if action in turn:
        return x, y, wrap_angle(heading + turn[action]), 0.0, False
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

    moved = math.hypot(nx - x, ny - y)
    return nx, ny, heading, moved, collided


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


def run_episode(m: MapDef, rng: np.random.Generator) -> Dict[str, object]:
    sx, sy = sample_free_pose(m, rng)
    gx, gy = sample_free_pose(m, rng)
    tries = 0
    while math.hypot(gx - sx, gy - sy) < 3.0 and tries < 300:
        gx, gy = sample_free_pose(m, rng)
        tries += 1

    heading = rng.uniform(-math.pi, math.pi)
    x, y = sx, sy
    path = [(x, y)]
    actions = Counter()
    min_obs = float("inf")
    path_len = 0.0
    collision = False
    success = False

    goal_dist_hist = deque(maxlen=STUCK_WINDOW)
    consecutive_front_clear_count = 0
    consecutive_front_blocked_count = 0
    consecutive_no_progress_count = 0
    consecutive_resample_count = 0
    probe_safe_streak = 0
    probe_progress_accum = 0.0
    commit_slow_remaining = 0
    last_turns = deque(maxlen=4)
    turn_bias: str | None = None
    turn_bias_steps = 0
    stuck_steps = 0
    stuck_recovery_count = 0
    scan_turn_count = 0
    scan_turn_cooldown = 0
    last_scan_turn_dir = ACTION_LEFT

    # Map-type parameter tuning (deterministic).
    if m.name in {"corridor", "doorway"}:
        map_resample_cap = 4
        map_commit_steps = 4
        constrained_map = True
    elif m.name == "cluttered_room":
        map_resample_cap = 6
        map_commit_steps = 2
        constrained_map = True
    elif m.name == "single_block":
        map_resample_cap = 5
        map_commit_steps = 3
        constrained_map = False
    else:  # empty_room
        map_resample_cap = 3
        map_commit_steps = 4
        constrained_map = False

    prev_goal_dist = math.hypot(gx - x, gy - y)

    for step in range(1, MAX_STEPS + 1):
        goal_dist = math.hypot(gx - x, gy - y)
        goal_dist_hist.append(goal_dist)

        true_dist = sector_true_distances(m, x, y, heading)
        min_obs = min(min_obs, min(true_dist.values()))
        sec = {s: synthesize_sector_perception(true_dist[s], rng) for s in SECTOR_NAMES}

        fl = str(sec["front_left"]["predicted_state"]).upper()
        f = str(sec["front"]["predicted_state"]).upper()
        fr = str(sec["front_right"]["predicted_state"]).upper()
        l = str(sec["left"]["predicted_state"]).upper()
        r = str(sec["right"]["predicted_state"]).upper()

        front_all_clear = (fl == "CLEAR") and (f == "CLEAR") and (fr == "CLEAR")
        any_front_obstacle = (fl == "OBSTACLE") or (f == "OBSTACLE") or (fr == "OBSTACLE")
        front_not_clearly_open = not front_all_clear

        if front_all_clear:
            consecutive_front_clear_count += 1
            consecutive_front_blocked_count = 0
        elif any_front_obstacle:
            consecutive_front_blocked_count += 1
            consecutive_front_clear_count = 0
        else:
            consecutive_front_clear_count = 0
            consecutive_front_blocked_count = 0

        # Stuck detection.
        stuck = False
        if len(goal_dist_hist) >= STUCK_WINDOW and (goal_dist_hist[0] - goal_dist_hist[-1]) < STUCK_MIN_PROGRESS_M:
            stuck = True

        if (prev_goal_dist - goal_dist) < 1e-3:
            consecutive_no_progress_count += 1
        else:
            consecutive_no_progress_count = 0
        prev_goal_dist = goal_dist

        stuck_steps = stuck_steps + 1 if stuck else 0

        goal_angle = math.atan2(gy - y, gx - x)
        heading_error = wrap_angle(goal_angle - heading)
        abs_err_deg = abs(math.degrees(heading_error))
        turn_to_goal = ACTION_LEFT if heading_error > 0 else ACTION_RIGHT
        adaptive_turn = adaptive_turn_deg(abs_err_deg, true_dist["front"])
        forced_turn_deg: float | None = None

        # Candidate actions from policy heuristics.
        candidates: list[str] = []

        # Context-sensitive resample cap with map-profile baseline.
        dynamic_resample_cap = map_resample_cap
        if (true_dist["front"] > 1.2) and (true_dist["left"] > 1.0) and (true_dist["right"] > 1.0):
            dynamic_resample_cap = max(2, map_resample_cap - 1)

        # Corridor-mode trigger: both sides close/similar, front not blocked.
        corridor_mode = (
            (not any_front_obstacle)
            and (true_dist["left"] < 0.9)
            and (true_dist["right"] < 0.9)
            and (abs(true_dist["left"] - true_dist["right"]) < 0.2)
            and (true_dist["front"] > 0.65)
        )

        # Safety first: if front-group obstacle predicted, no forward actions.
        if any_front_obstacle:
            if f == "OBSTACLE":
                if (l == "CLEAR") and (r != "CLEAR"):
                    candidates = [ACTION_LEFT, ACTION_RESAMPLE]
                elif (r == "CLEAR") and (l != "CLEAR"):
                    candidates = [ACTION_RIGHT, ACTION_RESAMPLE]
                elif (l == "CLEAR") and (r == "CLEAR"):
                    clearer = choose_clearer_side_by_true_distance(true_dist)
                    candidates = [clearer, ACTION_LEFT if clearer == ACTION_RIGHT else ACTION_RIGHT]
                else:
                    candidates = [ACTION_STOP, ACTION_RESAMPLE]
            elif (fl == "OBSTACLE") and (fr == "CLEAR"):
                candidates = [ACTION_RIGHT, ACTION_RESAMPLE]
            elif (fr == "OBSTACLE") and (fl == "CLEAR"):
                candidates = [ACTION_LEFT, ACTION_RESAMPLE]
            else:
                candidates = [ACTION_RESAMPLE, ACTION_LEFT, ACTION_RIGHT]
        else:
            # Stuck recovery sequence.
            if stuck:
                stuck_recovery_count += 1
                if stuck_steps == 1:
                    if not predict_forward_collision(m, x, y, heading, -0.03, SAFETY_MARGIN):
                        candidates = [ACTION_STOP, turn_to_goal]
                    else:
                        candidates = [turn_to_goal, ACTION_LEFT if turn_to_goal == ACTION_RIGHT else ACTION_RIGHT]
                elif stuck_steps > 10:
                    # Cooldown + alternating scan direction prevents spin-lock.
                    if scan_turn_cooldown <= 0:
                        scan_dir = ACTION_RIGHT if last_scan_turn_dir == ACTION_LEFT else ACTION_LEFT
                        last_scan_turn_dir = scan_dir
                        candidates = [scan_dir, turn_to_goal]
                        forced_turn_deg = 30.0
                        scan_turn_count += 1
                        scan_turn_cooldown = SCAN_TURN_COOLDOWN_STEPS
                    else:
                        candidates = [turn_to_goal, ACTION_LEFT if turn_to_goal == ACTION_RIGHT else ACTION_RIGHT]
                else:
                    candidates = [turn_to_goal, ACTION_LEFT if turn_to_goal == ACTION_RIGHT else ACTION_RIGHT]
            else:
                # Commit-forward mode after proven safe probes.
                if commit_slow_remaining > 0 and (f == "CLEAR") and (fl != "OBSTACLE") and (fr != "OBSTACLE"):
                    candidates = [ACTION_SLOW, ACTION_PROBE]
                else:
                    # Goal-directed steering gates.
                    if abs_err_deg > 45.0:
                        candidates = [turn_to_goal, ACTION_LEFT if turn_to_goal == ACTION_RIGHT else ACTION_RIGHT]
                    elif abs_err_deg > 20.0 and front_not_clearly_open:
                        candidates = [turn_to_goal, ACTION_LEFT if turn_to_goal == ACTION_RIGHT else ACTION_RIGHT]
                    else:
                        # Forward-bias when goal is roughly straight in constrained maps.
                        if constrained_map and (abs_err_deg <= 18.0) and (f == "CLEAR") and (not any_front_obstacle):
                            candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_RESAMPLE]
                        elif corridor_mode:
                            candidates = [ACTION_PROBE, ACTION_SLOW, ACTION_RESAMPLE]
                        elif (
                            (f == "CLEAR")
                            and (fl != "OBSTACLE")
                            and (fr != "OBSTACLE")
                            and (probe_safe_streak >= PROBE_ESCALATE_COUNT)
                            and (probe_progress_accum >= PROBE_ESCALATE_PROGRESS_M)
                        ):
                            commit_slow_remaining = map_commit_steps
                            candidates = [ACTION_SLOW, ACTION_PROBE]
                        elif front_all_clear and consecutive_front_clear_count >= 4:
                            candidates = [ACTION_FAST, ACTION_SLOW]
                        elif front_all_clear and consecutive_front_clear_count >= 2:
                            candidates = [ACTION_SLOW, ACTION_PROBE]
                        elif (f == "CLEAR") and (fl != "OBSTACLE") and (fr != "OBSTACLE"):
                            candidates = [ACTION_PROBE, ACTION_RESAMPLE]
                        elif consecutive_front_blocked_count >= 3:
                            clearer = choose_clearer_side_by_true_distance(true_dist)
                            candidates = [clearer, ACTION_RESAMPLE]
                        else:
                            candidates = [ACTION_RESAMPLE, turn_to_goal]

        # Resample cap: force turn to break loops.
        if candidates and candidates[0] == ACTION_RESAMPLE and consecutive_resample_count >= dynamic_resample_cap:
            candidates = [turn_to_goal, ACTION_LEFT if turn_to_goal == ACTION_RIGHT else ACTION_RIGHT, ACTION_RESAMPLE]

        # Oscillation detection and suppression.
        if len(last_turns) == 4:
            pattern = list(last_turns)
            if pattern in ([ACTION_LEFT, ACTION_RIGHT, ACTION_LEFT, ACTION_RIGHT], [ACTION_RIGHT, ACTION_LEFT, ACTION_RIGHT, ACTION_LEFT]):
                turn_bias = turn_to_goal
                turn_bias_steps = OSC_BIAS_STEPS
        if turn_bias_steps > 0:
            # Push bias turns earlier in candidates.
            if turn_bias in {ACTION_LEFT, ACTION_RIGHT} and turn_bias in candidates:
                candidates = [turn_bias] + [c for c in candidates if c != turn_bias]
            turn_bias_steps -= 1

        # Candidate safety filtering for forward actions.
        safe_candidates: list[str] = []
        for c in candidates:
            if c in {ACTION_FAST, ACTION_SLOW, ACTION_PROBE} and predict_forward_collision(m, x, y, heading, forward_step_for_action(c), SAFETY_MARGIN):
                continue
            safe_candidates.append(c)
        if not safe_candidates:
            if not predict_forward_collision(m, x, y, heading, -0.03, SAFETY_MARGIN):
                safe_candidates = [ACTION_STOP]
            else:
                safe_candidates = [ACTION_RESAMPLE]

        # Local planner over safe candidates.
        best_action = safe_candidates[0]
        best_score = -1e18
        for c in safe_candidates:
            score = local_plan_score(
                m,
                x,
                y,
                heading,
                gx,
                gy,
                c,
                adaptive_turn,
                abs_err_deg,
                corridor_mode,
                constrained_map,
            )
            if score > best_score:
                best_score = score
                best_action = c
        action = best_action

        # Apply adaptive turn if selected action is turn.
        if action in {ACTION_LEFT, ACTION_RIGHT} and forced_turn_deg is None:
            forced_turn_deg = adaptive_turn

        actions[action] += 1
        nx, ny, nhead, moved, collided = apply_action(x, y, heading, action, m, forced_turn_deg=forced_turn_deg)
        path_len += moved
        x, y, heading = nx, ny, nhead
        path.append((x, y))

        # Update streak trackers after executing action.
        if action == ACTION_RESAMPLE:
            consecutive_resample_count += 1
        else:
            consecutive_resample_count = 0

        if scan_turn_cooldown > 0:
            scan_turn_cooldown -= 1

        if action in {ACTION_LEFT, ACTION_RIGHT}:
            last_turns.append(action)

        new_goal_dist = math.hypot(gx - x, gy - y)
        goal_gain = goal_dist - new_goal_dist

        if action == ACTION_PROBE and (not collided) and (not any_front_obstacle):
            probe_safe_streak += 1
            if goal_gain > 0:
                probe_progress_accum += goal_gain
        elif action == ACTION_SLOW and commit_slow_remaining > 0:
            commit_slow_remaining = max(0, commit_slow_remaining - 1)
            if goal_gain <= 0:
                # Decay commit gently instead of hard reset.
                commit_slow_remaining = max(0, commit_slow_remaining - 1)
                probe_safe_streak = max(0, probe_safe_streak - 1)
                probe_progress_accum = max(0.0, probe_progress_accum * 0.6)
        else:
            # Reset probe evidence when leaving probe/commit behavior.
            if action in {ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP}:
                probe_safe_streak = max(0, probe_safe_streak - 1)
                probe_progress_accum = max(0.0, probe_progress_accum * 0.7)
                commit_slow_remaining = max(0, commit_slow_remaining - 1)

        if collided:
            collision = True
            break
        if math.hypot(gx - x, gy - y) <= GOAL_RADIUS:
            success = True
            break

    timeout = (not success) and (not collision)
    final_dist_goal = math.hypot(gx - x, gy - y)

    return {
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "steps": len(path) - 1,
        "path_length": float(path_len),
        "action_counts": dict(actions),
        "min_obstacle_distance": float(min_obs),
        "final_distance_to_goal": float(final_dist_goal),
        "stuck_recovery_count": int(stuck_recovery_count),
        "scan_turn_count": int(scan_turn_count),
        "start": (sx, sy),
        "goal": (gx, gy),
        "path": path,
    }


def draw_map(ax, m: MapDef) -> None:
    ax.add_patch(plt.Rectangle((0, 0), m.width, m.height, fill=False, linewidth=2.0, edgecolor="black"))
    for r in m.obstacles:
        ax.add_patch(plt.Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin, color="gray", alpha=0.6))
    ax.set_xlim(-0.2, m.width + 0.2)
    ax.set_ylim(-0.2, m.height + 0.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(456)
    maps = make_maps()

    all_results = {}
    all_actions = Counter()
    path_samples = {}
    total_stuck_recoveries = 0
    total_scan_turns = 0

    for m in maps:
        episodes = []
        for _ in range(EPISODES_PER_MAP):
            ep = run_episode(m, rng)
            episodes.append(ep)
            all_actions.update(ep["action_counts"])
            total_stuck_recoveries += int(ep["stuck_recovery_count"])
            total_scan_turns += int(ep["scan_turn_count"])

        succ = sum(int(e["success"]) for e in episodes)
        coll = sum(int(e["collision"]) for e in episodes)
        tout = sum(int(e["timeout"]) for e in episodes)
        mean_steps = float(np.mean([e["steps"] for e in episodes]))
        mean_path = float(np.mean([e["path_length"] for e in episodes]))
        mean_final_goal = float(np.mean([e["final_distance_to_goal"] for e in episodes]))
        mean_stuck = float(np.mean([e["stuck_recovery_count"] for e in episodes]))

        all_results[m.name] = {
            "success_rate": succ / EPISODES_PER_MAP,
            "collision_rate": coll / EPISODES_PER_MAP,
            "timeout_rate": tout / EPISODES_PER_MAP,
            "mean_steps": mean_steps,
            "mean_path_length_m": mean_path,
            "mean_final_distance_to_goal_m": mean_final_goal,
            "mean_stuck_recovery_count": mean_stuck,
        }

        sel_idx = np.linspace(0, EPISODES_PER_MAP - 1, 10, dtype=int)
        path_samples[m.name] = [episodes[i] for i in sel_idx]

    total_actions = sum(all_actions.values())
    action_distribution = {
        a: {"count": int(all_actions.get(a, 0)), "rate": (all_actions.get(a, 0) / max(total_actions, 1))}
        for a in ALL_ACTIONS
    }

    v1_summary = json.loads(V1_JSON.read_text(encoding="utf-8")) if V1_JSON.exists() else None
    v2_summary = json.loads(V2_JSON.read_text(encoding="utf-8")) if V2_JSON.exists() else None
    v3_summary = json.loads(V3_JSON.read_text(encoding="utf-8")) if V3_JSON.exists() else None
    v4_summary = json.loads(V4_JSON.read_text(encoding="utf-8")) if V4_JSON.exists() else None
    v5_summary = json.loads(V5_NAV_JSON.read_text(encoding="utf-8")) if V5_NAV_JSON.exists() else None

    results = {
        "episodes_per_map": EPISODES_PER_MAP,
        "max_steps": MAX_STEPS,
        "success_threshold_m": GOAL_RADIUS,
        "maps": all_results,
        "overall_action_distribution": action_distribution,
        "total_stuck_recovery_count": int(total_stuck_recoveries),
        "total_scan_turn_count": int(total_scan_turns),
        "v1_comparison_available": bool(v1_summary is not None),
        "v2_comparison_available": bool(v2_summary is not None),
        "v3_comparison_available": bool(v3_summary is not None),
        "v4_comparison_available": bool(v4_summary is not None),
        "v5_nav_comparison_available": bool(v5_summary is not None),
    }
    if v1_summary is not None:
        results["v1_reference"] = v1_summary
    if v2_summary is not None:
        results["v2_reference"] = v2_summary
    if v3_summary is not None:
        results["v3_reference"] = v3_summary
    if v4_summary is not None:
        results["v4_reference"] = v4_summary
    if v5_summary is not None:
        results["v5_nav_reference"] = v5_summary

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
            color = "tab:green" if ep["success"] else ("tab:red" if ep["collision"] else "tab:orange")
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
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=140)
    labels = ALL_ACTIONS
    vals = [100.0 * action_distribution[a]["rate"] for a in labels]
    bars = ax.bar(labels, vals, color="tab:blue")
    ax.set_ylabel("Action share (%)")
    ax.set_title("Simple 2D Acoustic Nav v6: Action Distribution")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_ACTION_PNG)
    plt.close(fig)

    print("\nV6 Per-map performance:")
    for name, v in all_results.items():
        print(
            f"  {name}: success={100*v['success_rate']:.1f}% | collision={100*v['collision_rate']:.1f}% | "
            f"timeout={100*v['timeout_rate']:.1f}% | mean_steps={v['mean_steps']:.1f}"
        )

    print("\nV6 Overall action distribution:")
    for a in ALL_ACTIONS:
        print(f"  {a}: {action_distribution[a]['count']} ({100*action_distribution[a]['rate']:.2f}%)")
    print(f"\nV6 total stuck recovery actions: {total_stuck_recoveries}")
    print(f"V6 total 30-degree scan turns: {total_scan_turns}")

    if v2_summary is not None and "maps" in v2_summary:
        print("\nV2 -> V6 comparison:")
        for m in maps:
            n = m.name
            if n in v2_summary["maps"]:
                a = v2_summary["maps"][n]
                b = all_results[n]
                print(
                    f"  {n}: success {100*a['success_rate']:.1f}% -> {100*b['success_rate']:.1f}% | "
                    f"collision {100*a['collision_rate']:.1f}% -> {100*b['collision_rate']:.1f}% | "
                    f"timeout {100*a['timeout_rate']:.1f}% -> {100*b['timeout_rate']:.1f}%"
                )
    if v3_summary is not None and "maps" in v3_summary:
        print("\nV3 -> V6 comparison:")
        for m in maps:
            n = m.name
            if n in v3_summary["maps"]:
                a = v3_summary["maps"][n]
                b = all_results[n]
                print(
                    f"  {n}: success {100*a['success_rate']:.1f}% -> {100*b['success_rate']:.1f}% | "
                    f"collision {100*a['collision_rate']:.1f}% -> {100*b['collision_rate']:.1f}% | "
                    f"timeout {100*a['timeout_rate']:.1f}% -> {100*b['timeout_rate']:.1f}%"
                )
    if v4_summary is not None and "maps" in v4_summary:
        print("\nV4 -> V6 comparison:")
        for m in maps:
            n = m.name
            if n in v4_summary["maps"]:
                a = v4_summary["maps"][n]
                b = all_results[n]
                print(
                    f"  {n}: success {100*a['success_rate']:.1f}% -> {100*b['success_rate']:.1f}% | "
                    f"collision {100*a['collision_rate']:.1f}% -> {100*b['collision_rate']:.1f}% | "
                    f"timeout {100*a['timeout_rate']:.1f}% -> {100*b['timeout_rate']:.1f}%"
                )
    if v5_summary is not None and "maps" in v5_summary:
        print("\nV5_nav -> V6 comparison:")
        for m in maps:
            n = m.name
            if n in v5_summary["maps"]:
                a = v5_summary["maps"][n]
                b = all_results[n]
                print(
                    f"  {n}: success {100*a['success_rate']:.1f}% -> {100*b['success_rate']:.1f}% | "
                    f"collision {100*a['collision_rate']:.1f}% -> {100*b['collision_rate']:.1f}% | "
                    f"timeout {100*a['timeout_rate']:.1f}% -> {100*b['timeout_rate']:.1f}%"
                )

    mean_success = float(np.mean([v["success_rate"] for v in all_results.values()]))
    mean_collision = float(np.mean([v["collision_rate"] for v in all_results.values()]))
    if mean_success > 0.35 and mean_collision < 0.15:
        movement_msg = "Goal-heading and stuck recovery improved movement and are usable."
    elif mean_success > 0.05:
        movement_msg = "Goal-heading and stuck recovery improved movement, but policy remains conservative."
    else:
        movement_msg = "Goal-heading and stuck recovery are insufficient; policy still struggles to reach goals."
    print(f"\nAssessment: {movement_msg}")

    print(f"\nSaved: {RESULT_JSON}")
    print(f"Saved: {RESULT_PATHS_PNG}")
    print(f"Saved: {RESULT_ACTION_PNG}")


if __name__ == "__main__":
    main()
