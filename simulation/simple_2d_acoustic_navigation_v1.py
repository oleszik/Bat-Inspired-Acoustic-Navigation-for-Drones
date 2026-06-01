"""
Simple 2D acoustic navigation simulator (v1) for policy v5 temporal confirmation.

This simulator tests whether the current multi-sector acoustic policy can navigate
basic maps safely and with useful motion. It does not retrain any model.
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
RESULT_JSON = RESULT_DIR / "simple_2d_acoustic_nav_v1_results.json"
RESULT_PATHS_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v1_paths.png"
RESULT_ACTION_PNG = RESULT_DIR / "simple_2d_acoustic_nav_v1_action_distribution.png"

EPISODES_PER_MAP = 100
MAX_STEPS = 250
ROBOT_RADIUS = 0.12
GOAL_RADIUS = 0.35
RAY_MAX_RANGE = 3.0
RAY_STEP = 0.02

SECTOR_NAMES = ["left", "front_left", "front", "front_right", "right"]
SECTOR_OFFSETS_RAD = {
    "left": math.radians(90.0),
    "front_left": math.radians(40.0),
    "front": 0.0,
    "front_right": math.radians(-40.0),
    "right": math.radians(-90.0),
}
SECTOR_CONE_HALF_RAD = math.radians(12.0)
SECTOR_CONE_SAMPLES = 5

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
    # Outside room bounds is treated as collision.
    if px < robot_radius or py < robot_radius or px > (m.width - robot_radius) or py > (m.height - robot_radius):
        return True
    # Inflate obstacles by robot radius.
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
    d = {}
    for sec in SECTOR_NAMES:
        center = heading + SECTOR_OFFSETS_RAD[sec]
        angles = np.linspace(center - SECTOR_CONE_HALF_RAD, center + SECTOR_CONE_HALF_RAD, SECTOR_CONE_SAMPLES)
        vals = [raycast_distance(m, x, y, a, RAY_MAX_RANGE, RAY_STEP) for a in angles]
        d[sec] = float(min(vals))
    return d


def synthesize_sector_perception(true_d: float, rng: np.random.Generator) -> Dict[str, float | str | int]:
    """
    Conservative synthetic perception behavior:
    - close obstacles mostly OBSTACLE
    - ambiguous mid/far ranges often UNCERTAIN
    - clear space can still become UNCERTAIN
    """
    # Obstacle/clear likelihood from true distance.
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
    p_clear = max(0.0, 1.0 - p_obs - p_unc)

    u = rng.random()
    if u < p_obs:
        state = "OBSTACLE"
    elif u < (p_obs + p_unc):
        state = "UNCERTAIN"
    else:
        state = "CLEAR"

    # Matched-filter detectability: weaker at longer range and under weak returns.
    if state == "OBSTACLE":
        detect_prob = float(np.clip(1.05 - 0.45 * true_d, 0.05, 0.98))
        matched_peak = int(rng.random() < detect_prob)
    else:
        matched_peak = 0

    # Feature synthesis consistent with prior policy thresholds.
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


def is_strong_clear(sec: Dict[str, float | str | int]) -> bool:
    return (
        str(sec["predicted_state"]).upper() == "CLEAR"
        and int(sec["matched_filter_peak_exists"]) == 0
        and float(sec["echo_validity_probability"]) <= 0.05
        and float(sec["peak_snr"]) <= 1.0
        and float(sec["peak_prominence"]) <= 0.10
    )


def is_soft_clear(sec: Dict[str, float | str | int]) -> bool:
    return (
        str(sec["predicted_state"]).upper() == "CLEAR"
        and int(sec["matched_filter_peak_exists"]) == 0
        and float(sec["echo_validity_probability"]) <= 0.10
        and float(sec["peak_snr"]) <= 1.5
        and float(sec["peak_prominence"]) <= 0.20
    )


def choose_turn_side_when_both_clear(sec: Dict[str, Dict[str, float | str | int]]) -> str:
    l = sec["left"]["predicted_distance_m"]
    r = sec["right"]["predicted_distance_m"]
    l_ok = np.isfinite(l) if isinstance(l, (float, np.floating)) else False
    r_ok = np.isfinite(r) if isinstance(r, (float, np.floating)) else False
    if l_ok and r_ok:
        return ACTION_LEFT if float(l) >= float(r) else ACTION_RIGHT
    if l_ok:
        return ACTION_LEFT
    if r_ok:
        return ACTION_RIGHT
    return ACTION_LEFT


def fallback_v4_turn_resample_stop(sec: Dict[str, Dict[str, float | str | int]]) -> str:
    l = str(sec["left"]["predicted_state"]).upper()
    fl = str(sec["front_left"]["predicted_state"]).upper()
    f = str(sec["front"]["predicted_state"]).upper()
    fr = str(sec["front_right"]["predicted_state"]).upper()
    r = str(sec["right"]["predicted_state"]).upper()

    any_front_obs = (fl == "OBSTACLE") or (f == "OBSTACLE") or (fr == "OBSTACLE")
    if (l == "OBSTACLE") and (r == "OBSTACLE") and any_front_obs:
        return ACTION_STOP
    if f == "OBSTACLE":
        if (l == "CLEAR") and (r != "CLEAR"):
            return ACTION_LEFT
        if (r == "CLEAR") and (l != "CLEAR"):
            return ACTION_RIGHT
        if (l == "CLEAR") and (r == "CLEAR"):
            return choose_turn_side_when_both_clear(sec)
        return ACTION_STOP
    if f == "CLEAR":
        if (fl == "OBSTACLE") and (fr == "CLEAR"):
            return ACTION_RIGHT
        if (fr == "OBSTACLE") and (fl == "CLEAR"):
            return ACTION_LEFT
        return ACTION_RESAMPLE
    # front uncertain
    if (l == "CLEAR") and (r != "CLEAR"):
        return ACTION_LEFT
    if (r == "CLEAR") and (l != "CLEAR"):
        return ACTION_RIGHT
    return ACTION_RESAMPLE


def policy_v5_temporal(sec_history: deque) -> str:
    """
    Temporal window uses last up to 3 frames.
    """
    if len(sec_history) == 0:
        return ACTION_RESAMPLE

    frames = list(sec_history)[-3:]
    any_front_obstacle_prediction = False
    strong_all_three_count = 0
    slow_rule_count = 0
    strong_front_count = 0

    for sec in frames:
        fl_state = str(sec["front_left"]["predicted_state"]).upper()
        f_state = str(sec["front"]["predicted_state"]).upper()
        fr_state = str(sec["front_right"]["predicted_state"]).upper()
        any_front_obstacle_prediction |= (fl_state == "OBSTACLE") or (f_state == "OBSTACLE") or (fr_state == "OBSTACLE")

        strong_all = is_strong_clear(sec["front_left"]) and is_strong_clear(sec["front"]) and is_strong_clear(sec["front_right"])
        slow_ok = is_strong_clear(sec["front"]) and is_soft_clear(sec["front_left"]) and is_soft_clear(sec["front_right"]) and (not any_front_obstacle_prediction)
        strong_front = is_strong_clear(sec["front"])

        strong_all_three_count += int(strong_all)
        slow_rule_count += int(slow_ok)
        strong_front_count += int(strong_front)

    current = frames[-1]
    cur_fl = str(current["front_left"]["predicted_state"]).upper()
    cur_f = str(current["front"]["predicted_state"]).upper()
    cur_fr = str(current["front_right"]["predicted_state"]).upper()

    # Temporal forward rule.
    if any_front_obstacle_prediction:
        action = fallback_v4_turn_resample_stop(current)
    elif len(frames) == 3 and strong_all_three_count == 3:
        action = ACTION_FAST
    elif len(frames) >= 2 and slow_rule_count >= 2:
        action = ACTION_SLOW
    elif strong_front_count >= 1:
        action = ACTION_PROBE
    else:
        action = fallback_v4_turn_resample_stop(current)

    # Final full-forward safety override.
    if (action in {ACTION_FAST, ACTION_SLOW}) and ((cur_fl != "CLEAR") or (cur_f != "CLEAR") or (cur_fr != "CLEAR")):
        action = ACTION_RESAMPLE

    return action


def apply_action(x: float, y: float, heading: float, action: str, m: MapDef) -> Tuple[float, float, float, float, bool]:
    """
    Returns (new_x, new_y, new_heading, moved_distance, collision).
    """
    step_fwd = {ACTION_FAST: 0.20, ACTION_SLOW: 0.08, ACTION_PROBE: 0.03}
    turn = {ACTION_LEFT: math.radians(15.0), ACTION_RIGHT: math.radians(-15.0)}

    if action in turn:
        return x, y, wrap_angle(heading + turn[action]), 0.0, False
    if action == ACTION_RESAMPLE:
        return x, y, heading, 0.0, False

    if action == ACTION_STOP:
        dist = -0.03
    else:
        dist = step_fwd.get(action, 0.0)

    # Segment collision check.
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
                Rect(5.5, 2.2, 6.5, 4.0),  # pinch
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
    # Ensure useful separation.
    tries = 0
    while math.hypot(gx - sx, gy - sy) < 3.0 and tries < 200:
        gx, gy = sample_free_pose(m, rng)
        tries += 1

    heading = rng.uniform(-math.pi, math.pi)
    x, y = sx, sy
    path = [(x, y)]
    history = deque(maxlen=3)
    actions = Counter()
    min_obs = float("inf")
    path_len = 0.0
    collision = False
    success = False

    for step in range(1, MAX_STEPS + 1):
        true_dist = sector_true_distances(m, x, y, heading)
        min_obs = min(min_obs, min(true_dist.values()))

        sec = {s: synthesize_sector_perception(true_dist[s], rng) for s in SECTOR_NAMES}
        history.append(sec)
        action = policy_v5_temporal(history)
        actions[action] += 1

        nx, ny, nhead, moved, collided = apply_action(x, y, heading, action, m)
        path_len += moved
        x, y, heading = nx, ny, nhead
        path.append((x, y))

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
    rng = np.random.default_rng(42)
    maps = make_maps()

    all_results = {}
    all_actions = Counter()
    path_samples = {}

    for m in maps:
        episodes = []
        for _ in range(EPISODES_PER_MAP):
            ep = run_episode(m, rng)
            episodes.append(ep)
            all_actions.update(ep["action_counts"])

        succ = sum(int(e["success"]) for e in episodes)
        coll = sum(int(e["collision"]) for e in episodes)
        tout = sum(int(e["timeout"]) for e in episodes)
        mean_steps = float(np.mean([e["steps"] for e in episodes]))
        mean_path = float(np.mean([e["path_length"] for e in episodes]))
        mean_final_goal = float(np.mean([e["final_distance_to_goal"] for e in episodes]))

        all_results[m.name] = {
            "success_rate": succ / EPISODES_PER_MAP,
            "collision_rate": coll / EPISODES_PER_MAP,
            "timeout_rate": tout / EPISODES_PER_MAP,
            "mean_steps": mean_steps,
            "mean_path_length_m": mean_path,
            "mean_final_distance_to_goal_m": mean_final_goal,
        }

        # Keep a few trajectories for plotting.
        sel_idx = np.linspace(0, EPISODES_PER_MAP - 1, 10, dtype=int)
        path_samples[m.name] = [episodes[i] for i in sel_idx]

    total_actions = sum(all_actions.values())
    action_distribution = {a: {"count": int(all_actions.get(a, 0)), "rate": (all_actions.get(a, 0) / max(total_actions, 1))} for a in ALL_ACTIONS}

    # Overall interpretation.
    mean_success = float(np.mean([v["success_rate"] for v in all_results.values()]))
    mean_collision = float(np.mean([v["collision_rate"] for v in all_results.values()]))
    stop_resample_rate = (
        action_distribution[ACTION_STOP]["rate"] + action_distribution[ACTION_RESAMPLE]["rate"]
    )
    if mean_collision <= 0.05 and mean_success >= 0.50:
        policy_status = "usable"
    elif mean_collision > 0.10:
        policy_status = "unsafe"
    else:
        policy_status = "too_conservative"

    results = {
        "episodes_per_map": EPISODES_PER_MAP,
        "max_steps": MAX_STEPS,
        "maps": all_results,
        "overall_action_distribution": action_distribution,
        "overall_mean_success_rate": mean_success,
        "overall_mean_collision_rate": mean_collision,
        "overall_stop_resample_rate": stop_resample_rate,
        "policy_assessment": policy_status,
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
            color = "tab:green" if ep["success"] else ("tab:red" if ep["collision"] else "tab:orange")
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.5, linewidth=1.2)
            sx, sy = ep["start"]
            gx, gy = ep["goal"]
            ax.scatter([sx], [sy], c="blue", s=12)
            ax.scatter([gx], [gy], c="black", s=12, marker="x")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULT_PATHS_PNG)
    plt.close(fig)

    # Action distribution plot.
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=140)
    labels = ALL_ACTIONS
    vals = [100.0 * action_distribution[a]["rate"] for a in labels]
    bars = ax.bar(labels, vals, color="tab:blue")
    ax.set_ylabel("Action share (%)")
    ax.set_title("Simple 2D Acoustic Nav v1: Action Distribution")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_ACTION_PNG)
    plt.close(fig)

    # Print summary.
    print("\nPer-map performance:")
    for name, v in all_results.items():
        print(
            f"  {name}: success={100*v['success_rate']:.1f}% | collision={100*v['collision_rate']:.1f}% | "
            f"timeout={100*v['timeout_rate']:.1f}% | mean_steps={v['mean_steps']:.1f}"
        )

    print("\nOverall action distribution:")
    for a in ALL_ACTIONS:
        print(f"  {a}: {action_distribution[a]['count']} ({100*action_distribution[a]['rate']:.2f}%)")

    print("\nPolicy assessment:")
    if policy_status == "too_conservative":
        print("  Current policy is safe but too conservative in this simulator.")
    elif policy_status == "unsafe":
        print("  Current policy appears unsafe in this simulator.")
    else:
        print("  Current policy appears usable for basic navigation in this simulator.")

    print(f"\nSaved: {RESULT_JSON}")
    print(f"Saved: {RESULT_PATHS_PNG}")
    print(f"Saved: {RESULT_ACTION_PNG}")


if __name__ == "__main__":
    main()

