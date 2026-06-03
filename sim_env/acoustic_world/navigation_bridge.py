from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .agent import AcousticAgent
from .environments import EnvironmentMap, world_to_cell


DEFAULT_NAV_MANIFEST = Path("runs/accepted_models/phase2d_mapper_guided_navigation_v3/manifest.json")


def load_navigation_manifest(path: str | Path = DEFAULT_NAV_MANIFEST) -> Dict[str, object]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def get_navigation_config(manifest: Dict[str, object]) -> Dict[str, object]:
    return {
        "accepted_model_name": manifest.get("accepted_model_name", "unknown"),
        "status": manifest.get("status", "unknown"),
        "results_json": manifest.get("results_json", ""),
        "accepted_metrics": manifest.get("accepted_metrics", {}),
    }


def _safe_forward(agent: AcousticAgent, env: EnvironmentMap, step_m: float) -> bool:
    nx = agent.x + step_m * math.cos(agent.theta)
    ny = agent.y + step_m * math.sin(agent.theta)
    h, w = env.shape
    if nx < agent.radius_m or ny < agent.radius_m or nx > env.width_m - agent.radius_m or ny > env.height_m - agent.radius_m:
        return False
    cx, cy = world_to_cell(nx, ny, env.cell_size, w, h)
    return not bool(env.ground_truth_occupancy[cy, cx])


def _frontier_target(confidence_map: np.ndarray, free_prob: np.ndarray) -> Optional[Tuple[int, int]]:
    h, w = confidence_map.shape
    best_score = -1e9
    best = None
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if free_prob[y, x] < 0.52:
                continue
            local = confidence_map[y - 1 : y + 2, x - 1 : x + 2]
            score = float((1.0 - np.mean(local)) + 0.25 * free_prob[y, x])
            if score > best_score:
                best_score = score
                best = (x, y)
    return best


def _low_confidence_frontiers(confidence_map: np.ndarray, free_prob: np.ndarray) -> List[Tuple[int, int]]:
    h, w = confidence_map.shape
    cells: List[Tuple[int, int]] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if free_prob[y, x] < 0.50:
                continue
            if confidence_map[y, x] < 0.45:
                cells.append((x, y))
    return cells


def init_navigation_state() -> Dict[str, Any]:
    return {
        "step": 0,
        "policy_mode": "simple",
        "current_target": None,
        "target_age": 0,
        "target_commit_steps": 55,
        "frontier_commit_steps": 20,
        "frontier_switch_min_distance_m": 0.5,
        "current_frontier_target": None,
        "frontier_commit_remaining": 0,
        "last_frontier_target": None,
        "frontier_commitment_count": 0,
        "premature_frontier_switch_count": 0,
        "target_reached_count": 0,
        "frontier_switch_reason": "init",
        "probe_streak": 0,
        "turn_streak": 0,
        "stagnation_event_count": 0,
        "forced_frontier_switch_count": 0,
        "safe_forward_preference_count": 0,
        "safe_forward_chosen_count": 0,
        "safe_forward_blocked_count": 0,
        "doorway_for_coverage_count": 0,
        "doorway_acceptance_threshold": 0.62,
        "doorway_behind_low_confidence_requirement": 0.25,
        "doorway_rejection_reasons": {
            "rejected_by_low_probability": 0,
            "rejected_by_collision_risk": 0,
            "rejected_by_low_behind_confidence_gain": 0,
            "rejected_by_structure_gate": 0,
        },
        "visited_counts": {},
        "revisit_penalty_weight": 0.55,
        "high_revisit_cell_count": 0,
        # v1.4 route-committed policy settings/state.
        "route_commit_steps": 50,
        "target_reached_radius_m": 0.35,
        "max_blocked_steps": 8,
        "heading_tolerance_deg": 15.0,
        "current_route_target": None,
        "route_commit_remaining": 0,
        "route_commitment_count": 0,
        "route_replan_count": 0,
        "last_distance_to_target": None,
        "distance_to_target": None,
        "distance_to_target_delta": 0.0,
        "progress_toward_target_count": 0,
        "no_progress_count": 0,
        "blocked_step_count": 0,
        "route_target_switch_reason": "init",
        "doorway_target_selected_count": 0,
        "doorway_target_reached_count": 0,
        "doorway_acceptance_reason": "none",
        # v1.5 coverage-sweep policy settings/state.
        "sweep_forward_burst_steps": 8,
        "sweep_turn_angle_steps": 3,
        "sweep_scan_interval": 25,
        "sweep_low_confidence_weight": 1.0,
        "sweep_unvisited_weight": 1.0,
        "sweep_collision_clearance_cells": 2,
        "sweep_oscillation_penalty": 1.0,
        "sweep_burst_remaining": 8,
        "sweep_pending_turn_steps": 0,
        "sweep_turn_sign": 1,
        "sweep_steps_since_scan": 0,
        "sweep_heading_rad": 0.0,
        "sweep_forward_burst_count": 0,
        "sweep_blocked_turn_count": 0,
        "sweep_scan_turn_count": 0,
        "sweep_direction_change_count": 0,
        "forward_safe_count": 0,
        "forward_blocked_count": 0,
        "low_confidence_direction_score_sum": 0.0,
        "unvisited_direction_score_sum": 0.0,
        "direction_score_count": 0,
        "oscillation_prevented_count": 0,
        "last_turn_sign": 0,
        "sweep_last_reason": "init",
        "frontier_blacklist": set(),
        "coverage_history": [],
        "recent_targets": [],
        "last_stagnation_recovery_active": False,
        # v1.6 adaptive hybrid policy settings/state.
        "adaptive_min_mode_steps": 40,
        "adaptive_coverage_gain_window": 50,
        "adaptive_selected_subpolicy": "stabilized_frontier",
        "adaptive_switch_reason": "init",
        "adaptive_switch_count": 0,
        "adaptive_mode_steps": 0,
        "steps_in_stabilized_frontier": 0,
        "steps_in_coverage_sweep": 0,
        "adaptive_switch_reason_counts": {},
        "adaptive_recent_actions": [],
        "adaptive_forward_blocked_streak": 0,
        "local_obstacle_density": 0.0,
        "local_free_space_ratio": 0.0,
        "local_wall_structure_score": 0.0,
        "local_doorway_score": 0.0,
        "local_clutter_score": 0.0,
        "local_corridor_score": 0.0,
    }


def _frontier_score(
    x: int,
    y: int,
    agent: AcousticAgent,
    env: EnvironmentMap,
    conf: np.ndarray,
    free_prob: np.ndarray,
    wall_prob: np.ndarray,
    door_prob: np.ndarray,
    stagnating: bool,
    recent_targets: List[Tuple[int, int]],
) -> float:
    wx = (x + 0.5) * env.cell_size
    wy = (y + 0.5) * env.cell_size
    dx = wx - agent.x
    dy = wy - agent.y
    dist = float(np.hypot(dx, dy))
    target_ang = math.atan2(dy, dx)
    err = abs((target_ang - agent.theta + math.pi) % (2.0 * math.pi) - math.pi)
    heading_bonus = 1.0 - min(1.0, err / math.pi)
    local = conf[max(0, y - 1) : min(conf.shape[0], y + 2), max(0, x - 1) : min(conf.shape[1], x + 2)]
    expected_new_cov = float(1.0 - np.mean(local))
    wall_risk = float(wall_prob[y, x])
    fake_door_risk = float(door_prob[y, x] * (1.0 - wall_prob[y, x]))
    revisit_penalty = 0.15 if (x, y) in recent_targets[-12:] else 0.0
    dist_term = (-0.10 * dist) if not stagnating else (+0.08 * dist)
    return (
        2.6 * expected_new_cov
        + 0.85 * heading_bonus
        + dist_term
        - 1.10 * wall_risk
        - 0.70 * fake_door_risk
        - revisit_penalty
        + 0.35 * float(free_prob[y, x])
    )


def _select_frontier_target_v3_like(
    env: EnvironmentMap,
    agent: AcousticAgent,
    conf: np.ndarray,
    free_prob: np.ndarray,
    wall_prob: np.ndarray,
    door_prob: np.ndarray,
    nav_state: Dict[str, Any],
    stagnating: bool,
) -> Tuple[Optional[Tuple[int, int]], float]:
    h, w = conf.shape
    blacklist: Set[Tuple[int, int]] = nav_state.get("frontier_blacklist", set())
    recent_targets: List[Tuple[int, int]] = nav_state.get("recent_targets", [])
    best = None
    best_score = -1e9
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if free_prob[y, x] < 0.52:
                continue
            if conf[y, x] > 0.78:
                continue
            if (x, y) in blacklist:
                continue
            s = _frontier_score(
                x=x,
                y=y,
                agent=agent,
                env=env,
                conf=conf,
                free_prob=free_prob,
                wall_prob=wall_prob,
                door_prob=door_prob,
                stagnating=stagnating,
                recent_targets=recent_targets,
            )
            if s > best_score:
                best = (x, y)
                best_score = s
    return best, float(best_score)


def _select_frontier_target_stabilized(
    env: EnvironmentMap,
    agent: AcousticAgent,
    conf: np.ndarray,
    free_prob: np.ndarray,
    wall_prob: np.ndarray,
    door_prob: np.ndarray,
    nav_state: Dict[str, Any],
    stagnating: bool,
) -> Tuple[Optional[Tuple[int, int]], float]:
    h, w = conf.shape
    blacklist: Set[Tuple[int, int]] = nav_state.get("frontier_blacklist", set())
    recent_targets: List[Tuple[int, int]] = nav_state.get("recent_targets", [])
    visited_counts: Dict[Tuple[int, int], int] = nav_state.get("visited_counts", {})
    revisit_penalty_weight = float(nav_state.get("revisit_penalty_weight", 0.55))
    min_switch_dist = float(nav_state.get("frontier_switch_min_distance_m", 0.5))
    last_target = nav_state.get("last_frontier_target")
    best = None
    best_score = -1e9
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if free_prob[y, x] < 0.52:
                continue
            if conf[y, x] > 0.80:
                continue
            if (x, y) in blacklist:
                continue
            s = _frontier_score(
                x=x,
                y=y,
                agent=agent,
                env=env,
                conf=conf,
                free_prob=free_prob,
                wall_prob=wall_prob,
                door_prob=door_prob,
                stagnating=stagnating,
                recent_targets=recent_targets,
            )
            visit_count = int(visited_counts.get((x, y), 0))
            s -= revisit_penalty_weight * min(1.0, visit_count / 4.0)
            if last_target is not None:
                lx, ly = last_target
                dist_to_last = float(np.hypot((x - lx) * env.cell_size, (y - ly) * env.cell_size))
                if dist_to_last < min_switch_dist:
                    s -= 0.40
            if s > best_score:
                best = (x, y)
                best_score = s
    return best, float(best_score)


def _forward_target_cell(agent: AcousticAgent, env: EnvironmentMap, dist_m: float = 0.8) -> Tuple[int, int]:
    px = agent.x + dist_m * math.cos(agent.theta)
    py = agent.y + dist_m * math.sin(agent.theta)
    h, w = env.shape
    cx, cy = world_to_cell(px, py, env.cell_size, w, h)
    cx = int(max(0, min(w - 1, cx)))
    cy = int(max(0, min(h - 1, cy)))
    return cx, cy


def _direction_probe_cell(
    agent: AcousticAgent,
    env: EnvironmentMap,
    angle_offset_rad: float,
    dist_m: float = 0.9,
) -> Tuple[int, int]:
    px = agent.x + dist_m * math.cos(agent.theta + angle_offset_rad)
    py = agent.y + dist_m * math.sin(agent.theta + angle_offset_rad)
    h, w = env.shape
    cx, cy = world_to_cell(px, py, env.cell_size, w, h)
    cx = int(max(0, min(w - 1, cx)))
    cy = int(max(0, min(h - 1, cy)))
    return cx, cy


def _run_length(mask: np.ndarray, axis: int) -> int:
    best = 0
    arr = mask if axis == 1 else mask.T
    for row in arr:
        current = 0
        for value in row:
            if bool(value):
                current += 1
                best = max(best, current)
            else:
                current = 0
    return int(best)


def _local_structure_diagnostics(
    env: EnvironmentMap,
    agent: AcousticAgent,
    sensor_obs: Dict[str, np.ndarray],
    predicted_maps: Dict[str, np.ndarray],
) -> Dict[str, float]:
    free_prob = predicted_maps["free_prob"]
    door_prob = predicted_maps["doorway_prob"]
    wall_prob = predicted_maps.get("wall_prob", np.zeros_like(free_prob))
    occ_prob = predicted_maps.get("occupancy_prob", 1.0 - free_prob)
    h, w = env.shape
    cx, cy = agent.cell(env)
    radius = 4
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    local_occ = occ_prob[y0:y1, x0:x1]
    local_free = free_prob[y0:y1, x0:x1]
    local_wall = wall_prob[y0:y1, x0:x1]
    local_door = door_prob[y0:y1, x0:x1]

    obstacle_density = float(np.mean(local_occ > 0.52)) if local_occ.size else 0.0
    free_space_ratio = float(np.mean(local_free > 0.52)) if local_free.size else 0.0
    wall_cells = local_wall > 0.18
    wall_density = float(np.mean(wall_cells)) if wall_cells.size else 0.0
    max_wall_run = max(_run_length(wall_cells, axis=0), _run_length(wall_cells, axis=1)) if wall_cells.size else 0
    wall_structure_score = float(min(1.0, (max_wall_run / max(1, radius * 2 + 1)) + 0.35 * wall_density))
    doorway_score = float(max(np.max(local_door) if local_door.size else 0.0, np.mean(local_door) if local_door.size else 0.0))

    ray_dist = sensor_obs["ray_distances_m"]
    front_dist = float(ray_dist[len(ray_dist) // 2])
    left_dist = float(np.mean(ray_dist[:3]))
    right_dist = float(np.mean(ray_dist[-3:]))
    side_balance = 1.0 - min(1.0, abs(left_dist - right_dist) / max(0.01, left_dist + right_dist))
    narrow_side_score = max(0.0, 1.0 - min(1.0, (left_dist + right_dist) / 1.2))
    long_front_score = min(1.0, front_dist / 1.4)
    corridor_score = float(min(1.0, 0.45 * wall_structure_score + 0.35 * side_balance + 0.20 * long_front_score))
    clutter_score = float(
        min(
            1.0,
            0.45 * obstacle_density
            + 0.30 * max(0.0, 1.0 - wall_structure_score)
            + 0.25 * max(0.0, 1.0 - side_balance),
        )
    )

    return {
        "local_obstacle_density": obstacle_density,
        "local_free_space_ratio": free_space_ratio,
        "local_wall_structure_score": wall_structure_score,
        "local_doorway_score": doorway_score,
        "local_clutter_score": clutter_score,
        "local_corridor_score": corridor_score,
        "local_narrow_passage_score": float(max(narrow_side_score, min(doorway_score, wall_structure_score))),
    }


def _adaptive_coverage_gain(nav_state: Dict[str, Any]) -> float:
    coverage_history = nav_state.get("coverage_history", [])
    window = int(nav_state.get("adaptive_coverage_gain_window", 50))
    if not isinstance(coverage_history, list) or len(coverage_history) <= window:
        return 1.0
    return float(coverage_history[-1] - coverage_history[-window - 1])


def _select_adaptive_subpolicy(nav_state: Dict[str, Any], diagnostics: Dict[str, float], front_dist: float) -> str:
    current = str(nav_state.get("adaptive_selected_subpolicy", "stabilized_frontier"))
    mode_steps = int(nav_state.get("adaptive_mode_steps", 0)) + 1
    nav_state["adaptive_mode_steps"] = mode_steps
    min_steps = int(nav_state.get("adaptive_min_mode_steps", 40))
    coverage_gain = _adaptive_coverage_gain(nav_state)
    recent_actions = nav_state.get("adaptive_recent_actions", [])
    if not isinstance(recent_actions, list):
        recent_actions = []
    recent_turns = sum(1 for action in recent_actions[-50:] if action in ("turn_left", "turn_right"))
    blocked_streak = int(nav_state.get("adaptive_forward_blocked_streak", 0))

    reason = str(nav_state.get("adaptive_switch_reason", "hold"))
    target_subpolicy = current
    can_switch = mode_steps >= min_steps

    doorway_like = diagnostics["local_doorway_score"] >= 0.42
    corridor_like = (
        diagnostics["local_corridor_score"] >= 0.66
        and diagnostics["local_wall_structure_score"] >= 0.52
        and diagnostics["local_obstacle_density"] < 0.32
    )
    narrow_passage = diagnostics["local_narrow_passage_score"] >= 0.52 and diagnostics["local_wall_structure_score"] >= 0.22
    cluttered = diagnostics["local_clutter_score"] >= 0.48 or diagnostics["local_obstacle_density"] >= 0.36
    low_gain_turning = recent_turns >= 26 and coverage_gain < 0.018
    repeatedly_blocked = blocked_streak >= 3 or front_dist < 0.11
    sweep_plateau = current == "coverage_sweep" and coverage_gain < 0.010

    if current == "stabilized_frontier":
        if can_switch and cluttered:
            target_subpolicy = "coverage_sweep"
            reason = "high_clutter_or_obstacle_density"
        elif can_switch and repeatedly_blocked:
            target_subpolicy = "coverage_sweep"
            reason = "repeated_forward_block"
        elif can_switch and low_gain_turning:
            target_subpolicy = "coverage_sweep"
            reason = "high_turn_low_coverage_gain"
        else:
            reason = "hold_stabilized_frontier"
    else:
        if can_switch and doorway_like:
            target_subpolicy = "stabilized_frontier"
            reason = "doorway_candidate_present"
        elif can_switch and corridor_like:
            target_subpolicy = "stabilized_frontier"
            reason = "corridor_wall_structure"
        elif can_switch and sweep_plateau:
            target_subpolicy = "stabilized_frontier"
            reason = "sweep_coverage_plateau"
        elif can_switch and narrow_passage:
            target_subpolicy = "stabilized_frontier"
            reason = "near_narrow_passage"
        else:
            reason = "hold_coverage_sweep"

    if target_subpolicy != current:
        nav_state["adaptive_switch_count"] = int(nav_state.get("adaptive_switch_count", 0)) + 1
        nav_state["adaptive_mode_steps"] = 0
        counts = nav_state.setdefault("adaptive_switch_reason_counts", {})
        counts[reason] = int(counts.get(reason, 0)) + 1

    nav_state["adaptive_selected_subpolicy"] = target_subpolicy
    nav_state["adaptive_switch_reason"] = reason
    return target_subpolicy


def choose_action_debug(
    env: EnvironmentMap,
    agent: AcousticAgent,
    sensor_obs: Dict[str, np.ndarray],
    predicted_maps: Dict[str, np.ndarray],
    policy_mode: str = "simple",
    nav_state: Optional[Dict[str, Any]] = None,
    coverage: Optional[float] = None,
) -> Tuple[str, Optional[Tuple[int, int]], Dict[str, object]]:
    free_prob = predicted_maps["free_prob"]
    door_prob = predicted_maps["doorway_prob"]
    wall_prob = predicted_maps.get("wall_prob", np.zeros_like(free_prob))
    conf = predicted_maps["confidence_map"]

    if nav_state is None:
        nav_state = init_navigation_state()
    nav_state["step"] = int(nav_state.get("step", 0)) + 1
    nav_state["policy_mode"] = policy_mode
    effective_policy_mode = policy_mode
    if coverage is None:
        coverage = 0.0
    cx0, cy0 = agent.cell(env)
    visited_counts: Dict[Tuple[int, int], int] = nav_state.setdefault("visited_counts", {})
    visited_counts[(cx0, cy0)] = int(visited_counts.get((cx0, cy0), 0)) + 1
    if visited_counts[(cx0, cy0)] >= 4:
        nav_state["high_revisit_cell_count"] = int(nav_state.get("high_revisit_cell_count", 0)) + 1
    coverage_history = nav_state.setdefault("coverage_history", [])
    coverage_history.append(float(coverage))
    stagnating = False
    if len(coverage_history) > 18:
        stagnating = (coverage_history[-1] - coverage_history[-18]) < 0.008
    nav_state["last_stagnation_recovery_active"] = bool(stagnating)
    if stagnating and not bool(nav_state.get("stagnation_flag", False)):
        nav_state["stagnation_event_count"] = int(nav_state.get("stagnation_event_count", 0)) + 1
        nav_state["stagnation_flag"] = True
    elif not stagnating:
        nav_state["stagnation_flag"] = False

    if policy_mode == "accepted_v3_like":
        target, target_score = _select_frontier_target_v3_like(
            env=env,
            agent=agent,
            conf=conf,
            free_prob=free_prob,
            wall_prob=wall_prob,
            door_prob=door_prob,
            nav_state=nav_state,
            stagnating=stagnating,
        )
    elif policy_mode in ("stabilized_frontier", "route_committed", "coverage_sweep", "adaptive_hybrid"):
        target, target_score = _select_frontier_target_stabilized(
            env=env,
            agent=agent,
            conf=conf,
            free_prob=free_prob,
            wall_prob=wall_prob,
            door_prob=door_prob,
            nav_state=nav_state,
            stagnating=stagnating,
        )
    else:
        target = _frontier_target(conf, free_prob)
        target_score = 0.0
    low_conf_frontiers = _low_confidence_frontiers(conf, free_prob)
    h, w = env.shape
    cx, cy = agent.cell(env)

    front_dist = float(sensor_obs["ray_distances_m"][len(sensor_obs["ray_distances_m"]) // 2])
    left_dist = float(np.mean(sensor_obs["ray_distances_m"][:3]))
    right_dist = float(np.mean(sensor_obs["ray_distances_m"][-3:]))

    local_door = float(np.mean(door_prob[max(0, cy - 1) : min(h, cy + 2), max(0, cx - 1) : min(w, cx + 2)]))
    wall_support = float(np.mean(wall_prob[max(0, cy - 1) : min(h, cy + 2), max(0, cx - 1) : min(w, cx + 2)]))
    behind_window = conf[max(0, cy - 6) : min(h, cy + 1), max(0, cx - 3) : min(w, cx + 4)]
    behind_low_conf = float(np.mean(behind_window < 0.50)) if behind_window.size else 0.0
    doorway_candidate = bool(local_door > 0.55 and front_dist > 0.15)
    doorway_rejection_reason = "none"
    doorway_accepted = False
    doorway_threshold = float(nav_state.get("doorway_acceptance_threshold", 0.62))
    doorway_behind_req = float(nav_state.get("doorway_behind_low_confidence_requirement", 0.25))
    safe_slow = _safe_forward(agent, env, 0.08)
    local_structure = _local_structure_diagnostics(
        env=env,
        agent=agent,
        sensor_obs=sensor_obs,
        predicted_maps=predicted_maps,
    )
    for key, value in local_structure.items():
        if key.startswith("local_"):
            nav_state[key] = float(value)
    if policy_mode == "adaptive_hybrid":
        effective_policy_mode = _select_adaptive_subpolicy(nav_state, local_structure, front_dist=front_dist)
        nav_state["steps_in_stabilized_frontier"] = int(nav_state.get("steps_in_stabilized_frontier", 0)) + int(
            effective_policy_mode == "stabilized_frontier"
        )
        nav_state["steps_in_coverage_sweep"] = int(nav_state.get("steps_in_coverage_sweep", 0)) + int(
            effective_policy_mode == "coverage_sweep"
        )
    else:
        nav_state["adaptive_selected_subpolicy"] = policy_mode
        nav_state["adaptive_switch_reason"] = "not_adaptive"
    if doorway_candidate:
        if local_door < doorway_threshold:
            doorway_rejection_reason = "rejected_by_low_probability"
            nav_state["doorway_acceptance_reason"] = "low_probability"
        elif not safe_slow:
            doorway_rejection_reason = "rejected_by_collision_risk"
            nav_state["doorway_acceptance_reason"] = "collision_risk"
        elif behind_low_conf < doorway_behind_req:
            doorway_rejection_reason = "rejected_by_low_behind_confidence_gain"
            nav_state["doorway_acceptance_reason"] = "low_behind_conf_gain"
        elif wall_support <= 0.08:
            doorway_rejection_reason = "rejected_by_structure_gate"
            nav_state["doorway_acceptance_reason"] = "structure_gate"
        else:
            doorway_accepted = True
            doorway_rejection_reason = "accepted"
            nav_state["doorway_for_coverage_count"] = int(nav_state.get("doorway_for_coverage_count", 0)) + 1
            nav_state["doorway_acceptance_reason"] = "accepted_safe_structured"
    elif local_door > 0.40:
        doorway_rejection_reason = "rejected_by_low_probability"

    if doorway_rejection_reason.startswith("rejected_"):
        rej = nav_state.setdefault(
            "doorway_rejection_reasons",
            {
                "rejected_by_low_probability": 0,
                "rejected_by_collision_risk": 0,
                "rejected_by_low_behind_confidence_gain": 0,
                "rejected_by_structure_gate": 0,
            },
        )
        rej[doorway_rejection_reason] = int(rej.get(doorway_rejection_reason, 0)) + 1
    doorway_candidate_cell = (cx, cy) if doorway_candidate else None

    chosen_action = "turn_left"
    reason = "default_turn"
    collision_risk = False
    safe_probe = _safe_forward(agent, env, 0.03)

    if effective_policy_mode == "accepted_v3_like":
        current_target = nav_state.get("current_target")
        target_age = int(nav_state.get("target_age", 0))
        commit_steps = int(nav_state.get("target_commit_steps", 55))
        blacklist: Set[Tuple[int, int]] = nav_state.setdefault("frontier_blacklist", set())
        recent_targets: List[Tuple[int, int]] = nav_state.setdefault("recent_targets", [])

        if current_target is None or target is None:
            nav_state["current_target"] = target
            nav_state["target_age"] = 0
        elif current_target != target and target_age < commit_steps and not stagnating:
            target = current_target
            target_age += 1
            nav_state["target_age"] = target_age
        else:
            if current_target != target:
                recent_targets.append(target)
                nav_state["target_age"] = 0
            nav_state["current_target"] = target

        # Coverage-driven doorway crossing when safe and useful.
        if doorway_accepted and behind_low_conf > 0.30 and safe_slow:
            chosen_action = "move_forward_slow"
            reason = "doorway_for_coverage_safe_crossing"
            nav_state["doorway_for_coverage_count"] = int(nav_state.get("doorway_for_coverage_count", 0)) + 1
        elif front_dist < 0.12:
            chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
            reason = "front_blocked_turn_to_clear_side"
            nav_state["turn_streak"] = int(nav_state.get("turn_streak", 0)) + 1
        else:
            if target is not None:
                tx, ty = target
                wx = (tx + 0.5) * env.cell_size
                wy = (ty + 0.5) * env.cell_size
                ang = math.atan2(wy - agent.y, wx - agent.x)
                err = (ang - agent.theta + math.pi) % (2.0 * math.pi) - math.pi
                # Probe limiting with safe-forward preference.
                if abs(err) > math.radians(16.0):
                    chosen_action = "turn_left" if err > 0 else "turn_right"
                    reason = "align_to_frontier_target"
                    nav_state["turn_streak"] = int(nav_state.get("turn_streak", 0)) + 1
                elif front_dist > 0.22 and safe_slow and (free_prob[cy, cx] > 0.52 or stagnating):
                    chosen_action = "move_forward_slow"
                    reason = "safe_forward_frontier_progress"
                    nav_state["safe_forward_preference_count"] = int(nav_state.get("safe_forward_preference_count", 0)) + 1
                    nav_state["probe_streak"] = 0
                elif front_dist > 0.12 and safe_probe and int(nav_state.get("probe_streak", 0)) < 2:
                    chosen_action = "probe_forward"
                    reason = "probe_unknown_frontier_boundary"
                    nav_state["probe_streak"] = int(nav_state.get("probe_streak", 0)) + 1
                elif safe_slow and front_dist > 0.18:
                    chosen_action = "move_forward_slow"
                    reason = "probe_limited_safe_forward_bias"
                    nav_state["safe_forward_preference_count"] = int(nav_state.get("safe_forward_preference_count", 0)) + 1
                    nav_state["probe_streak"] = 0
                else:
                    chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                    reason = "fallback_turn_due_to_local_block"
            else:
                if front_dist > 0.22 and safe_slow:
                    chosen_action = "move_forward_slow"
                    reason = "safe_forward_no_target"
                    nav_state["safe_forward_preference_count"] = int(nav_state.get("safe_forward_preference_count", 0)) + 1
                elif front_dist > 0.12 and safe_probe and int(nav_state.get("probe_streak", 0)) < 2:
                    chosen_action = "probe_forward"
                    reason = "probe_no_target"
                    nav_state["probe_streak"] = int(nav_state.get("probe_streak", 0)) + 1
                else:
                    chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                    reason = "turn_no_target_blocked"

        # Stagnation recovery: force target switch and blacklist old target.
        if stagnating and target is not None and nav_state.get("current_target") == target:
            blacklist.add(target)
            nav_state["forced_frontier_switch_count"] = int(nav_state.get("forced_frontier_switch_count", 0)) + 1
            nav_state["current_target"] = None
            nav_state["target_age"] = 0
            target, target_score = _select_frontier_target_v3_like(
                env=env,
                agent=agent,
                conf=conf,
                free_prob=free_prob,
                wall_prob=wall_prob,
                door_prob=door_prob,
                nav_state=nav_state,
                stagnating=True,
            )
            if target is not None:
                tx, ty = target
                wx = (tx + 0.5) * env.cell_size
                wy = (ty + 0.5) * env.cell_size
                ang = math.atan2(wy - agent.y, wx - agent.x)
                err = (ang - agent.theta + math.pi) % (2.0 * math.pi) - math.pi
                if abs(err) > math.radians(12.0):
                    chosen_action = "turn_left" if err > 0 else "turn_right"
                    reason = "stagnation_recovery_frontier_switch"
                elif safe_slow and front_dist > 0.16:
                    chosen_action = "move_forward_slow"
                    reason = "stagnation_recovery_safe_advance"
    elif effective_policy_mode == "coverage_sweep":
        sweep_burst_steps = int(nav_state.get("sweep_forward_burst_steps", 8))
        sweep_turn_steps = int(nav_state.get("sweep_turn_angle_steps", 3))
        sweep_scan_interval = int(nav_state.get("sweep_scan_interval", 25))
        low_conf_w = float(nav_state.get("sweep_low_confidence_weight", 1.0))
        unvisited_w = float(nav_state.get("sweep_unvisited_weight", 1.0))
        clearance_cells = int(nav_state.get("sweep_collision_clearance_cells", 2))
        osc_penalty = float(nav_state.get("sweep_oscillation_penalty", 1.0))
        pending_turn_steps = int(nav_state.get("sweep_pending_turn_steps", 0))
        turn_sign = int(nav_state.get("sweep_turn_sign", 1))
        burst_remaining = int(nav_state.get("sweep_burst_remaining", sweep_burst_steps))
        steps_since_scan = int(nav_state.get("sweep_steps_since_scan", 0)) + 1
        nav_state["sweep_steps_since_scan"] = steps_since_scan

        def _sweep_dir_score(sign: int) -> Tuple[float, float, float]:
            # sign: -1 right, 0 front, +1 left
            angle = math.radians(38.0) * float(sign)
            sx, sy = _direction_probe_cell(agent, env, angle_offset_rad=angle, dist_m=1.0)
            conf_local = float(conf[sy, sx])
            free_local = float(free_prob[sy, sx])
            wall_local = float(wall_prob[sy, sx])
            visits = int(visited_counts.get((sx, sy), 0))
            low_conf_score = max(0.0, 1.0 - conf_local)
            unvisited_score = max(0.0, 1.0 - min(1.0, visits / 4.0))
            score = (low_conf_w * low_conf_score) + (unvisited_w * unvisited_score) + 0.4 * free_local - 0.5 * wall_local
            return score, low_conf_score, unvisited_score

        # Respect multi-step sweep turns once selected.
        if pending_turn_steps > 0:
            chosen_action = "turn_left" if turn_sign > 0 else "turn_right"
            reason = "sweep_turn_commit"
            nav_state["sweep_pending_turn_steps"] = pending_turn_steps - 1
            nav_state["sweep_burst_remaining"] = sweep_burst_steps
            nav_state["sweep_heading_rad"] = float(agent.theta)
        else:
            front_score, front_low_conf, front_unvisited = _sweep_dir_score(0)
            left_score, left_low_conf, left_unvisited = _sweep_dir_score(+1)
            right_score, right_low_conf, right_unvisited = _sweep_dir_score(-1)
            nav_state["low_confidence_direction_score_sum"] = float(
                nav_state.get("low_confidence_direction_score_sum", 0.0) + (front_low_conf + left_low_conf + right_low_conf) / 3.0
            )
            nav_state["unvisited_direction_score_sum"] = float(
                nav_state.get("unvisited_direction_score_sum", 0.0) + (front_unvisited + left_unvisited + right_unvisited) / 3.0
            )
            nav_state["direction_score_count"] = int(nav_state.get("direction_score_count", 0)) + 1

            # Forward safety using clearance window.
            forward_clear = safe_slow and front_dist > max(0.12, env.cell_size * clearance_cells * 0.5)
            if forward_clear:
                nav_state["forward_safe_count"] = int(nav_state.get("forward_safe_count", 0)) + 1
            else:
                nav_state["forward_blocked_count"] = int(nav_state.get("forward_blocked_count", 0)) + 1

            should_scan = (steps_since_scan % max(1, sweep_scan_interval) == 0) and forward_clear

            if forward_clear and burst_remaining > 0 and front_score >= max(left_score, right_score) - 0.10:
                chosen_action = "move_forward_slow"
                reason = "sweep_forward_burst"
                nav_state["sweep_forward_burst_count"] = int(nav_state.get("sweep_forward_burst_count", 0)) + 1
                nav_state["sweep_burst_remaining"] = burst_remaining - 1
                nav_state["safe_forward_preference_count"] = int(nav_state.get("safe_forward_preference_count", 0)) + 1
                nav_state["safe_forward_chosen_count"] = int(nav_state.get("safe_forward_chosen_count", 0)) + 1
            else:
                # Choose side direction based on score; discourage immediate left-right oscillation.
                preferred_sign = +1 if left_score >= right_score else -1
                last_turn_sign = int(nav_state.get("last_turn_sign", 0))
                if last_turn_sign != 0 and preferred_sign == -last_turn_sign:
                    # only oscillate if opposite side is much better
                    best = left_score if preferred_sign > 0 else right_score
                    alt = right_score if preferred_sign > 0 else left_score
                    if best < alt + osc_penalty:
                        preferred_sign = last_turn_sign
                        nav_state["oscillation_prevented_count"] = int(nav_state.get("oscillation_prevented_count", 0)) + 1

                if should_scan:
                    nav_state["sweep_scan_turn_count"] = int(nav_state.get("sweep_scan_turn_count", 0)) + 1
                    reason = "sweep_scan_turn"
                else:
                    nav_state["sweep_blocked_turn_count"] = int(nav_state.get("sweep_blocked_turn_count", 0)) + 1
                    reason = "sweep_blocked_or_low_front"

                chosen_action = "turn_left" if preferred_sign > 0 else "turn_right"
                nav_state["sweep_turn_sign"] = preferred_sign
                nav_state["sweep_pending_turn_steps"] = max(0, sweep_turn_steps - 1)
                nav_state["sweep_burst_remaining"] = sweep_burst_steps
                if last_turn_sign != 0 and preferred_sign != last_turn_sign:
                    nav_state["sweep_direction_change_count"] = int(nav_state.get("sweep_direction_change_count", 0)) + 1
                nav_state["last_turn_sign"] = preferred_sign
                nav_state["sweep_last_reason"] = reason

            target = target if target is not None else _forward_target_cell(agent, env, dist_m=0.9)
            target_score = max(front_score, left_score, right_score)
            nav_state["sweep_heading_rad"] = float(agent.theta)

    elif effective_policy_mode == "stabilized_frontier":
        commit_steps = int(nav_state.get("frontier_commit_steps", 20))
        current_target = nav_state.get("current_frontier_target")
        commit_remaining = int(nav_state.get("frontier_commit_remaining", 0))
        switch_reason = "keep_current"

        # Decide whether to keep or switch frontier target.
        need_switch = False
        if current_target is None:
            need_switch = True
            switch_reason = "no_current_target"
        else:
            tx, ty = current_target
            wx = (tx + 0.5) * env.cell_size
            wy = (ty + 0.5) * env.cell_size
            dist_to_target = float(np.hypot(wx - agent.x, wy - agent.y))
            tgt_conf = float(conf[ty, tx]) if 0 <= ty < h and 0 <= tx < w else 1.0
            tgt_visits = int(visited_counts.get((tx, ty), 0))
            blocked_local = bool(front_dist < 0.10 and not safe_slow and not safe_probe)
            if dist_to_target < 0.20:
                need_switch = True
                switch_reason = "target_reached"
                nav_state["target_reached_count"] = int(nav_state.get("target_reached_count", 0)) + 1
            elif blocked_local:
                need_switch = True
                switch_reason = "unsafe_or_blocked"
            elif stagnating:
                need_switch = True
                switch_reason = "coverage_stagnation"
                nav_state["forced_frontier_switch_count"] = int(nav_state.get("forced_frontier_switch_count", 0)) + 1
            elif tgt_conf > 0.78 or tgt_visits >= 3:
                need_switch = True
                switch_reason = "target_high_conf_or_explored"
            elif commit_remaining <= 0:
                need_switch = True
                switch_reason = "commit_expired"

        if need_switch:
            prev = current_target
            current_target = target
            nav_state["current_frontier_target"] = current_target
            nav_state["frontier_commit_remaining"] = commit_steps
            nav_state["frontier_switch_reason"] = switch_reason
            if prev is not None and current_target is not None and switch_reason not in (
                "target_reached",
                "coverage_stagnation",
                "unsafe_or_blocked",
                "target_high_conf_or_explored",
            ):
                nav_state["premature_frontier_switch_count"] = int(nav_state.get("premature_frontier_switch_count", 0)) + 1
            if prev is not None:
                nav_state["last_frontier_target"] = prev
                recent_targets = nav_state.setdefault("recent_targets", [])
                recent_targets.append(prev)
        else:
            target = current_target
            nav_state["frontier_commit_remaining"] = max(0, commit_remaining - 1)
            nav_state["frontier_commitment_count"] = int(nav_state.get("frontier_commitment_count", 0)) + 1
            nav_state["frontier_switch_reason"] = "commit_hold"

        target = nav_state.get("current_frontier_target")
        if target is not None:
            tx, ty = target
            wx = (tx + 0.5) * env.cell_size
            wy = (ty + 0.5) * env.cell_size
            ang = math.atan2(wy - agent.y, wx - agent.x)
            err = (ang - agent.theta + math.pi) % (2.0 * math.pi) - math.pi
            # Safe-forward preference: advance if next cell is less confident / less visited.
            nxf = agent.x + 0.08 * math.cos(agent.theta)
            nyf = agent.y + 0.08 * math.sin(agent.theta)
            fx, fy = world_to_cell(nxf, nyf, env.cell_size, w, h)
            ahead_low_conf = bool(conf[fy, fx] < 0.68)
            ahead_low_visit = int(visited_counts.get((fx, fy), 0)) <= 1
            if doorway_accepted and behind_low_conf >= doorway_behind_req and safe_slow:
                chosen_action = "move_forward_slow"
                reason = "doorway_for_coverage_safe_crossing"
            elif abs(err) <= math.radians(16.0) and safe_slow and (ahead_low_conf or ahead_low_visit):
                chosen_action = "move_forward_slow"
                reason = "safe_forward_to_low_conf_or_low_visit"
                nav_state["safe_forward_preference_count"] = int(nav_state.get("safe_forward_preference_count", 0)) + 1
                nav_state["safe_forward_chosen_count"] = int(nav_state.get("safe_forward_chosen_count", 0)) + 1
            elif abs(err) <= math.radians(16.0) and not safe_slow:
                nav_state["safe_forward_blocked_count"] = int(nav_state.get("safe_forward_blocked_count", 0)) + 1
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "safe_forward_blocked_turn"
            elif abs(err) > math.radians(16.0):
                chosen_action = "turn_left" if err > 0 else "turn_right"
                reason = "align_to_committed_frontier"
            elif safe_probe and int(nav_state.get("probe_streak", 0)) < 1:
                chosen_action = "probe_forward"
                reason = "probe_frontier_boundary"
                nav_state["probe_streak"] = int(nav_state.get("probe_streak", 0)) + 1
            elif safe_slow:
                chosen_action = "move_forward_slow"
                reason = "safe_forward_fallback"
                nav_state["safe_forward_chosen_count"] = int(nav_state.get("safe_forward_chosen_count", 0)) + 1
            else:
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "fallback_turn_committed_frontier"
        else:
            if safe_slow and front_dist > 0.18:
                chosen_action = "move_forward_slow"
                reason = "safe_forward_no_frontier"
                nav_state["safe_forward_chosen_count"] = int(nav_state.get("safe_forward_chosen_count", 0)) + 1
            else:
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "turn_no_frontier"
    elif effective_policy_mode == "route_committed":
        route_commit_steps = int(nav_state.get("route_commit_steps", 50))
        target_reached_radius = float(nav_state.get("target_reached_radius_m", 0.35))
        max_blocked_steps = int(nav_state.get("max_blocked_steps", 8))
        heading_tolerance = math.radians(float(nav_state.get("heading_tolerance_deg", 15.0)))
        route_target = nav_state.get("current_route_target")
        route_commit_remaining = int(nav_state.get("route_commit_remaining", 0))
        switch_reason = "keep_route"
        should_replan = False

        # Doorway-as-target logic for coverage expansion in doorway-like cases.
        doorway_target: Optional[Tuple[int, int]] = None
        if doorway_accepted and behind_low_conf >= doorway_behind_req:
            doorway_target = _forward_target_cell(agent, env, dist_m=0.8)

        if route_target is None:
            should_replan = True
            switch_reason = "no_route_target"
        else:
            tx, ty = route_target
            wx = (tx + 0.5) * env.cell_size
            wy = (ty + 0.5) * env.cell_size
            dist_to_target = float(np.hypot(wx - agent.x, wy - agent.y))
            nav_state["distance_to_target"] = dist_to_target
            prev_dist = nav_state.get("last_distance_to_target")
            if isinstance(prev_dist, (int, float)):
                delta = float(prev_dist - dist_to_target)
                nav_state["distance_to_target_delta"] = delta
                if delta > 0.01:
                    nav_state["progress_toward_target_count"] = int(nav_state.get("progress_toward_target_count", 0)) + 1
                    nav_state["no_progress_count"] = 0
                else:
                    nav_state["no_progress_count"] = int(nav_state.get("no_progress_count", 0)) + 1
            nav_state["last_distance_to_target"] = dist_to_target

            tgt_conf = float(conf[ty, tx]) if 0 <= ty < h and 0 <= tx < w else 1.0
            tgt_visits = int(visited_counts.get((tx, ty), 0))
            local_blocked = bool(front_dist < 0.10 and not safe_slow and not safe_probe)
            if dist_to_target <= target_reached_radius:
                should_replan = True
                switch_reason = "target_reached"
                nav_state["target_reached_count"] = int(nav_state.get("target_reached_count", 0)) + 1
                if doorway_target is not None:
                    nav_state["doorway_target_reached_count"] = int(nav_state.get("doorway_target_reached_count", 0)) + 1
            elif local_blocked:
                nav_state["blocked_step_count"] = int(nav_state.get("blocked_step_count", 0)) + 1
                if int(nav_state.get("blocked_step_count", 0)) >= max_blocked_steps:
                    should_replan = True
                    switch_reason = "repeatedly_blocked"
            elif stagnating:
                should_replan = True
                switch_reason = "coverage_stagnation"
            elif tgt_conf > 0.82 or tgt_visits >= 4:
                should_replan = True
                switch_reason = "target_explored"
            elif route_commit_remaining <= 0:
                should_replan = True
                switch_reason = "commit_expired"
            elif int(nav_state.get("no_progress_count", 0)) >= 12:
                should_replan = True
                switch_reason = "no_progress"

        if should_replan:
            prev_route = route_target
            if doorway_target is not None:
                route_target = doorway_target
                nav_state["doorway_target_selected_count"] = int(nav_state.get("doorway_target_selected_count", 0)) + 1
                switch_reason = "doorway_target_selected"
            else:
                route_target = target
            nav_state["current_route_target"] = route_target
            nav_state["route_commit_remaining"] = route_commit_steps
            nav_state["route_replan_count"] = int(nav_state.get("route_replan_count", 0)) + 1
            nav_state["route_target_switch_reason"] = switch_reason
            nav_state["blocked_step_count"] = 0
            nav_state["no_progress_count"] = 0
            nav_state["last_distance_to_target"] = None
            if prev_route is not None and route_target is not None and prev_route != route_target:
                nav_state["last_frontier_target"] = prev_route
        else:
            nav_state["route_commit_remaining"] = max(0, route_commit_remaining - 1)
            nav_state["route_commitment_count"] = int(nav_state.get("route_commitment_count", 0)) + 1
            nav_state["route_target_switch_reason"] = "commit_hold"

        target = nav_state.get("current_route_target")
        if target is not None:
            tx, ty = target
            wx = (tx + 0.5) * env.cell_size
            wy = (ty + 0.5) * env.cell_size
            dist_to_target = float(np.hypot(wx - agent.x, wy - agent.y))
            nav_state["distance_to_target"] = dist_to_target
            ang = math.atan2(wy - agent.y, wx - agent.x)
            err = (ang - agent.theta + math.pi) % (2.0 * math.pi) - math.pi
            heading_error_deg = abs(math.degrees(err))
            nav_state["mean_heading_error_accum"] = float(nav_state.get("mean_heading_error_accum", 0.0)) + heading_error_deg
            nav_state["heading_error_count"] = int(nav_state.get("heading_error_count", 0)) + 1

            nxf = agent.x + 0.08 * math.cos(agent.theta)
            nyf = agent.y + 0.08 * math.sin(agent.theta)
            fx, fy = world_to_cell(nxf, nyf, env.cell_size, w, h)
            ahead_low_conf = bool(conf[fy, fx] < 0.70)
            ahead_low_visit = int(visited_counts.get((fx, fy), 0)) <= 1

            if abs(err) > heading_tolerance:
                chosen_action = "turn_left" if err > 0 else "turn_right"
                reason = "route_align_heading"
            elif safe_slow and front_dist > 0.12 and (ahead_low_conf or ahead_low_visit or dist_to_target > target_reached_radius):
                chosen_action = "move_forward_slow"
                reason = "route_safe_forward"
                nav_state["safe_forward_preference_count"] = int(nav_state.get("safe_forward_preference_count", 0)) + 1
                nav_state["safe_forward_chosen_count"] = int(nav_state.get("safe_forward_chosen_count", 0)) + 1
            elif not safe_slow:
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "route_forward_blocked_turn"
                nav_state["safe_forward_blocked_count"] = int(nav_state.get("safe_forward_blocked_count", 0)) + 1
                nav_state["blocked_step_count"] = int(nav_state.get("blocked_step_count", 0)) + 1
            elif safe_probe:
                chosen_action = "probe_forward"
                reason = "route_probe_when_tight"
            else:
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "route_turn_fallback"
        else:
            if safe_slow and front_dist > 0.16:
                chosen_action = "move_forward_slow"
                reason = "route_no_target_safe_forward"
                nav_state["safe_forward_chosen_count"] = int(nav_state.get("safe_forward_chosen_count", 0)) + 1
            else:
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "route_no_target_turn"
    elif doorway_accepted:
        chosen_action = "move_forward_slow"
        reason = "doorway_candidate_safe_forward"
    elif front_dist < 0.12:
        chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
        reason = "front_blocked_turn_to_clear_side"
    else:
        if target is not None:
            tx, ty = target
            wx = (tx + 0.5) * env.cell_size
            wy = (ty + 0.5) * env.cell_size
            ang = math.atan2(wy - agent.y, wx - agent.x)
            err = (ang - agent.theta + math.pi) % (2.0 * math.pi) - math.pi
            if abs(err) > math.radians(18.0):
                chosen_action = "turn_left" if err > 0 else "turn_right"
                reason = "align_to_frontier_target"
            elif front_dist > 0.22 and safe_slow:
                chosen_action = "move_forward_slow"
                reason = "safe_forward_frontier_progress"
            elif front_dist > 0.12 and safe_probe:
                chosen_action = "probe_forward"
                reason = "probe_unknown_frontier_boundary"
            else:
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "fallback_turn_due_to_local_block"
        else:
            if front_dist > 0.22 and safe_slow:
                chosen_action = "move_forward_slow"
                reason = "safe_forward_no_target"
            elif front_dist > 0.12 and safe_probe:
                chosen_action = "probe_forward"
                reason = "probe_no_target"
            else:
                chosen_action = "turn_left" if left_dist > right_dist else "turn_right"
                reason = "turn_no_target_blocked"

    if chosen_action == "move_forward_slow":
        collision_risk = not safe_slow
    elif chosen_action == "probe_forward":
        collision_risk = not safe_probe

    if policy_mode == "adaptive_hybrid":
        recent_actions = nav_state.setdefault("adaptive_recent_actions", [])
        recent_actions.append(chosen_action)
        if len(recent_actions) > 120:
            del recent_actions[: len(recent_actions) - 120]
        blocked_action = "blocked" in reason or (front_dist < 0.12 and chosen_action in ("turn_left", "turn_right"))
        if blocked_action:
            nav_state["adaptive_forward_blocked_streak"] = int(nav_state.get("adaptive_forward_blocked_streak", 0)) + 1
        elif chosen_action in ("move_forward_slow", "probe_forward"):
            nav_state["adaptive_forward_blocked_streak"] = 0

    # Confidence gain estimate: how much low-confidence free area exists in front arc.
    front_window = conf[max(0, cy - 4) : min(h, cy + 1), max(0, cx - 2) : min(w, cx + 3)]
    gain_est = float(np.mean(front_window < 0.5)) if front_window.size else 0.0

    debug = {
        "policy_mode": policy_mode,
        "adaptive_selected_subpolicy": str(nav_state.get("adaptive_selected_subpolicy", effective_policy_mode)),
        "adaptive_switch_reason": str(nav_state.get("adaptive_switch_reason", "")),
        "adaptive_switch_count": int(nav_state.get("adaptive_switch_count", 0)),
        "steps_in_stabilized_frontier": int(nav_state.get("steps_in_stabilized_frontier", 0)),
        "steps_in_coverage_sweep": int(nav_state.get("steps_in_coverage_sweep", 0)),
        "adaptive_min_mode_steps": int(nav_state.get("adaptive_min_mode_steps", 40)),
        "adaptive_coverage_gain_window": int(nav_state.get("adaptive_coverage_gain_window", 50)),
        "adaptive_switch_reason_counts": dict(nav_state.get("adaptive_switch_reason_counts", {})),
        "local_obstacle_density": float(nav_state.get("local_obstacle_density", 0.0)),
        "local_free_space_ratio": float(nav_state.get("local_free_space_ratio", 0.0)),
        "local_wall_structure_score": float(nav_state.get("local_wall_structure_score", 0.0)),
        "local_doorway_score": float(nav_state.get("local_doorway_score", 0.0)),
        "local_clutter_score": float(nav_state.get("local_clutter_score", 0.0)),
        "local_corridor_score": float(nav_state.get("local_corridor_score", 0.0)),
        "chosen_action": chosen_action,
        "action_reason": reason,
        "target_frontier": target,
        "target_frontier_score": float(target_score),
        "current_committed_frontier_target": nav_state.get(
            "current_frontier_target", nav_state.get("current_route_target")
        ),
        "frontier_commitment_remaining_steps": int(
            nav_state.get("frontier_commit_remaining", nav_state.get("route_commit_remaining", 0))
        ),
        "frontier_switch_reason": str(nav_state.get("frontier_switch_reason", "")),
        "low_confidence_frontier_cells": low_conf_frontiers,
        "doorway_candidate": doorway_candidate,
        "doorway_candidate_cell": doorway_candidate_cell,
        "doorway_accepted": doorway_accepted,
        "doorway_acceptance_reason": str(nav_state.get("doorway_acceptance_reason", "none")),
        "doorway_rejection_reason": doorway_rejection_reason,
        "doorway_candidate_count": 1 if doorway_candidate else 0,
        "doorway_accepted_count": 1 if doorway_accepted else 0,
        "doorway_rejected_count": 1 if doorway_candidate and not doorway_accepted else 0,
        "collision_risk": bool(collision_risk),
        "confidence_gain_estimate": gain_est,
        "stagnation_recovery_active": bool(stagnating),
        "stagnation_event_count": int(nav_state.get("stagnation_event_count", 0)),
        "forced_frontier_switch_count": int(nav_state.get("forced_frontier_switch_count", 0)),
        "frontier_blacklist_count": int(len(nav_state.get("frontier_blacklist", set()))),
        "safe_forward_preference_count": int(nav_state.get("safe_forward_preference_count", 0)),
        "safe_forward_chosen_count": int(nav_state.get("safe_forward_chosen_count", 0)),
        "safe_forward_blocked_count": int(nav_state.get("safe_forward_blocked_count", 0)),
        "doorway_for_coverage_count": int(nav_state.get("doorway_for_coverage_count", 0)),
        "frontier_commitment_count": int(nav_state.get("frontier_commitment_count", 0)),
        "premature_frontier_switch_count": int(nav_state.get("premature_frontier_switch_count", 0)),
        "target_reached_count": int(nav_state.get("target_reached_count", 0)),
        "revisit_penalty_weight": float(nav_state.get("revisit_penalty_weight", 0.0)),
        "high_revisit_cell_count": int(nav_state.get("high_revisit_cell_count", 0)),
        "doorway_rejection_reason_counts": dict(nav_state.get("doorway_rejection_reasons", {})),
        "route_commitment_count": int(nav_state.get("route_commitment_count", 0)),
        "route_replan_count": int(nav_state.get("route_replan_count", 0)),
        "distance_to_target": nav_state.get("distance_to_target"),
        "distance_to_target_delta": float(nav_state.get("distance_to_target_delta", 0.0)),
        "progress_toward_target_count": int(nav_state.get("progress_toward_target_count", 0)),
        "no_progress_count": int(nav_state.get("no_progress_count", 0)),
        "blocked_step_count": int(nav_state.get("blocked_step_count", 0)),
        "route_commit_remaining": int(nav_state.get("route_commit_remaining", 0)),
        "route_target_switch_reason": str(nav_state.get("route_target_switch_reason", "")),
        "doorway_target_selected_count": int(nav_state.get("doorway_target_selected_count", 0)),
        "doorway_target_reached_count": int(nav_state.get("doorway_target_reached_count", 0)),
        "sweep_heading_rad": float(nav_state.get("sweep_heading_rad", agent.theta)),
        "sweep_burst_remaining": int(nav_state.get("sweep_burst_remaining", 0)),
        "sweep_last_reason": str(nav_state.get("sweep_last_reason", "")),
        "sweep_direction_score": float(target_score),
        "sweep_forward_burst_count": int(nav_state.get("sweep_forward_burst_count", 0)),
        "sweep_blocked_turn_count": int(nav_state.get("sweep_blocked_turn_count", 0)),
        "sweep_scan_turn_count": int(nav_state.get("sweep_scan_turn_count", 0)),
        "sweep_direction_change_count": int(nav_state.get("sweep_direction_change_count", 0)),
        "forward_safe_count": int(nav_state.get("forward_safe_count", 0)),
        "forward_blocked_count": int(nav_state.get("forward_blocked_count", 0)),
        "low_confidence_direction_score": float(nav_state.get("low_confidence_direction_score_sum", 0.0)),
        "unvisited_direction_score": float(nav_state.get("unvisited_direction_score_sum", 0.0)),
        "oscillation_prevented_count": int(nav_state.get("oscillation_prevented_count", 0)),
        "nav_state": nav_state,
    }
    return chosen_action, target, debug


def choose_action_simple(
    env: EnvironmentMap,
    agent: AcousticAgent,
    sensor_obs: Dict[str, np.ndarray],
    predicted_maps: Dict[str, np.ndarray],
) -> Tuple[str, Optional[Tuple[int, int]]]:
    action, target, _ = choose_action_debug(env, agent, sensor_obs, predicted_maps, policy_mode="simple")
    return action, target
