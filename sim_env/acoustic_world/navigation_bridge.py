from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def choose_action_simple(
    env: EnvironmentMap,
    agent: AcousticAgent,
    sensor_obs: Dict[str, np.ndarray],
    predicted_maps: Dict[str, np.ndarray],
) -> Tuple[str, Optional[Tuple[int, int]]]:
    free_prob = predicted_maps["free_prob"]
    door_prob = predicted_maps["doorway_prob"]
    conf = predicted_maps["confidence_map"]

    target = _frontier_target(conf, free_prob)
    h, w = env.shape
    cx, cy = agent.cell(env)

    front_dist = float(sensor_obs["ray_distances_m"][len(sensor_obs["ray_distances_m"]) // 2])
    left_dist = float(np.mean(sensor_obs["ray_distances_m"][:3]))
    right_dist = float(np.mean(sensor_obs["ray_distances_m"][-3:]))

    # Doorway preference if safe and likely.
    local_door = float(np.mean(door_prob[max(0, cy - 1) : min(h, cy + 2), max(0, cx - 1) : min(w, cx + 2)]))
    if local_door > 0.55 and front_dist > 0.20 and _safe_forward(agent, env, 0.08):
        return "move_forward_slow", target

    if front_dist < 0.12:
        if left_dist > right_dist:
            return "turn_left", target
        return "turn_right", target

    # If heading is not aligned with frontier, rotate.
    if target is not None:
        tx, ty = target
        wx = (tx + 0.5) * env.cell_size
        wy = (ty + 0.5) * env.cell_size
        ang = math.atan2(wy - agent.y, wx - agent.x)
        err = (ang - agent.theta + math.pi) % (2.0 * math.pi) - math.pi
        if abs(err) > math.radians(18.0):
            return ("turn_left" if err > 0 else "turn_right"), target

    # Safe forward exploration bias.
    if front_dist > 0.22 and _safe_forward(agent, env, 0.08):
        return "move_forward_slow", target
    if front_dist > 0.12 and _safe_forward(agent, env, 0.03):
        return "probe_forward", target
    if left_dist > right_dist:
        return "turn_left", target
    return "turn_right", target
