from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .environments import EnvironmentMap, world_to_cell


def wrap_angle(theta: float) -> float:
    return float((theta + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class AcousticAgent:
    x: float
    y: float
    theta: float
    radius_m: float = 0.12
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trajectory.append((self.x, self.y))

    def _is_collision(self, env: EnvironmentMap, nx: float, ny: float) -> bool:
        h, w = env.shape
        if nx < self.radius_m or ny < self.radius_m or nx > env.width_m - self.radius_m or ny > env.height_m - self.radius_m:
            return True
        cx, cy = world_to_cell(nx, ny, env.cell_size, w, h)
        return bool(env.ground_truth_occupancy[cy, cx])

    def _advance(self, env: EnvironmentMap, distance_m: float) -> bool:
        nx = self.x + distance_m * math.cos(self.theta)
        ny = self.y + distance_m * math.sin(self.theta)
        if self._is_collision(env, nx, ny):
            return False
        self.x = nx
        self.y = ny
        self.trajectory.append((self.x, self.y))
        return True

    def move_forward_slow(self, env: EnvironmentMap, step_m: float = 0.08) -> bool:
        return self._advance(env, step_m)

    def probe_forward(self, env: EnvironmentMap, step_m: float = 0.03) -> bool:
        return self._advance(env, step_m)

    def stop_or_reverse(self, env: EnvironmentMap, step_m: float = 0.03) -> bool:
        return self._advance(env, -step_m)

    def turn_left(self, turn_deg: float = 15.0) -> bool:
        self.theta = wrap_angle(self.theta + math.radians(turn_deg))
        return True

    def turn_right(self, turn_deg: float = 15.0) -> bool:
        self.theta = wrap_angle(self.theta - math.radians(turn_deg))
        return True

    def apply_action(self, env: EnvironmentMap, action: str) -> bool:
        action = action.strip().lower()
        if action == "move_forward_slow":
            return self.move_forward_slow(env)
        if action == "probe_forward":
            return self.probe_forward(env)
        if action == "stop_or_reverse":
            return self.stop_or_reverse(env)
        if action == "turn_left":
            return self.turn_left()
        if action == "turn_right":
            return self.turn_right()
        return False

    def cell(self, env: EnvironmentMap) -> Tuple[int, int]:
        h, w = env.shape
        return world_to_cell(self.x, self.y, env.cell_size, w, h)
