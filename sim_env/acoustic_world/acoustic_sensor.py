from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import (
    DEFAULT_RAY_ANGLES_DEG,
    DIFFICULTY_PRESETS,
    ECHO_VECTOR_BINS,
    MAX_SENSOR_RANGE_M,
    SPEED_OF_SOUND_M_S,
)
from .environments import EnvironmentMap, world_to_cell


@dataclass
class AcousticSensor:
    ray_angles_deg: List[float]
    max_range_m: float
    echo_bins: int

    @classmethod
    def default(cls) -> "AcousticSensor":
        return cls(
            ray_angles_deg=list(DEFAULT_RAY_ANGLES_DEG),
            max_range_m=MAX_SENSOR_RANGE_M,
            echo_bins=ECHO_VECTOR_BINS,
        )

    def _cast_ray(self, env: EnvironmentMap, x: float, y: float, theta: float) -> float:
        step_m = max(0.03, 0.7 * env.cell_size)
        h, w = env.shape
        dist = 0.0
        while dist <= self.max_range_m:
            px = x + dist * math.cos(theta)
            py = y + dist * math.sin(theta)
            if px < 0.0 or py < 0.0 or px >= env.width_m or py >= env.height_m:
                return dist
            cx, cy = world_to_cell(px, py, env.cell_size, w, h)
            if env.ground_truth_occupancy[cy, cx]:
                return dist
            dist += step_m
        return self.max_range_m

    def sense(self, env: EnvironmentMap, x: float, y: float, heading: float, difficulty: str, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        preset = DIFFICULTY_PRESETS[difficulty]
        distances = []
        intensities = []
        ray_angles_rad = []

        for deg in self.ray_angles_deg:
            ang = heading + math.radians(deg)
            d_true = self._cast_ray(env, x, y, ang)
            if rng.random() < float(preset["dropout_prob"]):
                d_meas = self.max_range_m
            else:
                d_meas = d_true + float(rng.normal(0.0, float(preset["distance_noise_std"])))
                d_meas = float(np.clip(d_meas, 0.02, self.max_range_m))
            intensity = 1.0 / (1.0 + 1.8 * d_meas * d_meas)
            intensity += float(rng.normal(0.0, float(preset["intensity_noise_std"])))
            intensity = float(np.clip(intensity, 0.0, 1.0))
            distances.append(d_meas)
            intensities.append(intensity)
            ray_angles_rad.append(ang)

        distances_arr = np.asarray(distances, dtype=np.float32)
        intensities_arr = np.asarray(intensities, dtype=np.float32)
        ray_angles_arr = np.asarray(ray_angles_rad, dtype=np.float32)
        timing_vector = self._build_echo_timing_vector(distances_arr, intensities_arr)
        intensity_vector = self._build_echo_intensity_vector(distances_arr, intensities_arr)
        return {
            "ray_angles_rad": ray_angles_arr,
            "ray_distances_m": distances_arr,
            "ray_intensities": intensities_arr,
            "echo_timing_vector": timing_vector,
            "echo_intensity_vector": intensity_vector,
            "multichannel_features": np.stack([timing_vector, intensity_vector], axis=0),
        }

    def _build_echo_timing_vector(self, distances_m: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        vec = np.zeros(self.echo_bins, dtype=np.float32)
        max_delay_s = 2.0 * self.max_range_m / SPEED_OF_SOUND_M_S
        for d, amp in zip(distances_m, intensities):
            delay_s = 2.0 * float(d) / SPEED_OF_SOUND_M_S
            idx = int(np.clip(round((delay_s / max_delay_s) * (self.echo_bins - 1)), 0, self.echo_bins - 1))
            for k in range(-2, 3):
                j = idx + k
                if 0 <= j < self.echo_bins:
                    vec[j] += float(amp * math.exp(-0.5 * (k / 1.25) ** 2))
        return np.clip(vec, 0.0, 1.0)

    def _build_echo_intensity_vector(self, distances_m: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        vec = np.zeros(self.echo_bins, dtype=np.float32)
        for i, (d, amp) in enumerate(zip(distances_m, intensities)):
            idx = int(np.clip(round((float(d) / self.max_range_m) * (self.echo_bins - 1)), 0, self.echo_bins - 1))
            vec[idx] = max(vec[idx], float(amp))
            if idx + 1 < self.echo_bins:
                vec[idx + 1] = max(vec[idx + 1], float(0.5 * amp))
            if idx - 1 >= 0:
                vec[idx - 1] = max(vec[idx - 1], float(0.5 * amp))
        return np.clip(vec, 0.0, 1.0)
