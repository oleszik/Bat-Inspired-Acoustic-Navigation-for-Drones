from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .environments import EnvironmentMap


DEFAULT_MAPPER_MANIFEST = Path("runs/accepted_models/phase2c5_hybrid_acoustic_mapper/manifest.json")


def load_mapper_manifest(path: str | Path = DEFAULT_MAPPER_MANIFEST) -> Dict[str, object]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def get_mapper_config(manifest: Dict[str, object]) -> Dict[str, object]:
    cfg = {
        "accepted_model_name": manifest.get("accepted_model_name", "unknown"),
        "status": manifest.get("navigation_status", manifest.get("status", "unknown")),
        "selected_modes": manifest.get("selected_modes", {}),
        "fallback": manifest.get("fallback", {}),
    }
    return cfg


def dummy_predict_maps(
    env: EnvironmentMap,
    observed_mask: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Placeholder for neural mapper inference.
    Uses GT + controlled noise to generate plausible map predictions.
    """
    occ_gt = env.ground_truth_occupancy.astype(np.float32)
    wall_gt = env.wall_mask.astype(np.float32)
    door_gt = env.doorway_mask.astype(np.float32)
    free_gt = env.free_space_mask.astype(np.float32)

    observed_f = observed_mask.astype(np.float32)
    unknown_f = 1.0 - observed_f

    occ_noise = rng.normal(0.0, 0.07, size=occ_gt.shape).astype(np.float32)
    wall_noise = rng.normal(0.0, 0.06, size=wall_gt.shape).astype(np.float32)
    door_noise = rng.normal(0.0, 0.05, size=door_gt.shape).astype(np.float32)

    occ_prob = np.clip(0.55 * occ_gt + 0.45 * 0.5 + observed_f * (0.35 * (occ_gt - 0.5)) + occ_noise, 0.0, 1.0)
    wall_prob = np.clip(0.55 * wall_gt + 0.20 * occ_gt + wall_noise, 0.0, 1.0) * observed_f
    doorway_prob = np.clip(0.65 * door_gt + 0.10 * free_gt + door_noise, 0.0, 1.0) * observed_f
    free_prob = np.clip(0.60 * free_gt + 0.40 * 0.5 + observed_f * (0.35 * (free_gt - 0.5)) - occ_noise, 0.0, 1.0)

    # Confidence is high where repeatedly observed and where occupancy/free agree.
    consistency = 1.0 - np.abs((occ_prob + free_prob) - 1.0)
    confidence = np.clip(0.70 * observed_f + 0.30 * consistency - 0.20 * unknown_f, 0.0, 1.0)

    return {
        "occupancy_prob": occ_prob,
        "wall_prob": wall_prob,
        "doorway_prob": doorway_prob,
        "free_prob": free_prob,
        "confidence_map": confidence,
    }
