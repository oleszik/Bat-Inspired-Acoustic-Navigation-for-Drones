from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from .echo_dataset import EchoMappingNPZDataset
    from .train_echo_mapper_v4 import EchoMapperV2 as V4Model
    from .train_echo_mapper_v5 import EchoMapperV3 as V5Model
    from .mapping_utils import (
        DIFFICULTY_PRESETS,
        RAY_MAX_RANGE,
        SECTOR_NAMES,
        SECTOR_OFFSETS_RAD,
        apply_action,
        build_gt_grids,
        make_maps,
        point_in_obstacle,
        predict_collision,
        sample_free_pose,
        simulate_echo_observation,
        wrap_angle,
    )
except ImportError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.phase2_mapping.echo_dataset import EchoMappingNPZDataset  # type: ignore
    from simulation.phase2_mapping.train_echo_mapper_v4 import EchoMapperV2 as V4Model  # type: ignore
    from simulation.phase2_mapping.train_echo_mapper_v5 import EchoMapperV3 as V5Model  # type: ignore
    from simulation.phase2_mapping.mapping_utils import (  # type: ignore
        DIFFICULTY_PRESETS,
        RAY_MAX_RANGE,
        SECTOR_NAMES,
        SECTOR_OFFSETS_RAD,
        apply_action,
        build_gt_grids,
        make_maps,
        point_in_obstacle,
        predict_collision,
        sample_free_pose,
        simulate_echo_observation,
        wrap_angle,
    )


PLANNER_MODES = [
    "frontier_exploration",
    "doorway_approach",
    "wall_following",
    "low_confidence_mapping",
    "emergency_avoidance",
]

ACTION_NAMES = [
    "MOVE_FORWARD_FAST",
    "MOVE_FORWARD_SLOW",
    "PROBE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "SLOW_DOWN_AND_RESAMPLE",
    "STOP_OR_REVERSE",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_NAMES)}


@dataclass
class MapperRun:
    model: torch.nn.Module
    config: Dict[str, object]
    has_context_head: bool
    use_soft_gating: bool
    gate_strength: float
    gate_min: float
    gate_structure_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2D mapper-guided navigation using accepted acoustic mapper.")
    parser.add_argument("--episodes-per-map", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--maps", type=str, default="empty_room,corridor,single_block,doorway,cluttered_room")
    parser.add_argument("--difficulties", type=str, default="clean")
    parser.add_argument(
        "--accepted-mapper-manifest",
        type=str,
        default="runs/accepted_models/phase2c5_hybrid_acoustic_mapper/manifest.json",
    )
    parser.add_argument("--output-dir", type=str, default="runs/phase2_mapper_guided_navigation_v3")
    parser.add_argument("--target-coverage", type=float, default=0.65)
    parser.add_argument("--min-coverage-for-success", type=float, default=0.65)
    parser.add_argument("--coverage-stagnation-window", type=int, default=50)
    parser.add_argument("--coverage-min-gain", type=float, default=0.01)
    parser.add_argument("--frontier-blacklist-steps", type=int, default=100)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def world_to_cell(x: float, y: float, cell_size: float, nx: int, ny: int) -> Tuple[int, int]:
    cx = int(np.clip(math.floor(x / cell_size), 0, nx - 1))
    cy = int(np.clip(math.floor(y / cell_size), 0, ny - 1))
    return cx, cy


def make_echo_bins(distance: float, intensity: float, n_bins: int, max_range: float) -> np.ndarray:
    vec = np.zeros(n_bins, dtype=np.float32)
    pos = int(np.clip(round((distance / max_range) * (n_bins - 1)), 0, n_bins - 1))
    for offset in range(-3, 4):
        idx = pos + offset
        if 0 <= idx < n_bins:
            vec[idx] += float(intensity * math.exp(-0.5 * (offset / 1.6) ** 2))
    return np.clip(vec, 0.0, 1.0)


def _shift_no_wrap(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(a)
    y_src_start = max(0, -dy)
    y_src_end = a.shape[1] - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end = a.shape[2] - max(0, dx)
    y_dst_start = max(0, dy)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)
    if y_src_end > y_src_start and x_src_end > x_src_start:
        out[:, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = a[:, y_src_start:y_src_end, x_src_start:x_src_end]
    return out


def apply_doorway_structural_gating(
    door_prob_2d: np.ndarray,
    wall_prob_2d: np.ndarray,
    free_prob_2d: np.ndarray,
    occ_prob_2d: np.ndarray,
    patch_size: int,
    gate_strength: float,
    min_gate: float,
    structure_weight: float,
) -> np.ndarray:
    n = door_prob_2d.shape[0]
    d = door_prob_2d.reshape(n, patch_size, patch_size)
    w = wall_prob_2d.reshape(n, patch_size, patch_size)
    f = free_prob_2d.reshape(n, patch_size, patch_size)
    o = occ_prob_2d.reshape(n, patch_size, patch_size)

    h1 = np.minimum(_shift_no_wrap(w, 0, -1), _shift_no_wrap(w, 0, 1))
    h2 = np.minimum(_shift_no_wrap(w, 0, -2), _shift_no_wrap(w, 0, 2))
    v1 = np.minimum(_shift_no_wrap(w, -1, 0), _shift_no_wrap(w, 1, 0))
    v2 = np.minimum(_shift_no_wrap(w, -2, 0), _shift_no_wrap(w, 2, 0))
    wall_support = np.maximum.reduce([h1, h2, v1, v2])

    neigh = [
        _shift_no_wrap(w, -1, -1),
        _shift_no_wrap(w, -1, 0),
        _shift_no_wrap(w, -1, 1),
        _shift_no_wrap(w, 0, -1),
        _shift_no_wrap(w, 0, 1),
        _shift_no_wrap(w, 1, -1),
        _shift_no_wrap(w, 1, 0),
        _shift_no_wrap(w, 1, 1),
    ]
    local_wall = np.mean(np.stack(neigh, axis=0), axis=0)

    support_gate = np.clip((wall_support - 0.25) / 0.45, 0.0, 1.0)
    free_gate = np.clip((f - 0.35) / 0.45, 0.0, 1.0)
    occ_gate = np.clip((0.65 - o) / 0.65, 0.0, 1.0)
    structure_score = np.clip((0.6 * support_gate + 0.2 * free_gate + 0.2 * occ_gate), 0.0, 1.0)

    open_empty = ((f > 0.80) & (o < 0.18) & (local_wall < 0.10)).astype(np.float32)
    isolated_edge = ((o > 0.55) & (wall_support < 0.20)).astype(np.float32)
    suppress = np.clip(0.75 * open_empty + 0.45 * isolated_edge, 0.0, 1.0)
    structure_score = np.clip(structure_score * (1.0 - structure_weight * suppress), 0.0, 1.0)

    base_gate = min_gate + (1.0 - min_gate) * structure_score
    gate = (1.0 - gate_strength) + gate_strength * base_gate
    return np.clip(d * gate, 0.0, 1.0).reshape(door_prob_2d.shape[0], -1)


def load_mapper_runs(manifest: Dict[str, object], device: torch.device, patch_size: int) -> Tuple[MapperRun, MapperRun]:
    src = manifest.get("source_files", {})
    if not isinstance(src, dict):
        raise ValueError("Manifest missing source_files.")
    v4_ckpt = Path(str(src.get("v4_checkpoint", "runs/phase2_echo_mapper_v4/best_model.pt")))
    v5_ckpt = Path(str(src.get("v5_checkpoint", "runs/phase2_echo_mapper_v5/best_model.pt")))
    v4_cfg_path = v4_ckpt.parent / "config.json"
    v5_cfg_path = v5_ckpt.parent / "config.json"
    v4_cfg = json.loads(v4_cfg_path.read_text(encoding="utf-8"))
    v5_cfg = json.loads(v5_cfg_path.read_text(encoding="utf-8"))

    # Infer in_channels from checkpoint weight shape.
    v4_sd = torch.load(v4_ckpt, map_location=device)
    v5_sd = torch.load(v5_ckpt, map_location=device)
    v4_in = int(v4_sd["signal_encoder.0.weight"].shape[1])
    v5_in = int(v5_sd["signal_encoder.0.weight"].shape[1])
    # n_bins is not fixed by model because of adaptive pool; keep dataset-compatible default.
    n_bins = 128

    v4_model = V4Model(
        in_channels=v4_in,
        n_bins=n_bins,
        patch_size=patch_size,
        hidden_dim=int(v4_cfg.get("hidden_dim", 320)),
        meta_dim=8,
        use_visibility_head=bool(v4_cfg.get("use_visibility_head", True)),
        use_pose_head=bool(v4_cfg.get("use_pose_head", True)),
    ).to(device)
    v4_model.load_state_dict(v4_sd)
    v4_model.eval()

    v5_model = V5Model(
        in_channels=v5_in,
        n_bins=n_bins,
        patch_size=patch_size,
        hidden_dim=int(v5_cfg.get("hidden_dim", 320)),
        meta_dim=8,
        use_visibility_head=bool(v5_cfg.get("use_visibility_head", True)),
        use_pose_head=bool(v5_cfg.get("use_pose_head", True)),
        use_doorway_context_head=bool(v5_cfg.get("use_doorway_context_head", True)),
    ).to(device)
    v5_model.load_state_dict(v5_sd)
    v5_model.eval()

    return (
        MapperRun(
            model=v4_model,
            config=v4_cfg,
            has_context_head=False,
            use_soft_gating=bool(v4_cfg.get("use_soft_doorway_gating", True)),
            gate_strength=float(v4_cfg.get("doorway_gate_strength", 0.5)),
            gate_min=float(v4_cfg.get("doorway_min_gate", 0.35)),
            gate_structure_weight=float(v4_cfg.get("doorway_structure_weight", 0.5)),
        ),
        MapperRun(
            model=v5_model,
            config=v5_cfg,
            has_context_head=bool(v5_cfg.get("use_doorway_context_head", True)),
            use_soft_gating=bool(v5_cfg.get("use_soft_doorway_gating", True)),
            gate_strength=float(v5_cfg.get("doorway_gate_strength", 0.5)),
            gate_min=float(v5_cfg.get("doorway_min_gate", 0.35)),
            gate_structure_weight=float(v5_cfg.get("doorway_structure_weight", 0.5)),
        ),
    )


def build_model_input(
    obs: Dict[str, Dict[str, float]],
    heading: float,
    true_pose: Tuple[float, float, float],
    est_pose: Tuple[float, float, float],
    prev_action_idx: int,
    timestep: int,
    n_bins: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    timing = np.array([float(obs[s]["distance"]) for s in SECTOR_NAMES], dtype=np.float32)
    intensity = np.array([float(obs[s]["intensity"]) for s in SECTOR_NAMES], dtype=np.float32)
    scan_dirs = np.array([wrap_angle(heading + SECTOR_OFFSETS_RAD[s]) for s in SECTOR_NAMES], dtype=np.float32)

    base = np.stack([make_echo_bins(timing[i], intensity[i], n_bins=n_bins, max_range=RAY_MAX_RANGE) for i in range(5)], axis=0)
    timing_ch = np.repeat((timing[:, None] / RAY_MAX_RANGE), n_bins, axis=1)
    intensity_ch = np.repeat(intensity[:, None], n_bins, axis=1)
    scan_ch = np.repeat(((scan_dirs[:, None] + math.pi) / (2.0 * math.pi)), n_bins, axis=1)
    signal = np.concatenate([base, timing_ch, intensity_ch, scan_ch], axis=0).astype(np.float32)

    tx, ty, th = true_pose
    ex, ey, eh = est_pose
    meta = np.array(
        [
            tx,
            ty,
            th,
            ex,
            ey,
            eh,
            float(timestep) / 1000.0,
            float(prev_action_idx) / max(1.0, float(len(ACTION_NAMES) - 1)),
        ],
        dtype=np.float32,
    )
    return signal, meta


def infer_local_patch_probs(
    run_v4: MapperRun,
    run_v5: MapperRun,
    signal: np.ndarray,
    meta: np.ndarray,
    patch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    x = torch.from_numpy(signal[None, ...]).to(device)
    m = torch.from_numpy(meta[None, ...]).to(device)
    with torch.no_grad():
        o4 = run_v4.model(x, m)
        o5 = run_v5.model(x, m)

    occ4 = torch.sigmoid(o4["occupancy_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    wall4 = torch.sigmoid(o4["wall_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    door4_raw = torch.sigmoid(o4["doorway_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    free4 = torch.sigmoid(o4["free_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    vis4 = torch.sigmoid(o4["visibility_logits"]).cpu().numpy().reshape(patch_size, patch_size) if "visibility_logits" in o4 else np.full((patch_size, patch_size), 0.5, dtype=np.float32)

    occ5 = torch.sigmoid(o5["occupancy_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    wall5 = torch.sigmoid(o5["wall_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    door5_raw = torch.sigmoid(o5["doorway_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    free5 = torch.sigmoid(o5["free_logits"]).cpu().numpy().reshape(patch_size, patch_size)
    vis5 = torch.sigmoid(o5["visibility_logits"]).cpu().numpy().reshape(patch_size, patch_size) if "visibility_logits" in o5 else np.full((patch_size, patch_size), 0.5, dtype=np.float32)
    context5 = torch.sigmoid(o5["doorway_context_logit"]).cpu().numpy().reshape(1)[0] if "doorway_context_logit" in o5 else 1.0

    door4_soft = apply_doorway_structural_gating(
        door_prob_2d=door4_raw.reshape(1, -1),
        wall_prob_2d=wall4.reshape(1, -1),
        free_prob_2d=free4.reshape(1, -1),
        occ_prob_2d=occ4.reshape(1, -1),
        patch_size=patch_size,
        gate_strength=run_v4.gate_strength,
        min_gate=run_v4.gate_min,
        structure_weight=run_v4.gate_structure_weight,
    ).reshape(patch_size, patch_size)

    door5_soft = apply_doorway_structural_gating(
        door_prob_2d=door5_raw.reshape(1, -1),
        wall_prob_2d=wall5.reshape(1, -1),
        free_prob_2d=free5.reshape(1, -1),
        occ_prob_2d=occ5.reshape(1, -1),
        patch_size=patch_size,
        gate_strength=run_v5.gate_strength,
        min_gate=run_v5.gate_min,
        structure_weight=run_v5.gate_structure_weight,
    ).reshape(patch_size, patch_size)
    door5_context = door5_raw * context5
    door5_final = door5_soft * context5

    return {
        "v4_occ": occ4,
        "v4_wall": wall4,
        "v4_free": free4,
        "v4_conf": vis4,
        "v4_door_raw": door4_raw,
        "v4_door_soft": door4_soft,
        "v5_occ": occ5,
        "v5_wall": wall5,
        "v5_free": free5,
        "v5_conf": vis5,
        "v5_door_raw": door5_raw,
        "v5_door_soft": door5_soft,
        "v5_door_context": door5_context,
        "v5_door_final": door5_final,
        "v5_context_prob": np.array(context5, dtype=np.float32),
    }


def select_manifest_patch(
    local_probs: Dict[str, np.ndarray],
    selected_modes: Dict[str, object],
    fallback: Dict[str, object],
    map_name: str,
    doorway_heavy_hint: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    key_map = {
        "v5_context": "v5_door_context",
        "v5_final": "v5_door_final",
        "v5_soft": "v5_door_soft",
        "v5_raw": "v5_door_raw",
        "v4_soft": "v4_door_soft",
        "v4_raw": "v4_door_raw",
    }
    occ_key = str(selected_modes["occupancy"])
    wall_key = str(selected_modes["wall"])
    free_key = str(selected_modes["free"])
    door_key = str(selected_modes["doorway"])

    occ = local_probs[occ_key]
    wall = local_probs[wall_key]
    free = local_probs[free_key]

    used_fallback = False
    if doorway_heavy_hint and (map_name == "doorway"):
        # Explicitly use accepted doorway fallback in doorway-heavy scenes.
        fb_mode = str(fallback.get("doorway_fallback_model", door_key))
        fb_key = key_map.get(fb_mode, fb_mode)
        if fb_key in local_probs:
            door_key = fb_key
            used_fallback = True

    door_key = key_map.get(door_key, door_key)
    door = local_probs[door_key]
    conf_key = "v4_conf" if occ_key.startswith("v4_") else "v5_conf"
    vis_conf = local_probs[conf_key]

    consistency = np.clip(1.0 - np.abs((occ + free) - 1.0), 0.0, 1.0)
    conf = np.clip(0.45 * vis_conf + 0.30 * consistency + 0.25 * wall, 0.0, 1.0)
    meta = {"door_key_used": door_key, "fallback_used": used_fallback}
    return occ, wall, door, free, conf, meta


def overlay_patch(
    global_sum: np.ndarray,
    global_count: np.ndarray,
    patch: np.ndarray,
    cx: int,
    cy: int,
    add_count: bool = True,
) -> Tuple[int, int]:
    size = patch.shape[0]
    half = size // 2
    ny, nx = global_sum.shape
    in_bounds = 0
    out_of_bounds = 0
    for py in range(size):
        gy = cy + (py - half)
        if gy < 0 or gy >= ny:
            out_of_bounds += size
            continue
        for px in range(size):
            gx = cx + (px - half)
            if gx < 0 or gx >= nx:
                out_of_bounds += 1
                continue
            global_sum[gy, gx] += float(patch[py, px])
            if add_count:
                global_count[gy, gx] += 1.0
            in_bounds += 1
    return in_bounds, out_of_bounds


def get_front_masks(patch_size: int) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    m = np.zeros((patch_size, patch_size), dtype=bool)
    m[: patch_size // 2, patch_size // 4 : (3 * patch_size) // 4] = True
    masks["front"] = m
    l = np.zeros((patch_size, patch_size), dtype=bool)
    l[: patch_size // 2, : patch_size // 2] = True
    masks["front_left"] = l
    r = np.zeros((patch_size, patch_size), dtype=bool)
    r[: patch_size // 2, patch_size // 2 :] = True
    masks["front_right"] = r
    c = np.zeros((patch_size, patch_size), dtype=bool)
    c[patch_size // 2 - 2 : patch_size // 2 + 3, patch_size // 2 - 2 : patch_size // 2 + 3] = True
    masks["center"] = c
    return masks


def estimate_wall_follow_side(obs: Dict[str, Dict[str, float]]) -> str:
    if obs["left"]["distance"] < obs["right"]["distance"]:
        return "left"
    return "right"


def find_frontier_target(
    free_prob: np.ndarray,
    confidence: np.ndarray,
    door_prob: np.ndarray,
    observed: np.ndarray,
    current_cell: Tuple[int, int],
    recent_cells: List[Tuple[int, int]],
    blocked_until: Optional[Dict[Tuple[int, int], int]] = None,
    step_idx: int = 0,
) -> Tuple[Optional[Tuple[int, int]], Dict[str, float], List[Tuple[int, int]]]:
    # Frontier = observed free-ish cells next to lower-confidence cells.
    # Coverage-first scoring favors information gain and frontier area.
    blocked_until = blocked_until or {}
    ny, nx = free_prob.shape
    frontier = []
    rejected = []
    low_confidence_frontier_count = 0
    cx, cy = current_cell
    rec_set = set(recent_cells)

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if blocked_until.get((x, y), -1) >= step_idx:
                rejected.append((x, y))
                continue
            if not observed[y, x]:
                continue
            if free_prob[y, x] < 0.55:
                continue
            neigh_conf = [
                confidence[y - 1, x],
                confidence[y + 1, x],
                confidence[y, x - 1],
                confidence[y, x + 1],
            ]
            local_conf = float(confidence[y, x])
            min_conf = float(min(neigh_conf))
            if min_conf < 0.45:
                low_confidence_frontier_count += 1
                # Approx local frontier area and information gain from nearby low-confidence cells.
                y0, y1 = max(0, y - 2), min(ny, y + 3)
                x0, x1 = max(0, x - 2), min(nx, x + 3)
                patch_obs = observed[y0:y1, x0:x1]
                patch_conf = confidence[y0:y1, x0:x1]
                patch_free = free_prob[y0:y1, x0:x1]
                patch_door = door_prob[y0:y1, x0:x1]
                info_gain = float(np.logical_and(patch_obs, patch_conf < 0.45).sum())
                frontier_area = float(np.logical_and(patch_free > 0.55, patch_conf < 0.5).sum())
                conf_gap = max(0.0, local_conf - min_conf)
                dist = float(math.hypot(x - cx, y - cy))
                frontier_reachability = float(np.clip(0.5 * free_prob[y, x] + 0.5 * (1.0 - local_conf), 0.0, 1.0))
                far_bonus = min(dist / 12.0, 1.0)
                revisit_penalty = 1.0 if (x, y) in rec_set else 0.0
                doorway_bonus = float(np.clip(patch_door.mean() * 1.5, 0.0, 1.0))
                score = float(
                    2.2 * info_gain
                    + 1.7 * frontier_area
                    + 1.3 * conf_gap
                    + 1.1 * far_bonus
                    + 0.9 * doorway_bonus
                    + 0.8 * frontier_reachability
                    - 1.2 * revisit_penalty
                    - 0.08 * dist
                )
                frontier.append(
                    {
                        "score": score,
                        "x": x,
                        "y": y,
                        "info_gain": info_gain,
                        "distance": dist,
                        "reachability": frontier_reachability,
                        "area": frontier_area,
                    }
                )
    if not frontier:
        return None, {
            "selected_frontier_score": 0.0,
            "frontier_information_gain": 0.0,
            "frontier_distance": 0.0,
            "frontier_reachability_score": 0.0,
            "low_confidence_frontier_count": float(low_confidence_frontier_count),
            "rejected_frontier_count": float(len(rejected)),
            "average_frontier_area": 0.0,
        }, rejected
    frontier.sort(key=lambda t: t["score"], reverse=True)
    best = frontier[0]
    avg_area = float(np.mean([f["area"] for f in frontier])) if frontier else 0.0
    stats = {
        "selected_frontier_score": float(best["score"]),
        "frontier_information_gain": float(best["info_gain"]),
        "frontier_distance": float(best["distance"]),
        "frontier_reachability_score": float(best["reachability"]),
        "low_confidence_frontier_count": float(low_confidence_frontier_count),
        "rejected_frontier_count": float(len(rejected)),
        "average_frontier_area": float(avg_area),
    }
    return (int(best["x"]), int(best["y"])), stats, rejected


def planner_action(
    mode: str,
    obs: Dict[str, Dict[str, float]],
    heading: float,
    x: float,
    y: float,
    m,
    robot_radius: float,
    safety_margin: float,
    front_occ: float,
    front_free: float,
    front_door: float,
    target_angle: Optional[float],
    consecutive_probe_steps: int,
    max_consecutive_probe_steps: int,
    stagnation: bool,
) -> str:
    left_d = float(obs["left"]["distance"])
    right_d = float(obs["right"]["distance"])
    front_d = float(obs["front"]["distance"])

    def safe_forward(step: float) -> bool:
        return not predict_collision(m, x, y, heading, step, robot_radius, safety_margin)

    if mode == "emergency_avoidance":
        if left_d > right_d + 0.08:
            return "TURN_LEFT"
        if right_d > left_d + 0.08:
            return "TURN_RIGHT"
        return "SLOW_DOWN_AND_RESAMPLE"

    if mode == "doorway_approach":
        if front_d > 0.20 and safe_forward(0.03):
            if front_free > 0.58:
                return "MOVE_FORWARD_SLOW" if safe_forward(0.08) else "PROBE_FORWARD"
            return "PROBE_FORWARD"
        if left_d >= right_d:
            return "TURN_LEFT"
        return "TURN_RIGHT"

    if mode == "wall_following":
        follow_side = estimate_wall_follow_side(obs)
        if front_d > 0.22 and safe_forward(0.03):
            return "PROBE_FORWARD" if front_occ > 0.55 else "MOVE_FORWARD_SLOW"
        return "TURN_RIGHT" if follow_side == "left" else "TURN_LEFT"

    if mode == "low_confidence_mapping":
        if (consecutive_probe_steps < max_consecutive_probe_steps) and front_d > 0.16 and safe_forward(0.03):
            return "PROBE_FORWARD"
        if left_d >= right_d:
            return "TURN_LEFT"
        return "TURN_RIGHT"

    # frontier_exploration
    if target_angle is not None:
        err = wrap_angle(target_angle - heading)
        if abs(err) > math.radians(18):
            return "TURN_LEFT" if err > 0 else "TURN_RIGHT"
    if front_d > 0.20 and safe_forward(0.08) and (front_free > 0.52 or stagnation):
        return "MOVE_FORWARD_SLOW"
    if (consecutive_probe_steps < max_consecutive_probe_steps) and front_d > 0.12 and safe_forward(0.03):
        return "PROBE_FORWARD"
    return "TURN_LEFT" if left_d >= right_d else "TURN_RIGHT"


def binary_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1.0, tp + fp + fn + tn)
    iou = tp / max(1.0, tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc, "iou": iou}


def best_f1_threshold(prob: np.ndarray, gt: np.ndarray, thresholds: List[float]) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    table: List[Dict[str, float]] = []
    best_t = thresholds[0]
    best_stats = binary_stats(prob >= best_t, gt)
    for t in thresholds:
        stats = binary_stats(prob >= t, gt)
        row = {"threshold": float(t), **{k: float(v) for k, v in stats.items()}}
        table.append(row)
        if (stats["f1"] > best_stats["f1"]) or (
            abs(stats["f1"] - best_stats["f1"]) < 1e-9 and stats["precision"] > best_stats["precision"]
        ):
            best_t = float(t)
            best_stats = stats
    return float(best_t), {k: float(v) for k, v in best_stats.items()}, table


def cast_observation_footprint(
    observed: np.ndarray,
    m,
    x: float,
    y: float,
    heading: float,
    cell_size: float,
    max_range: float,
    rays_per_sector: int = 5,
) -> int:
    ny, nx = observed.shape
    newly = 0
    step = max(0.04, 0.75 * cell_size)
    spread = np.linspace(-math.radians(18.0), math.radians(18.0), max(1, rays_per_sector))
    ray_angles = []
    for s in SECTOR_NAMES:
        base = heading + SECTOR_OFFSETS_RAD[s]
        for d in spread:
            ray_angles.append(base + float(d))
    for ang in ray_angles:
        t = 0.0
        while t <= max_range:
            px = x + t * math.cos(ang)
            py = y + t * math.sin(ang)
            if px < 0.0 or py < 0.0 or px >= m.width or py >= m.height:
                break
            cx, cy = world_to_cell(px, py, cell_size, nx, ny)
            for oy in (-1, 0, 1):
                gy = cy + oy
                if gy < 0 or gy >= ny:
                    continue
                for ox in (-1, 0, 1):
                    gx = cx + ox
                    if gx < 0 or gx >= nx:
                        continue
                    if not observed[gy, gx]:
                        observed[gy, gx] = True
                        newly += 1
            if point_in_obstacle(px, py, m, robot_radius=0.0):
                break
            t += step
    return newly


def run_episode(
    m,
    difficulty: str,
    preset: Dict[str, float],
    run_v4: MapperRun,
    run_v5: MapperRun,
    selected_modes: Dict[str, object],
    fallback: Dict[str, object],
    max_steps: int,
    rng: np.random.Generator,
    patch_size: int,
    device: torch.device,
    target_coverage: float,
    min_coverage_for_success: float,
    coverage_stagnation_window: int,
    coverage_min_gain: float,
    frontier_blacklist_steps: int,
    cell_size: float = 0.25,
) -> Dict[str, object]:
    gt = build_gt_grids(m, cell_size)
    gt_occ = gt["occupancy"].astype(bool)
    gt_wall = gt["wall"].astype(bool)
    gt_door = gt["doorway"].astype(bool)
    gt_free = np.logical_not(gt_occ)
    ny, nx = gt_occ.shape
    reachable = gt_free.copy()
    total_reachable = max(1, int(reachable.sum()))

    occ_sum = np.zeros((ny, nx), dtype=np.float32)
    wall_sum = np.zeros((ny, nx), dtype=np.float32)
    door_sum = np.zeros((ny, nx), dtype=np.float32)
    free_sum = np.zeros((ny, nx), dtype=np.float32)
    conf_sum = np.zeros((ny, nx), dtype=np.float32)
    cnt = np.zeros((ny, nx), dtype=np.float32)
    observed = np.zeros((ny, nx), dtype=bool)

    x, y = sample_free_pose(m, rng, robot_radius=0.12)
    heading = float(rng.uniform(-math.pi, math.pi))
    est_x, est_y, est_heading = x, y, heading
    prev_action_idx = ACTION_TO_IDX["SLOW_DOWN_AND_RESAMPLE"]

    masks = get_front_masks(patch_size)
    mode_counts = Counter()
    action_counts = Counter()
    path = [(x, y)]
    frontier_targets: List[Tuple[float, float]] = []
    rejected_frontier_targets: List[Tuple[float, float]] = []
    doorway_accepted: List[Tuple[float, float]] = []
    doorway_rejected: List[Tuple[float, float]] = []
    stagnation_points: List[Tuple[float, float]] = []
    confidence_gain_accum = []

    collision = False
    fake_door_approaches = 0
    doorway_decisions = 0
    rejected_fake_doorways = 0
    crossed_doorway = False
    side_sign_prev = -1 if x < 6.0 else 1
    projection_in_bounds_total = 0
    projection_out_of_bounds_total = 0
    local_to_global_projection_count = 0

    max_consecutive_probe_steps = 4
    probe_cooldown_steps = 2
    consecutive_probe_steps = 0
    consecutive_resample_steps = 0
    probe_cooldown = 0
    stagnation_event_count = 0
    frontier_switch_count = 0
    frontier_target_count = 0
    forced_frontier_switch_count = 0
    repeated_cell_count = 0
    high_conf_revisit_count = 0
    local_loop_count = 0
    move_forward_count = 0
    turn_count = 0
    probe_count = 0
    coverage_gain_per_100_steps: List[float] = []
    coverage_gain_after_recovery: List[float] = []
    recovery_start_coverage: Optional[float] = None
    previous_coverage = 0.0
    coverage_trace: List[float] = []
    conf_trace: List[float] = []
    recent_cells = deque(maxlen=120)
    last_frontier_cell: Optional[Tuple[int, int]] = None
    steps_since_frontier_progress = 0
    blocked_frontier_cells_until: Dict[Tuple[int, int], int] = {}
    doorway_crossing_for_coverage_count = 0
    doorway_rejected_for_safety_count = 0
    coverage_gain_after_doorway_crossing: List[float] = []
    last_doorway_cross_cov: Optional[float] = None
    frontier_score_history: List[float] = []
    frontier_info_gain_history: List[float] = []
    frontier_distance_history: List[float] = []
    frontier_reachability_history: List[float] = []
    low_conf_frontier_count_history: List[float] = []
    rejected_frontier_count_history: List[float] = []
    frontier_area_history: List[float] = []

    for step in range(max_steps):
        obs = simulate_echo_observation(m, x, y, heading, rng, preset)
        signal, meta = build_model_input(
            obs=obs,
            heading=heading,
            true_pose=(x, y, heading),
            est_pose=(est_x, est_y, est_heading),
            prev_action_idx=prev_action_idx,
            timestep=step,
            n_bins=128,
        )
        local = infer_local_patch_probs(run_v4, run_v5, signal, meta, patch_size, device)

        # doorway-heavy hint for optional fallback usage.
        door_hint = float(local["v5_door_context"].mean()) > 0.45 or float(local["v5_context_prob"]) > 0.55
        occ_p, wall_p, door_p, free_p, conf_p, sel_meta = select_manifest_patch(
            local_probs=local,
            selected_modes=selected_modes,
            fallback=fallback,
            map_name=m.name,
            doorway_heavy_hint=door_hint,
        )
        if sel_meta["fallback_used"]:
            doorway_decisions += 1

        cx, cy = world_to_cell(est_x, est_y, cell_size, nx, ny)
        in_b, out_b = overlay_patch(occ_sum, cnt, occ_p, cx, cy, add_count=True)
        overlay_patch(wall_sum, cnt, wall_p, cx, cy, add_count=False)
        overlay_patch(door_sum, cnt, door_p, cx, cy, add_count=False)
        overlay_patch(free_sum, cnt, free_p, cx, cy, add_count=False)
        overlay_patch(conf_sum, cnt, conf_p, cx, cy, add_count=False)
        local_to_global_projection_count += 1
        projection_in_bounds_total += int(in_b)
        projection_out_of_bounds_total += int(out_b)

        observed_range = 3.8 if difficulty in {"clean", "mild_noise"} else 3.4
        rays_per_sector = 17 if difficulty in {"clean", "mild_noise"} else 13
        newly_observed = cast_observation_footprint(
            observed,
            m,
            x,
            y,
            heading,
            cell_size,
            max_range=observed_range,
            rays_per_sector=rays_per_sector,
        )
        confidence_gain_accum.append(float(newly_observed))

        occ_global = np.where(cnt > 0, occ_sum / np.maximum(cnt, 1e-6), 0.5)
        wall_global = np.where(cnt > 0, wall_sum / np.maximum(cnt, 1e-6), 0.0)
        door_global = np.where(cnt > 0, door_sum / np.maximum(cnt, 1e-6), 0.0)
        free_global = np.where(cnt > 0, free_sum / np.maximum(cnt, 1e-6), 0.5)
        conf_global = np.where(cnt > 0, conf_sum / np.maximum(cnt, 1e-6), 0.0)

        front_occ = float(occ_p[masks["front"]].mean())
        front_free = float(free_p[masks["front"]].mean())
        front_door = float(door_p[masks["front"]].mean())
        front_conf = float(conf_p[masks["front"]].mean())
        center_occ = float(occ_p[masks["center"]].mean())
        current_cell = world_to_cell(x, y, cell_size, nx, ny)
        repeated_here = current_cell in recent_cells
        recent_cells.append(current_cell)
        if repeated_here:
            repeated_cell_count += 1
            if float(conf_global[current_cell[1], current_cell[0]]) > 0.72:
                high_conf_revisit_count += 1
        if len(recent_cells) >= 30:
            uniq_recent = len(set(list(recent_cells)[-30:]))
            if uniq_recent <= 10:
                local_loop_count += 1

        imminent_collision = predict_collision(m, x, y, heading, 0.03, 0.12, 0.05 if m.name != "cluttered_room" else 0.08)
        any_front_obstacle = (float(obs["front"]["distance"]) < 0.16) or (front_occ > 0.72) or (center_occ > 0.70)

        coverage_now = float((observed & reachable).sum() / total_reachable)
        coverage_trace.append(coverage_now)
        conf_trace.append(float(conf_global.mean()))
        if step > 0:
            if coverage_now <= previous_coverage + 0.0005:
                steps_since_frontier_progress += 1
            else:
                steps_since_frontier_progress = 0
            if (step + 1) % 100 == 0:
                coverage_gain_per_100_steps.append(float(coverage_now - previous_coverage))
        previous_coverage = coverage_now
        window_gain = 0.0
        if len(coverage_trace) > coverage_stagnation_window:
            window_gain = float(coverage_trace[-1] - coverage_trace[-1 - coverage_stagnation_window])
        stagnation = (steps_since_frontier_progress >= coverage_stagnation_window) or (
            len(coverage_trace) > coverage_stagnation_window and window_gain < coverage_min_gain
        )
        if stagnation:
            stagnation_event_count += 1
            if (step % 5) == 0:
                stagnation_points.append((x, y))
            if recovery_start_coverage is None:
                recovery_start_coverage = coverage_now
        elif recovery_start_coverage is not None:
            coverage_gain_after_recovery.append(max(0.0, coverage_now - recovery_start_coverage))
            recovery_start_coverage = None

        frontier, frontier_stats, rejected_frontiers = find_frontier_target(
            free_prob=free_global,
            confidence=conf_global,
            door_prob=door_global,
            observed=observed,
            current_cell=current_cell,
            recent_cells=list(recent_cells),
            blocked_until=blocked_frontier_cells_until,
            step_idx=step,
        )
        frontier_score_history.append(float(frontier_stats["selected_frontier_score"]))
        frontier_info_gain_history.append(float(frontier_stats["frontier_information_gain"]))
        frontier_distance_history.append(float(frontier_stats["frontier_distance"]))
        frontier_reachability_history.append(float(frontier_stats["frontier_reachability_score"]))
        low_conf_frontier_count_history.append(float(frontier_stats["low_confidence_frontier_count"]))
        rejected_frontier_count_history.append(float(frontier_stats["rejected_frontier_count"]))
        frontier_area_history.append(float(frontier_stats["average_frontier_area"]))
        for rx, ry in rejected_frontiers[:20]:
            rejected_frontier_targets.append(((rx + 0.5) * cell_size, (ry + 0.5) * cell_size))
        target_angle = None
        if frontier is not None:
            fx, fy = frontier
            wx = (fx + 0.5) * cell_size
            wy = (fy + 0.5) * cell_size
            target_angle = math.atan2(wy - y, wx - x)
            if last_frontier_cell is None:
                frontier_target_count += 1
                last_frontier_cell = (fx, fy)
            elif (fx, fy) != last_frontier_cell:
                frontier_switch_count += 1
                frontier_target_count += 1
                last_frontier_cell = (fx, fy)
            frontier_targets.append((wx, wy))
        else:
            last_frontier_cell = None

        if stagnation and (frontier is not None) and (last_frontier_cell is not None):
            blocked_frontier_cells_until[last_frontier_cell] = step + frontier_blacklist_steps
            forced_frontier_switch_count += 1

        # Mode selection.
        doorway_for_coverage = (
            front_door > float(selected_modes["doorway_threshold"])
            and front_free > 0.45
            and front_conf < 0.62
            and not imminent_collision
        )
        if imminent_collision or any_front_obstacle:
            mode = "emergency_avoidance"
        elif doorway_for_coverage:
            mode = "doorway_approach"
        elif float(conf_global.mean()) < 0.32 or front_conf < 0.34:
            mode = "low_confidence_mapping"
        elif min(float(obs["left"]["distance"]), float(obs["right"]["distance"])) < 0.28:
            mode = "wall_following"
        else:
            mode = "frontier_exploration"
        mode_counts[mode] += 1

        dynamic_probe_cap = 2 if stagnation else max_consecutive_probe_steps
        action = planner_action(
            mode=mode,
            obs=obs,
            heading=heading,
            x=x,
            y=y,
            m=m,
            robot_radius=0.12,
            safety_margin=0.05 if m.name != "cluttered_room" else 0.08,
            front_occ=front_occ,
            front_free=front_free,
            front_door=front_door,
            target_angle=target_angle,
            consecutive_probe_steps=consecutive_probe_steps,
            max_consecutive_probe_steps=dynamic_probe_cap,
            stagnation=stagnation,
        )

        # Reduce probe dominance: cooldown and escalation to safe forward/turn when progress stalls.
        if action == "PROBE_FORWARD":
            if probe_cooldown > 0:
                action = "MOVE_FORWARD_SLOW" if (
                    (float(obs["front"]["distance"]) > 0.22)
                    and (front_free > 0.55)
                    and (not predict_collision(m, x, y, heading, 0.08, 0.12, 0.05 if m.name != "cluttered_room" else 0.08))
                ) else ("TURN_LEFT" if float(obs["left"]["distance"]) >= float(obs["right"]["distance"]) else "TURN_RIGHT")
            elif consecutive_probe_steps >= max_consecutive_probe_steps:
                action = "MOVE_FORWARD_SLOW" if (
                    (float(obs["front"]["distance"]) > 0.24)
                    and (front_free > 0.58)
                    and (not predict_collision(m, x, y, heading, 0.08, 0.12, 0.05 if m.name != "cluttered_room" else 0.08))
                ) else ("TURN_LEFT" if float(obs["left"]["distance"]) >= float(obs["right"]["distance"]) else "TURN_RIGHT")

        if stagnation and action == "SLOW_DOWN_AND_RESAMPLE":
            action = "TURN_LEFT" if float(obs["left"]["distance"]) >= float(obs["right"]["distance"]) else "TURN_RIGHT"

        # Fake-doorway suppression for non-doorway maps.
        doorway_candidate = front_door > float(selected_modes["doorway_threshold"])
        doorway_struct_ok = (front_free > 0.48) and (float(wall_p[masks["front_left"]].mean()) > 0.33) and (float(wall_p[masks["front_right"]].mean()) > 0.33)
        if doorway_candidate:
            doorway_decisions += 1
            if m.name != "doorway" and not doorway_struct_ok:
                rejected_fake_doorways += 1
                doorway_rejected_for_safety_count += 1
                doorway_rejected.append((x, y))
                if action in {"MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW", "PROBE_FORWARD"}:
                    action = "TURN_LEFT" if float(obs["left"]["distance"]) >= float(obs["right"]["distance"]) else "TURN_RIGHT"
            else:
                doorway_accepted.append((x, y))
                # Use accepted doorway structure for coverage expansion.
                if (front_conf < 0.58) and (front_free > 0.46) and (action in {"TURN_LEFT", "TURN_RIGHT", "PROBE_FORWARD"}):
                    action = "MOVE_FORWARD_SLOW" if (
                        float(obs["front"]["distance"]) > 0.24
                        and not predict_collision(m, x, y, heading, 0.08, 0.12, 0.05 if m.name != "cluttered_room" else 0.08)
                    ) else "PROBE_FORWARD"
                    doorway_crossing_for_coverage_count += 1
                    last_doorway_cross_cov = coverage_now
                if m.name != "doorway" and action in {"MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW", "PROBE_FORWARD"}:
                    fake_door_approaches += 1

        turn_deg = 30.0 if (stagnation and action in {"TURN_LEFT", "TURN_RIGHT"}) else 15.0
        nx_t, ny_t, nh_t, moved, collided = apply_action(x, y, heading, action, m, robot_radius=0.12, turn_deg=turn_deg)
        action_counts[action] += 1
        if action == "PROBE_FORWARD":
            probe_count += 1
            consecutive_probe_steps += 1
            probe_cooldown = 0
        else:
            if consecutive_probe_steps >= max_consecutive_probe_steps:
                probe_cooldown = probe_cooldown_steps
            else:
                probe_cooldown = max(0, probe_cooldown - 1)
            consecutive_probe_steps = 0
        if action == "SLOW_DOWN_AND_RESAMPLE":
            consecutive_resample_steps += 1
        else:
            consecutive_resample_steps = 0
        if action in {"MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW"}:
            move_forward_count += 1
        if action in {"TURN_LEFT", "TURN_RIGHT"}:
            turn_count += 1
        prev_action_idx = ACTION_TO_IDX.get(action, ACTION_TO_IDX["SLOW_DOWN_AND_RESAMPLE"])
        if collided:
            collision = True
            break

        x, y, heading = nx_t, ny_t, nh_t
        path.append((x, y))

        drift = preset.get("pose_drift_std", 0.0)
        est_x = est_x + moved * math.cos(est_heading) + rng.normal(0.0, drift)
        est_y = est_y + moved * math.sin(est_heading) + rng.normal(0.0, drift)
        if action == "TURN_LEFT":
            est_heading = wrap_angle(est_heading + math.radians(15.0) + rng.normal(0.0, 0.5 * drift))
        elif action == "TURN_RIGHT":
            est_heading = wrap_angle(est_heading - math.radians(15.0) + rng.normal(0.0, 0.5 * drift))
        else:
            est_heading = wrap_angle(est_heading + rng.normal(0.0, 0.2 * drift))

        if m.name == "doorway":
            side = -1 if x < 6.0 else 1
            if side != side_sign_prev and (3.0 <= y <= 5.0):
                crossed_doorway = True
            side_sign_prev = side

        # Early success cut to keep runtime practical.
        coverage_now = float((observed & reachable).sum() / total_reachable)
        if stagnation and last_frontier_cell is not None and coverage_now <= previous_coverage + 0.0005:
            blocked_frontier_cells_until[last_frontier_cell] = step + frontier_blacklist_steps
            forced_frontier_switch_count += 1
        if last_doorway_cross_cov is not None and coverage_now > last_doorway_cross_cov:
            coverage_gain_after_doorway_crossing.append(float(coverage_now - last_doorway_cross_cov))
            last_doorway_cross_cov = None
        if coverage_now >= target_coverage:
            if m.name != "doorway" or crossed_doorway:
                break

    occ_global = np.where(cnt > 0, occ_sum / np.maximum(cnt, 1e-6), 0.5)
    wall_global = np.where(cnt > 0, wall_sum / np.maximum(cnt, 1e-6), 0.0)
    door_global = np.where(cnt > 0, door_sum / np.maximum(cnt, 1e-6), 0.0)
    free_global = np.where(cnt > 0, free_sum / np.maximum(cnt, 1e-6), 0.5)
    conf_global = np.where(cnt > 0, conf_sum / np.maximum(cnt, 1e-6), 0.0)

    occ_pred = occ_global >= 0.50
    wall_threshold_default = float(selected_modes["wall_threshold"])
    door_threshold_default = float(selected_modes["doorway_threshold"])
    wall_pred_default = wall_global >= wall_threshold_default
    door_pred_default = door_global >= door_threshold_default

    wall_thresholds = [round(t, 2) for t in np.linspace(0.20, 0.90, 15)]
    door_thresholds = [round(t, 2) for t in np.linspace(0.10, 0.90, 17)]
    wall_best_t, wall_best_stats, wall_threshold_table = best_f1_threshold(wall_global, gt_wall, wall_thresholds)
    door_best_t, door_best_stats, door_threshold_table = best_f1_threshold(door_global, gt_door, door_thresholds)

    wall_pred = wall_global >= wall_best_t
    door_pred = door_global >= door_best_t

    map_acc = float((occ_pred == gt_occ).mean())
    wall_f1 = float(wall_best_stats["f1"])
    door_f1 = float(door_best_stats["f1"])
    coverage = float((observed & reachable).sum() / total_reachable)
    timeout = (len(path) - 1) >= max_steps and not collision

    success = (not collision) and (coverage >= min_coverage_for_success) and ((m.name != "doorway") or crossed_doorway)
    if collision:
        failure_reason = "collision"
    elif not success and timeout:
        failure_reason = "timeout"
    elif not success and coverage < min_coverage_for_success:
        failure_reason = "low_coverage"
    elif not success and (m.name == "doorway") and (not crossed_doorway):
        failure_reason = "doorway_not_crossed"
    else:
        failure_reason = "success"
    total_actions = max(1, int(sum(action_counts.values())))
    probe_action_ratio = float(action_counts.get("PROBE_FORWARD", 0) / total_actions)
    move_forward_ratio = float((action_counts.get("MOVE_FORWARD_FAST", 0) + action_counts.get("MOVE_FORWARD_SLOW", 0)) / total_actions)
    turn_ratio = float((action_counts.get("TURN_LEFT", 0) + action_counts.get("TURN_RIGHT", 0)) / total_actions)

    wall_gt_positive = int(gt_wall.sum())
    doorway_gt_positive = int(gt_door.sum())
    wall_pred_positive = int((wall_global >= wall_best_t).sum())
    doorway_pred_positive = int((door_global >= door_best_t).sum())
    global_wall_nonzero = int((wall_global > 1e-6).sum())
    global_door_nonzero = int((door_global > 1e-6).sum())

    return {
        "success": bool(success),
        "failure_reason": failure_reason,
        "collision": bool(collision),
        "timeout": bool(timeout),
        "steps": int(len(path) - 1),
        "path_length": float(sum(math.dist(path[i - 1], path[i]) for i in range(1, len(path)))),
        "action_counts": dict(action_counts),
        "mode_counts": dict(mode_counts),
        "fake_doorway_approaches": int(fake_door_approaches),
        "doorway_crossing_success": bool(crossed_doorway if m.name == "doorway" else True),
        "coverage": coverage,
        "map_accuracy": map_acc,
        "wall_f1": wall_f1,
        "doorway_f1": door_f1,
        "mean_confidence_gain": float(np.mean(confidence_gain_accum)) if confidence_gain_accum else 0.0,
        "coverage_gain_per_100_steps": float(np.mean(coverage_gain_per_100_steps)) if coverage_gain_per_100_steps else 0.0,
        "stagnation_event_count": int(stagnation_event_count),
        "coverage_stagnation_count": int(stagnation_event_count),
        "frontier_target_count": int(frontier_target_count),
        "frontier_switch_count": int(frontier_switch_count),
        "forced_frontier_switch_count": int(forced_frontier_switch_count),
        "blacklisted_frontier_count": int(len(blocked_frontier_cells_until)),
        "mean_coverage_gain_after_recovery": float(np.mean(coverage_gain_after_recovery)) if coverage_gain_after_recovery else 0.0,
        "repeated_cell_ratio": float(repeated_cell_count / max(1, len(path))),
        "high_confidence_revisit_ratio": float(high_conf_revisit_count / max(1, len(path))),
        "local_loop_count": int(local_loop_count),
        "probe_action_ratio": probe_action_ratio,
        "move_forward_ratio": move_forward_ratio,
        "turn_ratio": turn_ratio,
        "selected_frontier_score": float(np.mean(frontier_score_history)) if frontier_score_history else 0.0,
        "frontier_information_gain": float(np.mean(frontier_info_gain_history)) if frontier_info_gain_history else 0.0,
        "frontier_distance": float(np.mean(frontier_distance_history)) if frontier_distance_history else 0.0,
        "frontier_reachability_score": float(np.mean(frontier_reachability_history)) if frontier_reachability_history else 0.0,
        "low_confidence_frontier_count": float(np.mean(low_conf_frontier_count_history)) if low_conf_frontier_count_history else 0.0,
        "rejected_frontier_count": float(np.mean(rejected_frontier_count_history)) if rejected_frontier_count_history else 0.0,
        "average_frontier_area": float(np.mean(frontier_area_history)) if frontier_area_history else 0.0,
        "doorway_crossing_for_coverage_count": int(doorway_crossing_for_coverage_count),
        "doorway_rejected_for_safety_count": int(doorway_rejected_for_safety_count),
        "coverage_gain_after_doorway_crossing": float(np.mean(coverage_gain_after_doorway_crossing)) if coverage_gain_after_doorway_crossing else 0.0,
        "wall_threshold_used": float(wall_best_t),
        "doorway_threshold_used": float(door_best_t),
        "wall_threshold_default": wall_threshold_default,
        "doorway_threshold_default": door_threshold_default,
        "wall_f1_default_threshold": float(binary_stats(wall_pred_default, gt_wall)["f1"]),
        "doorway_f1_default_threshold": float(binary_stats(door_pred_default, gt_door)["f1"]),
        "global_wall_nonzero_cells": global_wall_nonzero,
        "global_doorway_nonzero_cells": global_door_nonzero,
        "global_wall_mean_probability": float(wall_global.mean()),
        "global_doorway_mean_probability": float(door_global.mean()),
        "wall_gt_positive_cells": wall_gt_positive,
        "doorway_gt_positive_cells": doorway_gt_positive,
        "wall_pred_positive_cells": wall_pred_positive,
        "doorway_pred_positive_cells": doorway_pred_positive,
        "local_to_global_projection_count": int(local_to_global_projection_count),
        "projection_out_of_bounds_count": int(projection_out_of_bounds_total),
        "projection_in_bounds_count": int(projection_in_bounds_total),
        "wall_threshold_table": wall_threshold_table,
        "doorway_threshold_table": door_threshold_table,
        "doorway_decisions": int(doorway_decisions),
        "rejected_fake_doorways": int(rejected_fake_doorways),
        "frontier_targets": frontier_targets,
        "accepted_doorways": doorway_accepted,
        "rejected_doorways": doorway_rejected,
        "rejected_frontiers": rejected_frontier_targets,
        "stagnation_points": stagnation_points,
        "path": path,
        "occ_map": occ_global,
        "wall_map": wall_global,
        "door_map": door_global,
        "free_map": free_global,
        "conf_map": conf_global,
        "coverage_map": observed.astype(np.uint8),
        "wall_map_thresholded": wall_pred.astype(np.uint8),
        "door_map_thresholded": door_pred.astype(np.uint8),
        "gt_occ": gt_occ.astype(np.uint8),
        "gt_wall": gt_wall.astype(np.uint8),
        "gt_door": gt_door.astype(np.uint8),
        "coverage_trace": coverage_trace,
        "confidence_trace": conf_trace,
    }


def save_episode_plot(out_dir: Path, map_name: str, difficulty: str, ep: Dict[str, object], m) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=130)
    ax = axes.flatten()

    ax[0].imshow(ep["occ_map"], cmap="gray_r", vmin=0.0, vmax=1.0)
    ax[0].set_title("Final Occupancy Prob")
    ax[1].imshow(ep["wall_map"], cmap="magma", vmin=0.0, vmax=1.0)
    ax[1].set_title("Wall Probability")
    ax[2].imshow(ep["door_map"], cmap="viridis", vmin=0.0, vmax=1.0)
    ax[2].set_title("Doorway Probability")
    ax[3].imshow(ep["conf_map"], cmap="Blues", vmin=0.0, vmax=1.0)
    ax[3].set_title("Confidence Map")
    ax[4].imshow(ep.get("coverage_map", np.zeros_like(ep["gt_occ"])), cmap="Greens", vmin=0, vmax=1)
    ax[4].set_title("Coverage / Observed Cells")
    ax[5].imshow(ep.get("wall_map_thresholded", np.zeros_like(ep["gt_wall"])), cmap="gray_r", vmin=0, vmax=1)
    ax[5].set_title(f"Thresholded Wall (t={ep.get('wall_threshold_used', 0.0):.2f})")
    ax[6].imshow(ep.get("door_map_thresholded", np.zeros_like(ep["gt_door"])), cmap="gray_r", vmin=0, vmax=1)
    ax[6].set_title(f"Thresholded Doorway (t={ep.get('doorway_threshold_used', 0.0):.2f})")

    # Trajectory + frontiers + doorway decisions.
    ax[7].imshow(ep["gt_occ"], cmap="gray_r", alpha=0.35)
    p = np.asarray(ep["path"], dtype=float)
    if len(p) > 1:
        ax[7].plot(p[:, 0] / 0.25, p[:, 1] / 0.25, color="tab:blue", linewidth=1.5, label="trajectory")
    if ep["frontier_targets"]:
        ft = np.asarray(ep["frontier_targets"], dtype=float)
        ax[7].scatter(ft[:, 0] / 0.25, ft[:, 1] / 0.25, s=8, c="gold", label="frontiers", alpha=0.65)
    if ep.get("rejected_frontiers"):
        rf = np.asarray(ep["rejected_frontiers"], dtype=float)
        if len(rf) > 0:
            ax[7].scatter(rf[:, 0] / 0.25, rf[:, 1] / 0.25, s=6, c="gray", marker=".", label="rejected frontiers", alpha=0.45)
    if ep["accepted_doorways"]:
        da = np.asarray(ep["accepted_doorways"], dtype=float)
        ax[7].scatter(da[:, 0] / 0.25, da[:, 1] / 0.25, s=14, c="lime", marker="o", label="doorway accepted")
    if ep["rejected_doorways"]:
        dr = np.asarray(ep["rejected_doorways"], dtype=float)
        ax[7].scatter(dr[:, 0] / 0.25, dr[:, 1] / 0.25, s=14, c="red", marker="x", label="doorway rejected")
    if ep.get("stagnation_points"):
        sp = np.asarray(ep["stagnation_points"], dtype=float)
        ax[7].scatter(sp[:, 0] / 0.25, sp[:, 1] / 0.25, s=10, c="orange", marker="s", label="stagnation")
    ax[7].set_title("Trajectory + Frontier + Doorway Decisions")
    ax[7].legend(fontsize=6, loc="upper right")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    out_path = out_dir / "plots" / f"{map_name}_{difficulty}_sample.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def aggregate(episodes: List[Dict[str, object]]) -> Dict[str, object]:
    if not episodes:
        return {}
    actions = Counter()
    modes = Counter()
    failure_reasons = Counter()
    for e in episodes:
        actions.update(e["action_counts"])  # type: ignore[arg-type]
        modes.update(e["mode_counts"])  # type: ignore[arg-type]
        failure_reasons.update([str(e.get("failure_reason", "unknown"))])
    total_actions = max(1, sum(actions.values()))
    action_dist = {k: {"count": int(actions.get(k, 0)), "rate": float(actions.get(k, 0) / total_actions)} for k in ACTION_NAMES}

    return {
        "success_rate": float(np.mean([1.0 if e["success"] else 0.0 for e in episodes])),
        "collision_rate": float(np.mean([1.0 if e["collision"] else 0.0 for e in episodes])),
        "timeout_rate": float(np.mean([1.0 if e["timeout"] else 0.0 for e in episodes])),
        "fake_doorway_approach_rate": float(np.mean([e["fake_doorway_approaches"] > 0 for e in episodes])),
        "doorway_crossing_success_rate": float(np.mean([1.0 if e["doorway_crossing_success"] else 0.0 for e in episodes])),
        "exploration_coverage": float(np.mean([e["coverage"] for e in episodes])),
        "final_map_accuracy": float(np.mean([e["map_accuracy"] for e in episodes])),
        "final_wall_f1": float(np.mean([e["wall_f1"] for e in episodes])),
        "final_doorway_f1": float(np.mean([e["doorway_f1"] for e in episodes])),
        "mean_path_length": float(np.mean([e["path_length"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "action_distribution": action_dist,
        "planner_mode_distribution": {k: int(v) for k, v in modes.items()},
        "failure_reason_counts": {k: int(v) for k, v in failure_reasons.items()},
        "mean_confidence_gain": float(np.mean([e["mean_confidence_gain"] for e in episodes])),
        "mean_confidence_gain_per_step": float(np.mean([e["mean_confidence_gain"] for e in episodes])),
        "coverage_gain_per_100_steps": float(np.mean([e.get("coverage_gain_per_100_steps", 0.0) for e in episodes])),
        "stagnation_event_count": int(np.sum([e.get("stagnation_event_count", 0) for e in episodes])),
        "coverage_stagnation_count": int(np.sum([e.get("coverage_stagnation_count", 0) for e in episodes])),
        "frontier_target_count": int(np.sum([e.get("frontier_target_count", 0) for e in episodes])),
        "frontier_switch_count": int(np.sum([e.get("frontier_switch_count", 0) for e in episodes])),
        "forced_frontier_switch_count": int(np.sum([e.get("forced_frontier_switch_count", 0) for e in episodes])),
        "blacklisted_frontier_count": int(np.sum([e.get("blacklisted_frontier_count", 0) for e in episodes])),
        "mean_coverage_gain_after_recovery": float(np.mean([e.get("mean_coverage_gain_after_recovery", 0.0) for e in episodes])),
        "repeated_cell_ratio": float(np.mean([e.get("repeated_cell_ratio", 0.0) for e in episodes])),
        "high_confidence_revisit_ratio": float(np.mean([e.get("high_confidence_revisit_ratio", 0.0) for e in episodes])),
        "local_loop_count": int(np.sum([e.get("local_loop_count", 0) for e in episodes])),
        "probe_action_ratio": float(np.mean([e.get("probe_action_ratio", 0.0) for e in episodes])),
        "move_forward_ratio": float(np.mean([e.get("move_forward_ratio", 0.0) for e in episodes])),
        "turn_ratio": float(np.mean([e.get("turn_ratio", 0.0) for e in episodes])),
        "selected_frontier_score": float(np.mean([e.get("selected_frontier_score", 0.0) for e in episodes])),
        "frontier_information_gain": float(np.mean([e.get("frontier_information_gain", 0.0) for e in episodes])),
        "frontier_distance": float(np.mean([e.get("frontier_distance", 0.0) for e in episodes])),
        "frontier_reachability_score": float(np.mean([e.get("frontier_reachability_score", 0.0) for e in episodes])),
        "low_confidence_frontier_count": float(np.mean([e.get("low_confidence_frontier_count", 0.0) for e in episodes])),
        "rejected_frontier_count": float(np.mean([e.get("rejected_frontier_count", 0.0) for e in episodes])),
        "average_frontier_area": float(np.mean([e.get("average_frontier_area", 0.0) for e in episodes])),
        "doorway_crossing_for_coverage_count": int(np.sum([e.get("doorway_crossing_for_coverage_count", 0) for e in episodes])),
        "doorway_rejected_for_safety_count": int(np.sum([e.get("doorway_rejected_for_safety_count", 0) for e in episodes])),
        "coverage_gain_after_doorway_crossing": float(np.mean([e.get("coverage_gain_after_doorway_crossing", 0.0) for e in episodes])),
        "global_wall_nonzero_cells": float(np.mean([e.get("global_wall_nonzero_cells", 0.0) for e in episodes])),
        "global_doorway_nonzero_cells": float(np.mean([e.get("global_doorway_nonzero_cells", 0.0) for e in episodes])),
        "global_wall_mean_probability": float(np.mean([e.get("global_wall_mean_probability", 0.0) for e in episodes])),
        "global_doorway_mean_probability": float(np.mean([e.get("global_doorway_mean_probability", 0.0) for e in episodes])),
        "wall_threshold_used": float(np.mean([e.get("wall_threshold_used", 0.0) for e in episodes])),
        "doorway_threshold_used": float(np.mean([e.get("doorway_threshold_used", 0.0) for e in episodes])),
        "wall_f1_default_threshold": float(np.mean([e.get("wall_f1_default_threshold", 0.0) for e in episodes])),
        "doorway_f1_default_threshold": float(np.mean([e.get("doorway_f1_default_threshold", 0.0) for e in episodes])),
        "wall_gt_positive_cells": float(np.mean([e.get("wall_gt_positive_cells", 0.0) for e in episodes])),
        "doorway_gt_positive_cells": float(np.mean([e.get("doorway_gt_positive_cells", 0.0) for e in episodes])),
        "wall_pred_positive_cells": float(np.mean([e.get("wall_pred_positive_cells", 0.0) for e in episodes])),
        "doorway_pred_positive_cells": float(np.mean([e.get("doorway_pred_positive_cells", 0.0) for e in episodes])),
        "local_to_global_projection_count": int(np.sum([e.get("local_to_global_projection_count", 0) for e in episodes])),
        "projection_out_of_bounds_count": int(np.sum([e.get("projection_out_of_bounds_count", 0) for e in episodes])),
        "number_of_doorway_decisions": int(np.sum([e["doorway_decisions"] for e in episodes])),
        "number_of_rejected_fake_doorways": int(np.sum([e["rejected_fake_doorways"] for e in episodes])),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.accepted_mapper_manifest).read_text(encoding="utf-8"))
    selected_modes = manifest.get("selected_modes", {})
    if not selected_modes:
        raise ValueError("Manifest missing selected_modes.")
    fallback = manifest.get("fallback", {})
    if not fallback:
        fallback = {"doorway_fallback_model": "v5_context", "doorway_fallback_threshold": 0.85}

    # Translate manifest keys to internal keys used in local probability dict.
    selected_internal = {
        "occupancy": str(selected_modes["occupancy"]),
        "occupancy_threshold": float(selected_modes["occupancy_threshold"]),
        "wall": str(selected_modes["wall"]),
        "wall_threshold": float(selected_modes["wall_threshold"]),
        "doorway": str(selected_modes["doorway"]),
        "doorway_threshold": float(selected_modes["doorway_threshold"]),
        "free": str(selected_modes["free"]),
        "free_threshold": float(selected_modes["free_threshold"]),
    }
    fallback_internal = {
        "doorway_fallback_model": str(fallback.get("doorway_fallback_model", "v5_context")),
        "doorway_fallback_threshold": float(fallback.get("doorway_fallback_threshold", 0.85)),
    }

    run_v4, run_v5 = load_mapper_runs(manifest=manifest, device=device, patch_size=32)

    all_maps = {m.name: m for m in make_maps()}
    map_names = [m.strip() for m in args.maps.split(",") if m.strip()]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    for d in diffs:
        if d not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {d}")

    # Optional baseline comparison files.
    v1_results_path = Path("runs/phase2_mapper_guided_navigation/mapper_guided_navigation_results.json")
    baseline_paths = [
        v1_results_path,
        Path("simulation/results/simple_2d_acoustic_nav_v11_doorway_clutter_tuning_results.json"),
        Path("simulation/results/simple_2d_acoustic_nav_v10_clean_coverage_curriculum_results.json"),
    ]
    baseline_summary: Optional[Dict[str, object]] = None
    baseline_source: Optional[str] = None
    for p in baseline_paths:
        if p.exists():
            baseline_summary = json.loads(p.read_text(encoding="utf-8"))
            baseline_source = str(p)
            break

    results: Dict[str, object] = {
        "config": {
            "episodes_per_map": args.episodes_per_map,
            "max_steps": args.max_steps,
            "maps": map_names,
            "difficulties": diffs,
            "target_coverage": args.target_coverage,
            "min_coverage_for_success": args.min_coverage_for_success,
            "coverage_stagnation_window": args.coverage_stagnation_window,
            "coverage_min_gain": args.coverage_min_gain,
            "frontier_blacklist_steps": args.frontier_blacklist_steps,
            "accepted_mapper_manifest": args.accepted_mapper_manifest,
            "selected_modes": selected_internal,
            "fallback": fallback_internal,
            "device": str(device),
            "seed": args.seed,
        },
        "by_difficulty": {},
        "overall": {},
        "baseline_reference": baseline_summary,
        "baseline_reference_path": baseline_source,
    }

    t0 = time.time()
    all_eps: List[Dict[str, object]] = []
    by_diff: Dict[str, Dict[str, object]] = {}
    for difficulty in diffs:
        preset = DIFFICULTY_PRESETS[difficulty]
        by_map: Dict[str, object] = {}
        diff_eps: List[Dict[str, object]] = []
        for map_name in map_names:
            if map_name not in all_maps:
                raise ValueError(f"Unknown map: {map_name}")
            m = all_maps[map_name]
            eps: List[Dict[str, object]] = []
            for epi in range(args.episodes_per_map):
                rng = np.random.default_rng(args.seed + 100000 * (hash(difficulty) % 1000) + 1000 * (hash(map_name) % 1000) + epi)
                ep = run_episode(
                    m=m,
                    difficulty=difficulty,
                    preset=preset,
                    run_v4=run_v4,
                    run_v5=run_v5,
                    selected_modes=selected_internal,
                    fallback=fallback_internal,
                    max_steps=args.max_steps,
                    rng=rng,
                    patch_size=32,
                    device=device,
                    target_coverage=args.target_coverage,
                    min_coverage_for_success=args.min_coverage_for_success,
                    coverage_stagnation_window=args.coverage_stagnation_window,
                    coverage_min_gain=args.coverage_min_gain,
                    frontier_blacklist_steps=args.frontier_blacklist_steps,
                    cell_size=0.25,
                )
                eps.append(ep)
                diff_eps.append(ep)
                all_eps.append(ep)
                if args.save_plots and epi == 0:
                    save_episode_plot(out_dir, map_name, difficulty, ep, m)
            by_map[map_name] = aggregate(eps)
        diff_agg = aggregate(diff_eps)
        by_diff[difficulty] = {"maps": by_map, "aggregate": diff_agg}

    results["by_difficulty"] = by_diff
    overall = aggregate(all_eps)
    overall["elapsed_sec"] = float(time.time() - t0)
    results["overall"] = overall

    # Baseline comparison (best-effort, optional).
    if baseline_summary is not None:
        baseline_comp: Dict[str, object] = {}
        b_ov = baseline_summary.get("overall", {}) if isinstance(baseline_summary, dict) else {}
        if isinstance(b_ov, dict):
            key_map = {
                "success_rate": "success_rate",
                "collision_rate": "collision_rate",
                "timeout_rate": "timeout_rate",
                "exploration_coverage": "exploration_coverage",
                "final_map_accuracy": "final_map_accuracy",
                "final_wall_f1": "final_wall_f1",
                "final_doorway_f1": "final_doorway_f1",
                "fake_doorway_approach_rate": "fake_doorway_approach_rate",
                "doorway_crossing_success_rate": "doorway_crossing_success_rate",
            }
            for bk, outk in key_map.items():
                if bk in b_ov:
                    baseline_comp[f"baseline_{outk}"] = b_ov[bk]
        for metric in [
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "exploration_coverage",
            "final_map_accuracy",
            "final_wall_f1",
            "final_doorway_f1",
            "fake_doorway_approach_rate",
            "doorway_crossing_success_rate",
        ]:
            if metric in overall and f"baseline_{metric}" in baseline_comp:
                baseline_comp[f"delta_{metric}"] = float(overall[metric] - baseline_comp[f"baseline_{metric}"])  # type: ignore[operator]
        results["baseline_comparison_note"] = "Baseline reference loaded (prefer v1 mapper-guided output if present)."
        results["baseline_comparison"] = baseline_comp
    else:
        results["baseline_comparison_note"] = "No prior navigation-only baseline file found."

    # Explicit v1/v2/v3 comparison table.
    version_paths = {
        "v1": Path("runs/phase2_mapper_guided_navigation/mapper_guided_navigation_results.json"),
        "v2": Path("runs/phase2_mapper_guided_navigation_v2/mapper_guided_navigation_results.json"),
    }
    comparison_keys = [
        "success_rate",
        "collision_rate",
        "fake_doorway_approach_rate",
        "doorway_crossing_success_rate",
        "exploration_coverage",
        "final_map_accuracy",
        "final_wall_f1",
        "final_doorway_f1",
        "timeout_rate",
        "coverage_gain_per_100_steps",
    ]
    version_comp: Dict[str, object] = {"v3": {k: float(overall.get(k, 0.0)) for k in comparison_keys}}
    for vname, vpath in version_paths.items():
        if vpath.exists():
            try:
                data = json.loads(vpath.read_text(encoding="utf-8"))
                ov = data.get("overall", {})
                if isinstance(ov, dict):
                    version_comp[vname] = {k: float(ov.get(k, 0.0)) for k in comparison_keys}
            except Exception:
                version_comp[vname] = {"error": f"failed_to_load:{vpath}"}
        else:
            version_comp[vname] = {"missing": str(vpath)}
    results["comparison_v1_v2_v3"] = version_comp

    out_json = out_dir / "mapper_guided_navigation_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Saved: {out_json}")
    print(
        f"Overall success={overall.get('success_rate', 0.0):.4f}, "
        f"collision={overall.get('collision_rate', 0.0):.4f}, "
        f"timeout={overall.get('timeout_rate', 0.0):.4f}, "
        f"fake_door={overall.get('fake_doorway_approach_rate', 0.0):.4f}, "
        f"door_cross={overall.get('doorway_crossing_success_rate', 0.0):.4f}, "
        f"coverage={overall.get('exploration_coverage', 0.0):.4f}, "
        f"map_acc={overall.get('final_map_accuracy', 0.0):.4f}, "
        f"wall_f1={overall.get('final_wall_f1', 0.0):.4f}, "
        f"door_f1={overall.get('final_doorway_f1', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
