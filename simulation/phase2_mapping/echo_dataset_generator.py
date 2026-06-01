from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover
    distance_transform_edt = None

try:
    from .mapping_utils import (
        DIFFICULTY_PRESETS,
        SECTOR_NAMES,
        SECTOR_OFFSETS_RAD,
        RAY_MAX_RANGE,
        apply_action,
        build_gt_grids,
        make_maps,
        point_in_obstacle,
        sample_free_pose,
        simulate_echo_observation,
        wrap_angle,
    )
except ImportError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.phase2_mapping.mapping_utils import (
        DIFFICULTY_PRESETS,
        SECTOR_NAMES,
        SECTOR_OFFSETS_RAD,
        RAY_MAX_RANGE,
        apply_action,
        build_gt_grids,
        make_maps,
        point_in_obstacle,
        sample_free_pose,
        simulate_echo_observation,
        wrap_angle,
    )


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
class MapCache:
    occupancy: np.ndarray
    wall: np.ndarray
    doorway: np.ndarray
    free: np.ndarray
    dist_to_wall: np.ndarray
    nx: int
    ny: int


@dataclass
class StepRecord:
    map_name: str
    difficulty: str
    episode_idx: int
    timestep: int
    action: str
    true_pose: Tuple[float, float, float]
    est_pose: Tuple[float, float, float]
    echo_timing: np.ndarray
    echo_intensity: np.ndarray
    scan_dirs_abs: np.ndarray
    near_structure_score: float
    near_doorway_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2B supervised echo mapping dataset generator.")
    parser.add_argument("--episodes-per-map", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--maps", type=str, default="empty_room,corridor,single_block,doorway,cluttered_room")
    parser.add_argument("--difficulties", type=str, default="clean,mild_noise,medium_noise,hard_noise")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--echo-bins", type=int, default=128)
    parser.add_argument("--samples-per-episode", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="datasets/phase2_echo_mapping")
    parser.add_argument("--save-sample-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--oversample-doorway-factor", type=float, default=2.0)
    parser.add_argument("--oversample-corridor-factor", type=float, default=1.5)
    parser.add_argument("--oversample-structure-factor", type=float, default=2.0)
    parser.add_argument("--sample-plot-count", type=int, default=24)
    return parser.parse_args()


def make_echo_bins(distance: float, intensity: float, n_bins: int, max_range: float) -> np.ndarray:
    vec = np.zeros(n_bins, dtype=np.float32)
    pos = int(np.clip(round((distance / max_range) * (n_bins - 1)), 0, n_bins - 1))
    for offset in range(-3, 4):
        idx = pos + offset
        if 0 <= idx < n_bins:
            vec[idx] += float(intensity * math.exp(-0.5 * (offset / 1.6) ** 2))
    return np.clip(vec, 0.0, 1.0)


def world_to_cell(x: float, y: float, cell_size: float, nx: int, ny: int) -> Tuple[int, int]:
    cx = int(np.clip(math.floor(x / cell_size), 0, nx - 1))
    cy = int(np.clip(math.floor(y / cell_size), 0, ny - 1))
    return cx, cy


def extract_patch(grid: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    half = size // 2
    out = np.zeros((size, size), dtype=np.float32)
    for py in range(size):
        for px in range(size):
            gx = cx + (px - half)
            gy = cy + (py - half)
            if 0 <= gx < grid.shape[1] and 0 <= gy < grid.shape[0]:
                out[py, px] = float(grid[gy, gx])
    return out


def build_map_cache(map_def, cell_size: float) -> MapCache:
    gt = build_gt_grids(map_def, cell_size)
    occ = gt["occupancy"].astype(np.uint8)
    wall = gt["wall"].astype(np.uint8)
    doorway = gt["doorway"].astype(np.uint8)
    free = (1 - occ).astype(np.uint8)

    if distance_transform_edt is not None:
        dist = distance_transform_edt(~occ.astype(bool)) * cell_size
        dist = dist.astype(np.float32)
    else:
        dist = np.zeros_like(occ, dtype=np.float32)

    ny, nx = occ.shape
    return MapCache(occupancy=occ, wall=wall, doorway=doorway, free=free, dist_to_wall=dist, nx=nx, ny=ny)


def compute_visibility_grid(
    map_def,
    x: float,
    y: float,
    cell_size: float,
    nx: int,
    ny: int,
    max_range: float,
    n_rays: int = 48,
) -> np.ndarray:
    vis = np.zeros((ny, nx), dtype=np.float32)
    step = max(0.04, cell_size * 0.75)
    for ridx in range(n_rays):
        ang = (2.0 * math.pi) * (ridx / n_rays)
        t = 0.0
        while t <= max_range:
            px = x + t * math.cos(ang)
            py = y + t * math.sin(ang)
            if px < 0.0 or py < 0.0 or px >= map_def.width or py >= map_def.height:
                break
            cx, cy = world_to_cell(px, py, cell_size, nx, ny)
            vis[cy, cx] = 1.0
            if point_in_obstacle(px, py, map_def, robot_radius=0.0):
                break
            t += step
    return vis


def choose_action(obs: Dict[str, Dict[str, float]], rng: np.random.Generator) -> str:
    front = float(obs["front"]["distance"])
    fl = float(obs["front_left"]["distance"])
    fr = float(obs["front_right"]["distance"])
    left = float(obs["left"]["distance"])
    right = float(obs["right"]["distance"])

    if front > 1.2 and fl > 0.9 and fr > 0.9:
        return "MOVE_FORWARD_SLOW" if rng.random() < 0.85 else "PROBE_FORWARD"
    if front > 0.55 and fl > 0.35 and fr > 0.35:
        r = rng.random()
        if r < 0.62:
            return "PROBE_FORWARD"
        if r < 0.82:
            return "MOVE_FORWARD_SLOW"
        return "TURN_LEFT" if left >= right else "TURN_RIGHT"

    if left > right + 0.1:
        return "TURN_LEFT"
    if right > left + 0.1:
        return "TURN_RIGHT"
    return "SLOW_DOWN_AND_RESAMPLE" if rng.random() < 0.35 else ("TURN_LEFT" if rng.random() < 0.5 else "TURN_RIGHT")


def safe_apply_action(map_def, x: float, y: float, heading: float, action: str):
    nx, ny, nh, moved, collided = apply_action(
        x,
        y,
        heading,
        action,
        map_def,
        robot_radius=0.12,
        turn_deg=15.0,
    )
    if not collided:
        return nx, ny, nh, moved, action

    for fallback in ("TURN_LEFT", "TURN_RIGHT", "SLOW_DOWN_AND_RESAMPLE"):
        fx, fy, fh, fm, fcol = apply_action(
            x,
            y,
            heading,
            fallback,
            map_def,
            robot_radius=0.12,
            turn_deg=15.0,
        )
        if not fcol:
            return fx, fy, fh, fm, fallback

    return x, y, heading, 0.0, "STOP_OR_REVERSE"


def structure_scores(map_name: str, obs: Dict[str, Dict[str, float]], x: float, y: float) -> Tuple[float, float]:
    dists = np.array([float(obs[s]["distance"]) for s in SECTOR_NAMES], dtype=np.float32)
    near_structure = float(np.clip((0.95 - float(dists.min())) / 0.95, 0.0, 1.0))

    near_door = 0.0
    if map_name == "doorway":
        near_x = max(0.0, 1.0 - abs(x - 6.0) / 1.3)
        near_y = max(0.0, 1.0 - abs(y - 4.0) / 1.8)
        near_door = float(np.clip(near_x * near_y, 0.0, 1.0))
    return near_structure, near_door


def sample_episode_records(
    records: Sequence[StepRecord],
    samples_per_episode: int,
    map_name: str,
    rng: np.random.Generator,
    oversample_doorway_factor: float,
    oversample_corridor_factor: float,
    oversample_structure_factor: float,
) -> List[StepRecord]:
    if not records:
        return []

    n = min(samples_per_episode, len(records))
    weights = np.ones(len(records), dtype=np.float64)

    if map_name == "doorway":
        for i, r in enumerate(records):
            weights[i] *= 1.0 + (oversample_doorway_factor - 1.0) * r.near_doorway_score
    if map_name == "corridor":
        for i, r in enumerate(records):
            weights[i] *= 1.0 + (oversample_corridor_factor - 1.0) * max(0.25, r.near_structure_score)

    for i, r in enumerate(records):
        weights[i] *= 1.0 + (oversample_structure_factor - 1.0) * r.near_structure_score

    weights = np.clip(weights, 1e-6, None)
    weights /= weights.sum()
    idx = rng.choice(len(records), size=n, replace=False, p=weights)
    idx.sort()
    return [records[i] for i in idx]


def save_sample_plot(
    out_dir: Path,
    plot_idx: int,
    sample: Dict[str, np.ndarray],
    global_occ: np.ndarray,
    map_name: str,
    true_pose: np.ndarray,
) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=130)
    axes = axes.flatten()

    axes[0].imshow(sample["occupancy_patch"], cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[0].set_title("GT Occupancy Patch")
    axes[1].imshow(sample["wall_patch"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("GT Wall Patch")
    axes[2].imshow(sample["doorway_patch"], cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("GT Doorway Patch")
    axes[3].imshow(sample["free_patch"], cmap="Blues", vmin=0.0, vmax=1.0)
    axes[3].set_title("GT Free-Space Patch")

    xvals = np.arange(sample["echo_timing"].shape[0])
    axes[4].plot(xvals, sample["echo_timing"], marker="o", label="timing (m)")
    axes[4].plot(xvals, sample["echo_intensity"], marker="s", label="intensity")
    axes[4].set_title("Echo Timing / Intensity")
    axes[4].set_xticks(xvals)
    axes[4].set_xticklabels(SECTOR_NAMES, rotation=20, fontsize=8)
    axes[4].legend(fontsize=7)

    axes[5].imshow(global_occ, cmap="gray_r", vmin=0.0, vmax=1.0)
    gx = int(np.clip(round(true_pose[0]), 0, global_occ.shape[1] - 1))
    gy = int(np.clip(round(true_pose[1]), 0, global_occ.shape[0] - 1))
    axes[5].scatter([gx], [gy], c="lime", s=18)
    axes[5].set_title(f"Global Map + Pose ({map_name})")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / f"sample_{plot_idx:04d}_{map_name}.png")
    plt.close(fig)


def build_split_dict(indices: np.ndarray, full: Dict[str, np.ndarray], shared: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in full.items():
        out[k] = v[indices]
    for k, v in shared.items():
        out[k] = v
    return out


def main() -> None:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "sample_visualizations"
    if args.save_sample_plots:
        vis_dir.mkdir(parents=True, exist_ok=True)

    maps_all = {m.name: m for m in make_maps()}
    map_names = [m.strip() for m in args.maps.split(",") if m.strip()]
    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]

    for name in map_names:
        if name not in maps_all:
            raise ValueError(f"Unknown map: {name}")
    for d in difficulties:
        if d not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {d}")

    map_caches: Dict[str, MapCache] = {}
    map_idx_lookup = {name: idx for idx, name in enumerate(map_names)}
    diff_idx_lookup = {name: idx for idx, name in enumerate(difficulties)}

    padded_occ_refs: List[np.ndarray] = []
    grid_shapes: List[Tuple[int, int]] = []

    for name in map_names:
        cache = build_map_cache(maps_all[name], args.cell_size)
        map_caches[name] = cache
        grid_shapes.append((cache.ny, cache.nx))

    max_ny = max(s[0] for s in grid_shapes)
    max_nx = max(s[1] for s in grid_shapes)

    for name in map_names:
        occ = map_caches[name].occupancy.astype(np.float32)
        padded = np.zeros((max_ny, max_nx), dtype=np.float32)
        padded[: occ.shape[0], : occ.shape[1]] = occ
        padded_occ_refs.append(padded)

    global_occupancy_refs = np.stack(padded_occ_refs, axis=0)

    records_selected: List[StepRecord] = []

    for d in difficulties:
        preset = DIFFICULTY_PRESETS[d]
        for map_name in map_names:
            m = maps_all[map_name]
            for ep in range(args.episodes_per_map):
                x, y = sample_free_pose(m, rng, robot_radius=0.12)
                heading = float(rng.uniform(-math.pi, math.pi))
                est_x, est_y, est_h = x, y, heading

                episode_records: List[StepRecord] = []

                for t in range(args.max_steps):
                    obs = simulate_echo_observation(m, x, y, heading, rng, preset)
                    action = choose_action(obs, rng)

                    timing = np.array([float(obs[s]["distance"]) for s in SECTOR_NAMES], dtype=np.float32)
                    intensity = np.array([float(obs[s]["intensity"]) for s in SECTOR_NAMES], dtype=np.float32)
                    scan_dirs_abs = np.array([wrap_angle(heading + SECTOR_OFFSETS_RAD[s]) for s in SECTOR_NAMES], dtype=np.float32)
                    structure_score, doorway_score = structure_scores(map_name, obs, x, y)

                    episode_records.append(
                        StepRecord(
                            map_name=map_name,
                            difficulty=d,
                            episode_idx=ep,
                            timestep=t,
                            action=action,
                            true_pose=(float(x), float(y), float(heading)),
                            est_pose=(float(est_x), float(est_y), float(est_h)),
                            echo_timing=timing,
                            echo_intensity=intensity,
                            scan_dirs_abs=scan_dirs_abs,
                            near_structure_score=structure_score,
                            near_doorway_score=doorway_score,
                        )
                    )

                    nx, ny, nh, moved, applied_action = safe_apply_action(m, x, y, heading, action)
                    x, y, heading = nx, ny, nh

                    if applied_action in ("MOVE_FORWARD_SLOW", "PROBE_FORWARD"):
                        est_x += moved * math.cos(est_h)
                        est_y += moved * math.sin(est_h)
                    elif applied_action == "TURN_LEFT":
                        est_h += math.radians(15.0)
                    elif applied_action == "TURN_RIGHT":
                        est_h -= math.radians(15.0)

                    est_x += float(rng.normal(0.0, preset.get("pose_drift_std", 0.0)))
                    est_y += float(rng.normal(0.0, preset.get("pose_drift_std", 0.0)))
                    est_h = wrap_angle(est_h + float(rng.normal(0.0, 0.25 * preset.get("pose_drift_std", 0.0))))

                picked = sample_episode_records(
                    records=episode_records,
                    samples_per_episode=args.samples_per_episode,
                    map_name=map_name,
                    rng=rng,
                    oversample_doorway_factor=args.oversample_doorway_factor,
                    oversample_corridor_factor=args.oversample_corridor_factor,
                    oversample_structure_factor=args.oversample_structure_factor,
                )
                records_selected.extend(picked)

    n = len(records_selected)
    if n == 0:
        raise RuntimeError("No samples were generated.")

    echo_timing = np.zeros((n, len(SECTOR_NAMES)), dtype=np.float32)
    echo_intensity = np.zeros((n, len(SECTOR_NAMES)), dtype=np.float32)
    echo_multichannel = np.zeros((n, len(SECTOR_NAMES), args.echo_bins), dtype=np.float32)
    true_pose = np.zeros((n, 3), dtype=np.float32)
    est_pose = np.zeros((n, 3), dtype=np.float32)
    pose_correction = np.zeros((n, 3), dtype=np.float32)
    map_idx = np.zeros((n,), dtype=np.int32)
    diff_idx = np.zeros((n,), dtype=np.int32)
    timestep = np.zeros((n,), dtype=np.int32)
    action_idx = np.zeros((n,), dtype=np.int32)
    scan_dirs = np.zeros((n, len(SECTOR_NAMES)), dtype=np.float32)

    occupancy_patch = np.zeros((n, args.patch_size, args.patch_size), dtype=np.float32)
    wall_patch = np.zeros((n, args.patch_size, args.patch_size), dtype=np.float32)
    doorway_patch = np.zeros((n, args.patch_size, args.patch_size), dtype=np.float32)
    free_patch = np.zeros((n, args.patch_size, args.patch_size), dtype=np.float32)
    visibility_patch = np.zeros((n, args.patch_size, args.patch_size), dtype=np.float32)
    distance_to_nearest_wall = np.zeros((n,), dtype=np.float32)

    sample_map_names: List[str] = []
    sample_difficulties: List[str] = []

    for i, rec in enumerate(records_selected):
        cache = map_caches[rec.map_name]
        m = maps_all[rec.map_name]

        x, y, th = rec.true_pose
        ex, ey, eh = rec.est_pose
        cx, cy = world_to_cell(x, y, args.cell_size, cache.nx, cache.ny)

        vis_global = compute_visibility_grid(
            map_def=m,
            x=x,
            y=y,
            cell_size=args.cell_size,
            nx=cache.nx,
            ny=cache.ny,
            max_range=RAY_MAX_RANGE,
            n_rays=48,
        )

        echo_timing[i] = rec.echo_timing
        echo_intensity[i] = rec.echo_intensity
        for s_idx in range(len(SECTOR_NAMES)):
            echo_multichannel[i, s_idx] = make_echo_bins(
                distance=float(rec.echo_timing[s_idx]),
                intensity=float(rec.echo_intensity[s_idx]),
                n_bins=args.echo_bins,
                max_range=RAY_MAX_RANGE,
            )

        true_pose[i] = np.array([x, y, th], dtype=np.float32)
        est_pose[i] = np.array([ex, ey, eh], dtype=np.float32)
        pose_correction[i] = np.array([x - ex, y - ey, wrap_angle(th - eh)], dtype=np.float32)
        map_idx[i] = map_idx_lookup[rec.map_name]
        diff_idx[i] = diff_idx_lookup[rec.difficulty]
        timestep[i] = rec.timestep
        action_idx[i] = ACTION_TO_IDX[rec.action]
        scan_dirs[i] = rec.scan_dirs_abs

        occupancy_patch[i] = extract_patch(cache.occupancy.astype(np.float32), cx, cy, args.patch_size)
        wall_patch[i] = extract_patch(cache.wall.astype(np.float32), cx, cy, args.patch_size)
        doorway_patch[i] = extract_patch(cache.doorway.astype(np.float32), cx, cy, args.patch_size)
        free_patch[i] = extract_patch(cache.free.astype(np.float32), cx, cy, args.patch_size)
        visibility_patch[i] = extract_patch(vis_global.astype(np.float32), cx, cy, args.patch_size)
        distance_to_nearest_wall[i] = float(cache.dist_to_wall[cy, cx])

        sample_map_names.append(rec.map_name)
        sample_difficulties.append(rec.difficulty)

    idx_all = np.arange(n)
    rng.shuffle(idx_all)

    train_end = int(round(args.train_ratio * n))
    val_end = train_end + int(round(args.val_ratio * n))
    train_idx = idx_all[:train_end]
    val_idx = idx_all[train_end:val_end]
    test_idx = idx_all[val_end:]

    full_arrays = {
        "echo_timing_vector": echo_timing,
        "echo_intensity_vector": echo_intensity,
        "echo_multichannel": echo_multichannel,
        "true_pose": true_pose,
        "estimated_pose": est_pose,
        "pose_correction_target": pose_correction,
        "map_index": map_idx,
        "difficulty_index": diff_idx,
        "timestep_index": timestep,
        "action_index": action_idx,
        "scan_directions_rad": scan_dirs,
        "occupancy_patch": occupancy_patch,
        "wall_patch": wall_patch,
        "doorway_patch": doorway_patch,
        "free_space_patch": free_patch,
        "visibility_mask_patch": visibility_patch,
        "distance_to_nearest_wall": distance_to_nearest_wall,
    }

    shared_arrays = {
        "map_names": np.array(map_names, dtype="U32"),
        "difficulty_names": np.array(difficulties, dtype="U32"),
        "action_names": np.array(ACTION_NAMES, dtype="U32"),
        "global_occupancy_grids": global_occupancy_refs.astype(np.float32),
        "global_grid_shapes": np.array(grid_shapes, dtype=np.int32),
        "cell_size": np.array([args.cell_size], dtype=np.float32),
        "grid_size_hint": np.array([args.grid_size], dtype=np.int32),
        "patch_size": np.array([args.patch_size], dtype=np.int32),
    }

    np.savez_compressed(out_dir / "train.npz", **build_split_dict(train_idx, full_arrays, shared_arrays))
    np.savez_compressed(out_dir / "val.npz", **build_split_dict(val_idx, full_arrays, shared_arrays))
    np.savez_compressed(out_dir / "test.npz", **build_split_dict(test_idx, full_arrays, shared_arrays))

    map_counts = {name: int(np.sum(map_idx == i)) for name, i in map_idx_lookup.items()}
    diff_counts = {name: int(np.sum(diff_idx == i)) for name, i in diff_idx_lookup.items()}

    occupancy_positive_ratio = float(occupancy_patch.mean())
    wall_positive_ratio = float(wall_patch.mean())
    doorway_positive_ratio = float(doorway_patch.mean())
    free_space_ratio = float(free_patch.mean())

    pose_range = {
        "x_min": float(true_pose[:, 0].min()),
        "x_max": float(true_pose[:, 0].max()),
        "y_min": float(true_pose[:, 1].min()),
        "y_max": float(true_pose[:, 1].max()),
        "theta_min": float(true_pose[:, 2].min()),
        "theta_max": float(true_pose[:, 2].max()),
    }

    manifest = {
        "phase": "Phase 2B dataset generation (ground-truth labels)",
        "total_samples": int(n),
        "train_count": int(train_idx.size),
        "val_count": int(val_idx.size),
        "test_count": int(test_idx.size),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "episodes_per_map": int(args.episodes_per_map),
        "max_steps": int(args.max_steps),
        "samples_per_episode": int(args.samples_per_episode),
        "maps": map_names,
        "difficulties": difficulties,
        "samples_per_map": map_counts,
        "samples_per_difficulty": diff_counts,
        "occupancy_positive_ratio": occupancy_positive_ratio,
        "wall_positive_ratio": wall_positive_ratio,
        "doorway_positive_ratio": doorway_positive_ratio,
        "free_space_ratio": free_space_ratio,
        "mean_echo_distance": float(echo_timing.mean()),
        "mean_echo_intensity": float(echo_intensity.mean()),
        "pose_range": pose_range,
        "patch_size": int(args.patch_size),
        "cell_size": float(args.cell_size),
        "grid_size_hint": int(args.grid_size),
        "corridor_representation": {
            "sample_count": int(map_counts.get("corridor", 0)),
            "ratio": float(map_counts.get("corridor", 0) / max(1, n)),
        },
        "doorway_representation": {
            "sample_count": int(map_counts.get("doorway", 0)),
            "ratio": float(map_counts.get("doorway", 0) / max(1, n)),
        },
        "balancing": {
            "oversample_doorway_factor": float(args.oversample_doorway_factor),
            "oversample_corridor_factor": float(args.oversample_corridor_factor),
            "oversample_structure_factor": float(args.oversample_structure_factor),
        },
        "files": {
            "train": str((out_dir / "train.npz").as_posix()),
            "val": str((out_dir / "val.npz").as_posix()),
            "test": str((out_dir / "test.npz").as_posix()),
            "manifest": str((out_dir / "dataset_manifest.json").as_posix()),
        },
    }

    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.save_sample_plots and plt is not None:
        plot_idx = np.arange(n)
        rng.shuffle(plot_idx)
        plot_idx = plot_idx[: min(args.sample_plot_count, n)]
        for pidx, i in enumerate(plot_idx):
            sample = {
                "occupancy_patch": occupancy_patch[i],
                "wall_patch": wall_patch[i],
                "doorway_patch": doorway_patch[i],
                "free_patch": free_patch[i],
                "echo_timing": echo_timing[i],
                "echo_intensity": echo_intensity[i],
            }
            mname = map_names[int(map_idx[i])]
            gy, gx = grid_shapes[int(map_idx[i])]
            glob = global_occupancy_refs[int(map_idx[i]), :gy, :gx]
            tp = true_pose[i].copy()
            tp[0] = tp[0] / args.cell_size
            tp[1] = tp[1] / args.cell_size
            save_sample_plot(vis_dir, pidx, sample, glob, mname, tp)

    print("Phase 2B dataset generation complete")
    print(f"Total samples: {n}")
    print(f"Split counts: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}")
    print(f"Samples per map: {map_counts}")
    print(f"Samples per difficulty: {diff_counts}")
    print(f"Doorway ratio: {manifest['doorway_representation']['ratio']:.4f}")
    print(f"Wall positive ratio: {wall_positive_ratio:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
