from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.phase2_mapping.mapping_utils import (  # noqa: E402
    DIFFICULTY_PRESETS,
    SECTOR_NAMES,
    apply_action,
    build_gt_grids,
    compute_wall_reconstruction_error,
    doorway_precision_recall,
    make_maps,
    point_in_obstacle,
    predict_collision,
    sample_free_pose,
    simulate_echo_observation,
)

try:
    from scipy.ndimage import label as ndi_label
except Exception:  # pragma: no cover
    ndi_label = None


ACTION_FAST = "MOVE_FORWARD_FAST"
ACTION_SLOW = "MOVE_FORWARD_SLOW"
ACTION_PROBE = "PROBE_FORWARD"
ACTION_LEFT = "TURN_LEFT"
ACTION_RIGHT = "TURN_RIGHT"
ACTION_RESAMPLE = "SLOW_DOWN_AND_RESAMPLE"
ACTION_STOP = "STOP_OR_REVERSE"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _grid_shape(width: float, height: float, cell_size: float) -> Tuple[int, int]:
    nx = int(math.ceil(width / cell_size))
    ny = int(math.ceil(height / cell_size))
    return ny, nx


def _world_to_cell(x: float, y: float, cell_size: float, nx: int, ny: int) -> Tuple[int, int]:
    cx = int(np.clip(math.floor(x / cell_size), 0, nx - 1))
    cy = int(np.clip(math.floor(y / cell_size), 0, ny - 1))
    return cx, cy


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int32)
    acc = np.zeros_like(m, dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            src = np.zeros_like(m)
            ys = slice(max(0, dy), m.shape[0] + min(0, dy))
            xs = slice(max(0, dx), m.shape[1] + min(0, dx))
            y2 = slice(max(0, -dy), m.shape[0] + min(0, -dy))
            x2 = slice(max(0, -dx), m.shape[1] + min(0, -dx))
            src[ys, xs] = m[y2, x2]
            acc += src
    return acc


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    if ndi_label is not None:
        return ndi_label(mask.astype(np.uint8))
    # fallback BFS label
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    cid = 0
    for y in range(ny):
        for x in range(nx):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            cid += 1
            q = deque([(x, y)])
            labels[y, x] = cid
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxp, nyp = cx + dx, cy + dy
                    if 0 <= nxp < nx and 0 <= nyp < ny and mask[nyp, nxp] and labels[nyp, nxp] == 0:
                        labels[nyp, nxp] = cid
                        q.append((nxp, nyp))
    return labels, cid


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, num = _label_components(mask)
    if num == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(mask, dtype=bool)
    for cid in range(1, len(counts)):
        if counts[cid] >= min_area:
            keep |= labels == cid
    return keep


def _extract_wall_segments(wall_mask: np.ndarray, min_len: int = 3) -> List[Dict[str, object]]:
    ny, nx = wall_mask.shape
    segs: List[Dict[str, object]] = []
    # Horizontal
    for y in range(ny):
        x = 0
        while x < nx:
            if wall_mask[y, x] == 0:
                x += 1
                continue
            x0 = x
            while x < nx and wall_mask[y, x] == 1:
                x += 1
            x1 = x - 1
            ln = x1 - x0 + 1
            if ln >= min_len:
                segs.append({"ori": "H", "x0": x0, "y0": y, "x1": x1, "y1": y, "length": ln})
    # Vertical
    for x in range(nx):
        y = 0
        while y < ny:
            if wall_mask[y, x] == 0:
                y += 1
                continue
            y0 = y
            while y < ny and wall_mask[y, x] == 1:
                y += 1
            y1 = y - 1
            ln = y1 - y0 + 1
            if ln >= min_len:
                segs.append({"ori": "V", "x0": x, "y0": y0, "x1": x, "y1": y1, "length": ln})
    return segs


def _detect_doorway_candidates(
    wall_segments: List[Dict[str, object]],
    occ_prob: np.ndarray,
    free_obs: np.ndarray,
    wall_prob: np.ndarray,
    gt_door: np.ndarray,
    cell_size: float,
) -> Tuple[np.ndarray, Dict[str, float], List[Tuple[int, int]]]:
    ny, nx = occ_prob.shape
    doorway_prob = np.zeros((ny, nx), dtype=np.float32)
    candidates = 0
    accepted = 0
    rejected = 0
    false_doors = 0
    accepted_centers: List[Tuple[int, int]] = []

    min_gap_cells = max(2, int(round(0.6 / cell_size)))
    max_gap_cells = max(min_gap_cells + 1, int(round(1.6 / cell_size)))

    # compare segment pairs of same orientation
    for i in range(len(wall_segments)):
        a = wall_segments[i]
        for j in range(i + 1, len(wall_segments)):
            b = wall_segments[j]
            if a["ori"] != b["ori"]:
                continue
            if a["ori"] == "H":
                if abs(int(a["y0"]) - int(b["y0"])) > 1:
                    continue
                y = int(round((int(a["y0"]) + int(b["y0"])) / 2))
                left, right = (a, b) if int(a["x0"]) <= int(b["x0"]) else (b, a)
                gap = int(right["x0"]) - int(left["x1"]) - 1
                if gap < min_gap_cells or gap > max_gap_cells:
                    continue
                x0 = int(left["x1"]) + 1
                x1 = int(right["x0"]) - 1
                candidates += 1
                gap_occ = float(occ_prob[max(0, y - 1): min(ny, y + 2), max(0, x0): min(nx, x1 + 1)].mean())
                gap_free = float(free_obs[max(0, y - 1): min(ny, y + 2), max(0, x0): min(nx, x1 + 1)].mean())
                side_conf = min(float(wall_prob[y, int(left["x1"])]), float(wall_prob[y, int(right["x0"])]))
                if gap_occ < 0.35 and gap_free >= 2.0 and side_conf > 0.50:
                    cx = int(round((x0 + x1) / 2))
                    cy = y
                    for yy in range(max(0, cy - 1), min(ny, cy + 2)):
                        for xx in range(max(0, cx - 1), min(nx, cx + 2)):
                            doorway_prob[yy, xx] += float(np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / 1.2)))
                    accepted += 1
                    accepted_centers.append((cx, cy))
                    if gt_door[cy, cx] == 0:
                        false_doors += 1
                else:
                    rejected += 1
            else:
                if abs(int(a["x0"]) - int(b["x0"])) > 1:
                    continue
                x = int(round((int(a["x0"]) + int(b["x0"])) / 2))
                up, down = (a, b) if int(a["y0"]) <= int(b["y0"]) else (b, a)
                gap = int(down["y0"]) - int(up["y1"]) - 1
                if gap < min_gap_cells or gap > max_gap_cells:
                    continue
                y0 = int(up["y1"]) + 1
                y1 = int(down["y0"]) - 1
                candidates += 1
                gap_occ = float(occ_prob[max(0, y0): min(ny, y1 + 1), max(0, x - 1): min(nx, x + 2)].mean())
                gap_free = float(free_obs[max(0, y0): min(ny, y1 + 1), max(0, x - 1): min(nx, x + 2)].mean())
                side_conf = min(float(wall_prob[int(up["y1"]), x]), float(wall_prob[int(down["y0"]), x]))
                if gap_occ < 0.35 and gap_free >= 2.0 and side_conf > 0.50:
                    cx = x
                    cy = int(round((y0 + y1) / 2))
                    for yy in range(max(0, cy - 1), min(ny, cy + 2)):
                        for xx in range(max(0, cx - 1), min(nx, cx + 2)):
                            doorway_prob[yy, xx] += float(np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / 1.2)))
                    accepted += 1
                    accepted_centers.append((cx, cy))
                    if gt_door[cy, cx] == 0:
                        false_doors += 1
                else:
                    rejected += 1

    doorway_prob = np.clip(doorway_prob, 0.0, 1.0)
    diag = {
        "doorway_candidate_count": float(candidates),
        "accepted_doorway_candidate_count": float(accepted),
        "rejected_doorway_candidate_count": float(rejected),
        "false_doorway_count": float(false_doors),
    }
    return doorway_prob, diag, accepted_centers


def _build_frontier_map(observed_mask: np.ndarray, occ_prob: np.ndarray) -> np.ndarray:
    free_known = (observed_mask > 0) & (occ_prob < 0.4)
    unknown = observed_mask <= 0
    frontier = np.zeros_like(free_known, dtype=bool)
    ny, nx = frontier.shape
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if not free_known[y, x]:
                continue
            if unknown[y - 1:y + 2, x - 1:x + 2].any():
                frontier[y, x] = True
    return frontier


def _choose_mapping_action(
    m,
    x: float,
    y: float,
    heading: float,
    obs: Dict[str, Dict[str, float]],
    robot_radius: float,
    safety_margin: float,
    occ_prob: np.ndarray,
    observed_count: np.ndarray,
    cell_size: float,
    no_new_info_steps: int,
    last_frontier_dir: Optional[str],
) -> Tuple[str, Optional[str], bool, bool]:
    front = float(obs["front"]["distance"])
    fl = float(obs["front_left"]["distance"])
    fr = float(obs["front_right"]["distance"])
    left = float(obs["left"]["distance"])
    right = float(obs["right"]["distance"])

    def safe_forward(dist: float) -> bool:
        return (
            front > dist
            and fl > 0.12
            and fr > 0.12
            and (not predict_collision(m, x, y, heading, dist, robot_radius, safety_margin))
        )

    frontier_switch = False
    used_recovery = False
    # baseline safe move
    if safe_forward(0.08):
        action = ACTION_SLOW
    elif safe_forward(0.03):
        action = ACTION_PROBE
    else:
        action = ACTION_LEFT if left >= right else ACTION_RIGHT

    # local loop/stagnation handling for clutter maps
    if m.name == "cluttered_room" and no_new_info_steps >= 12:
        used_recovery = True
        # controlled recovery: turn to explore frontier direction, occasional resample.
        ny, nx = occ_prob.shape
        cx, cy = _world_to_cell(x, y, cell_size, nx, ny)
        frontier = _build_frontier_map((observed_count > 0).astype(np.uint8), occ_prob)
        if frontier.any():
            ys, xs = np.where(frontier)
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            idx = int(np.argmin(d2))
            tx, ty = int(xs[idx]), int(ys[idx])
            dx = tx - cx
            action_dir = "LEFT" if dx < 0 else "RIGHT"
            if last_frontier_dir is not None and action_dir != last_frontier_dir:
                frontier_switch = True
            last_frontier_dir = action_dir
            action = ACTION_LEFT if action_dir == "LEFT" else ACTION_RIGHT
            if no_new_info_steps % 4 == 0:
                action = ACTION_RESAMPLE
        else:
            action = ACTION_RESAMPLE if no_new_info_steps % 3 == 0 else (ACTION_LEFT if left >= right else ACTION_RIGHT)
    return action, last_frontier_dir, frontier_switch, used_recovery


def _save_overlay_plot(
    out_dir: Path,
    map_name: str,
    difficulty: str,
    ep: Dict[str, object],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=130)
    axes = axes.flatten()
    axes[0].imshow(ep["gt_occupancy"], cmap="gray_r")
    axes[0].set_title("GT Occupancy")
    axes[1].imshow(ep["pred_occupancy"], cmap="gray_r")
    axes[1].set_title("Pred Occupancy")
    axes[2].imshow(ep["wall_probability"], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Wall Probability")
    axes[3].imshow(ep["doorway_probability"], cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("Doorway Probability")
    axes[4].imshow(ep["confidence"], cmap="Blues", vmin=0, vmax=1)
    axes[4].set_title("Confidence")
    # wall segments overlay
    axes[5].imshow(ep["pred_occupancy"], cmap="gray_r", alpha=0.5)
    for s in ep["wall_segments"]:
        axes[5].plot([s["x0"], s["x1"]], [s["y0"], s["y1"]], "r-", linewidth=1.0)
    axes[5].set_title("Wall Segments")
    # doorway candidates
    axes[6].imshow(ep["doorway_probability"], cmap="viridis", vmin=0, vmax=1)
    for cx, cy in ep["accepted_doorway_centers"]:
        axes[6].plot(cx, cy, "rx", markersize=6)
    axes[6].set_title("Accepted Doorway Cands")
    # trajectories
    tpath = np.asarray(ep["true_path"], dtype=float)
    epath = np.asarray(ep["est_path"], dtype=float)
    axes[7].plot(tpath[:, 0], tpath[:, 1], label="true")
    axes[7].plot(epath[:, 0], epath[:, 1], label="estimated")
    axes[7].set_title("True vs Est Trajectory")
    axes[7].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"{map_name}_{difficulty}_overlay.png")
    plt.close(fig)


def run_single_episode(
    m,
    difficulty: str,
    preset: Dict[str, float],
    max_steps: int,
    cell_size: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    robot_radius = 0.12
    safety_margin = 0.05 if m.name != "cluttered_room" else 0.08
    ny, nx = _grid_shape(m.width, m.height, cell_size)
    gt = build_gt_grids(m, cell_size)
    gt_occ = gt["occupancy"].astype(np.uint8)
    gt_wall = gt["wall"].astype(np.uint8)
    gt_door = gt["doorway"].astype(np.uint8)

    occ_logodds = np.zeros((ny, nx), dtype=np.float32)
    free_obs = np.zeros((ny, nx), dtype=np.float32)
    occ_obs = np.zeros((ny, nx), dtype=np.float32)
    total_obs = np.zeros((ny, nx), dtype=np.float32)

    x, y = sample_free_pose(m, rng, robot_radius=robot_radius)
    heading = float(rng.uniform(-math.pi, math.pi))
    est_x, est_y, est_heading = x, y, heading

    action_counts = Counter()
    true_path = [(x, y)]
    est_path = [(est_x, est_y)]
    loc_err = []
    collisions = 0
    timeout_flag = 1
    no_new_info_steps = 0
    last_frontier_dir = None
    frontier_switch_count = 0
    clutter_recovery_count = 0
    loop_stagnation_count = 0

    sector_offsets = {
        "left": math.radians(90.0),
        "front_left": math.radians(40.0),
        "front": 0.0,
        "front_right": math.radians(-40.0),
        "right": math.radians(-90.0),
    }

    for _ in range(max_steps):
        obs = simulate_echo_observation(m, x, y, heading, rng, preset)
        obs_before = float((total_obs > 0).sum())

        # inverse sensor update
        for sec in SECTOR_NAMES:
            ang = heading + sector_offsets[sec]
            d = float(obs[sec]["distance"])
            intensity = float(obs[sec]["intensity"])
            step = max(0.04, cell_size * 0.6)
            t = 0.0
            while t < max(0.0, d - step):
                px = est_x + t * math.cos(ang)
                py = est_y + t * math.sin(ang)
                cx, cy = _world_to_cell(px, py, cell_size, nx, ny)
                free_obs[cy, cx] += 1.0
                total_obs[cy, cx] += 1.0
                occ_logodds[cy, cx] -= 0.08
                t += step
            if d < 2.95:
                px = est_x + d * math.cos(ang)
                py = est_y + d * math.sin(ang)
                cx, cy = _world_to_cell(px, py, cell_size, nx, ny)
                occ_obs[cy, cx] += (0.8 + 0.6 * intensity)
                total_obs[cy, cx] += 1.0
                occ_logodds[cy, cx] += (0.30 + 0.25 * intensity)

        occ_prob_raw = _sigmoid(occ_logodds)
        occ_ratio = occ_obs / np.maximum(1e-6, (occ_obs + free_obs))

        # occupancy cleanup
        occ_mask = ((occ_ratio > 0.58) & (occ_obs >= 2.0)) | (occ_prob_raw > 0.68)
        neigh = _neighbor_count(occ_mask.astype(np.uint8))
        occ_mask = occ_mask & (neigh >= 2)
        occ_mask = _remove_small_components(occ_mask, min_area=4)

        # suppress weak border wall hallucinations
        border = np.zeros_like(occ_mask, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        occ_mask[border & (occ_obs < 4.0)] = False

        occ_pred = occ_mask.astype(np.uint8)

        # wall probability refinement from stable occupied + neighborhood alignment
        stable_occ = occ_mask & (occ_obs >= 2.0)
        free_neighbors = _neighbor_count((~occ_mask).astype(np.uint8))
        wall_candidate = stable_occ & (free_neighbors >= 1)
        support = _neighbor_count(wall_candidate.astype(np.uint8)).astype(np.float32) / 8.0
        wall_probability = np.clip(
            0.45 * occ_ratio + 0.35 * support + 0.20 * np.clip(total_obs / 8.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        wall_probability[~wall_candidate] *= 0.3
        wall_probability[(support < 0.25) & (wall_probability < 0.6)] = 0.0
        wall_mask = (wall_probability >= 0.45).astype(np.uint8)
        wall_mask = _remove_small_components(wall_mask.astype(bool), min_area=3).astype(np.uint8)

        wall_segments = _extract_wall_segments(wall_mask, min_len=3)
        mean_seg_len = float(np.mean([s["length"] for s in wall_segments])) if wall_segments else 0.0

        # doorway probability via geometric wall-gap-wall candidates
        doorway_prob, door_diag, accepted_centers = _detect_doorway_candidates(
            wall_segments, occ_prob_raw, free_obs, wall_probability, gt_door, cell_size
        )
        door_pred = (doorway_prob >= 0.35).astype(np.uint8)

        # confidence: high with repeated consistent observations, low in unknown/conflict
        contradiction = np.minimum(occ_obs, free_obs) / np.maximum(1.0, total_obs)
        agreement = np.abs(occ_obs - free_obs) / np.maximum(1.0, total_obs)
        confidence = np.clip(np.clip(total_obs / 8.0, 0.0, 1.0) * agreement * (1.0 - 0.7 * contradiction), 0.0, 1.0)

        observed_count = total_obs.copy()
        obs_after = float((observed_count > 0).sum())
        if obs_after <= obs_before + 0.5:
            no_new_info_steps += 1
        else:
            no_new_info_steps = 0
        if no_new_info_steps >= 12:
            loop_stagnation_count += 1

        action, last_frontier_dir, frontier_switched, used_recovery = _choose_mapping_action(
            m,
            x,
            y,
            heading,
            obs,
            robot_radius,
            safety_margin,
            occ_prob_raw,
            observed_count,
            cell_size,
            no_new_info_steps,
            last_frontier_dir,
        )
        if frontier_switched:
            frontier_switch_count += 1
        if used_recovery:
            clutter_recovery_count += 1

        action_counts[action] += 1
        nx_t, ny_t, nh_t, moved, collided = apply_action(x, y, heading, action, m, robot_radius=robot_radius, turn_deg=15.0)
        if collided:
            collisions += 1
            timeout_flag = 0
            break

        x, y, heading = nx_t, ny_t, nh_t
        # dead-reckoning estimated pose (with drift)
        drift = preset["pose_drift_std"]
        est_x = est_x + moved * math.cos(est_heading) + rng.normal(0.0, drift)
        est_y = est_y + moved * math.sin(est_heading) + rng.normal(0.0, drift)
        if action == ACTION_LEFT:
            est_heading += math.radians(15.0) + rng.normal(0.0, 0.5 * drift)
        elif action == ACTION_RIGHT:
            est_heading -= math.radians(15.0) + rng.normal(0.0, 0.5 * drift)
        est_heading = ((est_heading + math.pi) % (2.0 * math.pi)) - math.pi

        true_path.append((x, y))
        est_path.append((est_x, est_y))
        loc_err.append(math.hypot(est_x - x, est_y - y))

    # final reconstruction + metrics
    occ_prob_raw = _sigmoid(occ_logodds)
    occ_ratio = occ_obs / np.maximum(1e-6, (occ_obs + free_obs))
    occ_mask = ((occ_ratio > 0.58) & (occ_obs >= 2.0)) | (occ_prob_raw > 0.68)
    occ_mask = occ_mask & (_neighbor_count(occ_mask.astype(np.uint8)) >= 2)
    occ_mask = _remove_small_components(occ_mask, min_area=4)
    occ_pred = occ_mask.astype(np.uint8)

    stable_occ = occ_mask & (occ_obs >= 2.0)
    free_neighbors = _neighbor_count((~occ_mask).astype(np.uint8))
    wall_candidate = stable_occ & (free_neighbors >= 1)
    support = _neighbor_count(wall_candidate.astype(np.uint8)).astype(np.float32) / 8.0
    wall_probability = np.clip(0.45 * occ_ratio + 0.35 * support + 0.20 * np.clip(total_obs / 8.0, 0.0, 1.0), 0.0, 1.0)
    wall_probability[~wall_candidate] *= 0.3
    wall_probability[(support < 0.25) & (wall_probability < 0.6)] = 0.0
    wall_mask = (wall_probability >= 0.45).astype(np.uint8)
    wall_mask = _remove_small_components(wall_mask.astype(bool), min_area=3).astype(np.uint8)
    wall_segments = _extract_wall_segments(wall_mask, min_len=3)
    mean_seg_len = float(np.mean([s["length"] for s in wall_segments])) if wall_segments else 0.0

    doorway_prob, door_diag, accepted_centers = _detect_doorway_candidates(
        wall_segments, occ_prob_raw, free_obs, wall_probability, gt_door, cell_size
    )
    door_pred = (doorway_prob >= 0.35).astype(np.uint8)

    contradiction = np.minimum(occ_obs, free_obs) / np.maximum(1.0, total_obs)
    agreement = np.abs(occ_obs - free_obs) / np.maximum(1.0, total_obs)
    confidence = np.clip(np.clip(total_obs / 8.0, 0.0, 1.0) * agreement * (1.0 - 0.7 * contradiction), 0.0, 1.0)

    map_acc = float((occ_pred == gt_occ).mean())
    wall_err = float(compute_wall_reconstruction_error(wall_mask, gt_wall, cell_size))
    door_metrics = doorway_precision_recall(door_pred, gt_door) if m.name == "doorway" else {"precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}
    loc_rmse = float(np.sqrt(np.mean(np.square(loc_err)))) if loc_err else 0.0

    conf_norm = confidence
    correctness = (occ_pred == gt_occ).astype(np.float32)
    conf_quality = float((conf_norm * correctness).sum() / max(1e-6, conf_norm.sum()))
    high_mask = conf_norm >= 0.65
    low_mask = conf_norm <= 0.25
    high_acc = float(correctness[high_mask].mean()) if high_mask.any() else 0.0
    low_acc = float(correctness[low_mask].mean()) if low_mask.any() else 0.0
    observed_cell_ratio = float((total_obs > 0).mean())

    return {
        "map_accuracy": map_acc,
        "wall_reconstruction_error_m": wall_err,
        "doorway_precision": float(door_metrics["precision"]),
        "doorway_recall": float(door_metrics["recall"]),
        "localization_rmse_m": loc_rmse,
        "map_confidence_quality": conf_quality,
        "high_confidence_accuracy": high_acc,
        "low_confidence_accuracy": low_acc,
        "observed_cell_ratio": observed_cell_ratio,
        "collision_count": int(collisions),
        "timeout": int(timeout_flag),
        "steps": int(len(true_path) - 1),
        "action_counts": dict(action_counts),
        "final_coverage_proxy": observed_cell_ratio,
        "true_path": true_path,
        "est_path": est_path,
        "pred_occupancy": occ_pred,
        "gt_occupancy": gt_occ,
        "wall_probability": wall_probability,
        "doorway_probability": doorway_prob,
        "confidence": conf_norm,
        "wall_candidate_count": float(int((wall_candidate > 0).sum())),
        "extracted_wall_segment_count": float(len(wall_segments)),
        "mean_wall_segment_length": mean_seg_len,
        "wall_segments": wall_segments,
        "accepted_doorway_centers": accepted_centers,
        "doorway_candidate_count": float(door_diag["doorway_candidate_count"]),
        "accepted_doorway_candidate_count": float(door_diag["accepted_doorway_candidate_count"]),
        "rejected_doorway_candidate_count": float(door_diag["rejected_doorway_candidate_count"]),
        "false_doorway_count": float(door_diag["false_doorway_count"]),
        "frontier_switch_count": float(frontier_switch_count),
        "clutter_recovery_count": float(clutter_recovery_count),
        "loop_stagnation_count": float(loop_stagnation_count),
        "failure_reason": "collision" if collisions > 0 else "timeout",
    }


def summarize_episodes(episodes: List[Dict[str, object]]) -> Dict[str, object]:
    action_counter = Counter()
    failure_counter = Counter()
    for e in episodes:
        action_counter.update(e["action_counts"])  # type: ignore[arg-type]
        failure_counter.update([e["failure_reason"]])  # type: ignore[arg-type]
    total_actions = sum(action_counter.values())
    action_dist = {
        a: {"count": int(action_counter.get(a, 0)), "rate": float(action_counter.get(a, 0) / max(1, total_actions))}
        for a in [ACTION_FAST, ACTION_SLOW, ACTION_PROBE, ACTION_LEFT, ACTION_RIGHT, ACTION_RESAMPLE, ACTION_STOP]
    }
    return {
        "coverage_success_rate": float(np.mean([1.0 if e["final_coverage_proxy"] >= 0.60 else 0.0 for e in episodes])),
        "map_accuracy": float(np.mean([e["map_accuracy"] for e in episodes])),
        "wall_reconstruction_error_m": float(np.mean([e["wall_reconstruction_error_m"] for e in episodes])),
        "doorway_precision": float(np.mean([e["doorway_precision"] for e in episodes])),
        "doorway_recall": float(np.mean([e["doorway_recall"] for e in episodes])),
        "localization_rmse_m": float(np.mean([e["localization_rmse_m"] for e in episodes])),
        "map_confidence_quality": float(np.mean([e["map_confidence_quality"] for e in episodes])),
        "high_confidence_accuracy": float(np.mean([e["high_confidence_accuracy"] for e in episodes])),
        "low_confidence_accuracy": float(np.mean([e["low_confidence_accuracy"] for e in episodes])),
        "observed_cell_ratio": float(np.mean([e["observed_cell_ratio"] for e in episodes])),
        "collision_rate": float(np.mean([1.0 if e["collision_count"] > 0 else 0.0 for e in episodes])),
        "timeout_rate": float(np.mean([e["timeout"] for e in episodes])),
        "mean_final_coverage": float(np.mean([e["final_coverage_proxy"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "action_distribution": action_dist,
        "doorway_commitment_count": 0.0,  # kept for interface compatibility
        "clutter_recovery_count": float(np.mean([e["clutter_recovery_count"] for e in episodes])),
        "loop_stagnation_count": float(np.mean([e["loop_stagnation_count"] for e in episodes])),
        "average_frontier_switches": float(np.mean([e["frontier_switch_count"] for e in episodes])),
        "wall_candidate_count": float(np.mean([e["wall_candidate_count"] for e in episodes])),
        "extracted_wall_segment_count": float(np.mean([e["extracted_wall_segment_count"] for e in episodes])),
        "mean_wall_segment_length": float(np.mean([e["mean_wall_segment_length"] for e in episodes])),
        "doorway_candidate_count": float(np.mean([e["doorway_candidate_count"] for e in episodes])),
        "accepted_doorway_candidate_count": float(np.mean([e["accepted_doorway_candidate_count"] for e in episodes])),
        "rejected_doorway_candidate_count": float(np.mean([e["rejected_doorway_candidate_count"] for e in episodes])),
        "false_doorway_count": float(np.mean([e["false_doorway_count"] for e in episodes])),
        "failure_reason_counts": dict(failure_counter),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2A.1 improved acoustic mapping baseline.")
    p.add_argument("--episodes-per-map", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--maps", type=str, default="empty_room,corridor,single_block,doorway,cluttered_room")
    p.add_argument("--difficulties", type=str, default="clean,mild_noise,medium_noise,hard_noise")
    p.add_argument("--grid-size", type=int, default=0)
    p.add_argument("--cell-size", type=float, default=0.25)
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/phase2_mapping_v2")
    p.add_argument("--seed", type=int, default=20260531)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    maps = {m.name: m for m in make_maps()}
    map_names = [m.strip() for m in args.maps.split(",") if m.strip()]
    diff_names = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    for d in diff_names:
        if d not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {d}")

    rng = np.random.default_rng(args.seed)
    all_results: Dict[str, Dict[str, object]] = defaultdict(dict)
    samples: Dict[Tuple[str, str], Dict[str, object]] = {}

    for d in diff_names:
        preset = DIFFICULTY_PRESETS[d]
        acc_so_far = []
        loc_so_far = []
        for mi, map_name in enumerate(map_names, start=1):
            episodes = []
            for ep_idx in range(1, args.episodes_per_map + 1):
                ep = run_single_episode(maps[map_name], d, preset, args.max_steps, args.cell_size, rng)
                episodes.append(ep)
                acc_so_far.append(ep["map_accuracy"])
                loc_so_far.append(ep["localization_rmse_m"])
                elapsed = time.time() - t0
                print(
                    f"[Phase2A.1] map={map_name} ({mi}/{len(map_names)}) diff={d} ep={ep_idx}/{args.episodes_per_map} "
                    f"elapsed={elapsed:.1f}s mean_acc={np.mean(acc_so_far):.3f} mean_loc_rmse={np.mean(loc_so_far):.3f}m"
                )
            all_results[d][map_name] = summarize_episodes(episodes)
            samples[(d, map_name)] = episodes[-1]
            if args.save_plots:
                _save_overlay_plot(out_dir, map_name, d, episodes[-1])

    # aggregate
    agg = {}
    for d in diff_names:
        vals = [all_results[d][m] for m in map_names]
        agg[d] = {
            "mean_map_accuracy": float(np.mean([v["map_accuracy"] for v in vals])),
            "mean_wall_reconstruction_error_m": float(np.mean([v["wall_reconstruction_error_m"] for v in vals])),
            "mean_doorway_precision": float(np.mean([v["doorway_precision"] for v in vals])),
            "mean_doorway_recall": float(np.mean([v["doorway_recall"] for v in vals])),
            "mean_localization_rmse_m": float(np.mean([v["localization_rmse_m"] for v in vals])),
            "mean_collision_rate": float(np.mean([v["collision_rate"] for v in vals])),
            "mean_timeout_rate": float(np.mean([v["timeout_rate"] for v in vals])),
        }

    # v1 comparison if exists
    v1_path = Path("runs/phase2_mapping/phase2_mapping_results.json")
    comparison_v1_v2 = {}
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        for d in diff_names:
            if d in v1.get("difficulty_aggregate", {}) and d in agg:
                comparison_v1_v2[d] = {
                    "v1_map_accuracy": v1["difficulty_aggregate"][d]["mean_map_accuracy"],
                    "v2_map_accuracy": agg[d]["mean_map_accuracy"],
                    "v1_wall_reconstruction_error_m": v1["difficulty_aggregate"][d]["mean_wall_reconstruction_error_m"],
                    "v2_wall_reconstruction_error_m": agg[d]["mean_wall_reconstruction_error_m"],
                    "v1_doorway_precision": v1["difficulty_aggregate"][d]["mean_doorway_precision"],
                    "v2_doorway_precision": agg[d]["mean_doorway_precision"],
                    "v1_doorway_recall": v1["difficulty_aggregate"][d]["mean_doorway_recall"],
                    "v2_doorway_recall": agg[d]["mean_doorway_recall"],
                    "v1_localization_rmse_m": v1["difficulty_aggregate"][d]["mean_localization_rmse_m"],
                    "v2_localization_rmse_m": agg[d]["mean_localization_rmse_m"],
                    "v1_collision_rate": v1["difficulty_aggregate"][d]["mean_collision_rate"],
                    "v2_collision_rate": agg[d]["mean_collision_rate"],
                    "v1_timeout_rate": v1["difficulty_aggregate"][d]["mean_timeout_rate"],
                    "v2_timeout_rate": agg[d]["mean_timeout_rate"],
                }

    # acceptance
    accepted_clean = ("clean" in agg and agg["clean"]["mean_map_accuracy"] >= 0.75 and agg["clean"]["mean_wall_reconstruction_error_m"] < 1.75 and agg["clean"]["mean_doorway_recall"] > 0.10 and agg["clean"]["mean_collision_rate"] <= 0.0)
    accepted_mild = ("mild_noise" in agg and agg["mild_noise"]["mean_map_accuracy"] >= 0.72 and agg["mild_noise"]["mean_collision_rate"] <= 0.0)
    accepted_medium = ("medium_noise" in agg and agg["medium_noise"]["mean_map_accuracy"] >= 0.68 and agg["medium_noise"]["mean_collision_rate"] <= 0.0)
    accepted_hard = ("hard_noise" in agg and agg["hard_noise"]["mean_map_accuracy"] >= 0.63 and agg["hard_noise"]["mean_collision_rate"] <= 0.0)
    accepted_overall = bool(accepted_clean and accepted_mild and accepted_medium and accepted_hard)
    needs_more_tuning = not accepted_overall

    results = {
        "phase": "Phase 2A.1 improved acoustic mapping baseline",
        "episodes_per_map": args.episodes_per_map,
        "max_steps": args.max_steps,
        "cell_size": args.cell_size,
        "maps": map_names,
        "difficulties": diff_names,
        "per_map_metrics": all_results,
        "difficulty_aggregate": agg,
        "comparison_v1_v2": comparison_v1_v2,
        "acceptance": {
            "accepted_mapping_clean": bool(accepted_clean),
            "accepted_mapping_mild": bool(accepted_mild),
            "accepted_mapping_medium": bool(accepted_medium),
            "accepted_mapping_hard": bool(accepted_hard),
            "accepted_mapping_overall": bool(accepted_overall),
            "needs_more_mapping_tuning": bool(needs_more_tuning),
        },
    }
    (out_dir / "phase2_mapping_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "per_map_metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\nPhase 2A.1 acceptance:")
    print(results["acceptance"])
    print(f"Saved: {out_dir / 'phase2_mapping_results.json'}")


if __name__ == "__main__":
    main()

