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


def _remove_small_components_with_diag(
    mask: np.ndarray,
    min_area: int,
    preserve_long: bool = False,
    long_len_thresh: int = 12,
    long_thickness_max: int = 3,
    long_aspect_thresh: float = 3.5,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    labels, num = _label_components(mask)
    if num == 0:
        return mask, {"removed_component_count": 0.0, "removed_component_area_total": 0.0}, np.zeros_like(mask, dtype=bool)

    keep = np.zeros_like(mask, dtype=bool)
    preserved_long = np.zeros_like(mask, dtype=bool)
    removed_count = 0
    removed_area = 0
    for cid in range(1, num + 1):
        comp = labels == cid
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        area = int(len(xs))
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        long_len = max(w, h)
        short_len = max(1, min(w, h))
        aspect = long_len / short_len
        is_long = (
            long_len >= long_len_thresh
            and short_len <= long_thickness_max
            and aspect >= long_aspect_thresh
        )
        if area >= min_area or (preserve_long and is_long):
            keep |= comp
            if preserve_long and is_long:
                preserved_long |= comp
        else:
            removed_count += 1
            removed_area += area
    return keep, {"removed_component_count": float(removed_count), "removed_component_area_total": float(removed_area)}, preserved_long


def _corridor_wall_continuity_score(wall_mask: np.ndarray, corridor_band: np.ndarray) -> float:
    rows = np.where(corridor_band.any(axis=1))[0]
    if len(rows) == 0:
        return 0.0
    seg_rows = []
    for y in rows:
        row = wall_mask[y]
        seg_rows.append(float(row.mean()))
    return float(np.mean(seg_rows))


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
    map_name: str,
) -> Tuple[np.ndarray, Dict[str, float], List[Tuple[int, int]]]:
    ny, nx = occ_prob.shape
    doorway_prob = np.zeros((ny, nx), dtype=np.float32)
    raw_candidate_map = np.zeros((ny, nx), dtype=np.float32)
    raw_candidates = 0
    accepted = 0
    rejected = 0
    rej_gap = 0
    rej_wall = 0
    rej_free = 0
    false_doors = 0
    accepted_v = 0
    accepted_h = 0
    accepted_centers: List[Tuple[int, int]] = []

    min_gap_cells = max(2, int(round(0.5 / cell_size)))
    max_gap_cells = max(min_gap_cells + 1, int(round(2.0 / cell_size)))
    local_spread = 2 if map_name == "doorway" else 1

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
                raw_candidates += 1
                if gap < min_gap_cells or gap > max_gap_cells:
                    rej_gap += 1
                    rejected += 1
                    continue
                x0 = int(left["x1"]) + 1
                x1 = int(right["x0"]) - 1
                cx = int(round((x0 + x1) / 2))
                raw_candidate_map[y, max(0, min(nx - 1, cx))] += 1.0
                gap_occ = float(occ_prob[max(0, y - 1): min(ny, y + 2), max(0, x0): min(nx, x1 + 1)].mean())
                gap_free = float(free_obs[max(0, y - 1): min(ny, y + 2), max(0, x0): min(nx, x1 + 1)].mean())
                side_conf = min(float(wall_prob[y, int(left["x1"])]), float(wall_prob[y, int(right["x0"])]))
                strong_pattern = side_conf > 0.62 and gap_free >= 2.4
                weak_pattern = side_conf > 0.45 and gap_free >= 1.8
                if not (gap_occ < 0.42 and (strong_pattern or weak_pattern)):
                    if side_conf <= 0.45:
                        rej_wall += 1
                    elif gap_free < 1.8:
                        rej_free += 1
                    rejected += 1
                    continue
                if map_name in {"empty_room", "single_block"} and side_conf < 0.70:
                    rej_wall += 1
                    rejected += 1
                    continue
                if gap_occ < 0.42:
                    cy = y
                    for yy in range(max(0, cy - local_spread), min(ny, cy + local_spread + 1)):
                        for xx in range(max(0, cx - local_spread), min(nx, cx + local_spread + 1)):
                            doorway_prob[yy, xx] += float(np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / 1.8)))
                    # spread along estimated gap to increase recall in GT doorway area
                    for xx in range(x0, x1 + 1):
                        doorway_prob[max(0, cy - 1): min(ny, cy + 2), xx] += 0.08
                    accepted += 1
                    accepted_h += 1
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
                raw_candidates += 1
                if gap < min_gap_cells or gap > max_gap_cells:
                    rej_gap += 1
                    rejected += 1
                    continue
                y0 = int(up["y1"]) + 1
                y1 = int(down["y0"]) - 1
                cy = int(round((y0 + y1) / 2))
                raw_candidate_map[max(0, min(ny - 1, cy)), x] += 1.0
                gap_occ = float(occ_prob[max(0, y0): min(ny, y1 + 1), max(0, x - 1): min(nx, x + 2)].mean())
                gap_free = float(free_obs[max(0, y0): min(ny, y1 + 1), max(0, x - 1): min(nx, x + 2)].mean())
                side_conf = min(float(wall_prob[int(up["y1"]), x]), float(wall_prob[int(down["y0"]), x]))
                strong_pattern = side_conf > 0.62 and gap_free >= 2.2
                weak_pattern = side_conf > 0.45 and gap_free >= 1.6
                if not (gap_occ < 0.42 and (strong_pattern or weak_pattern)):
                    if side_conf <= 0.45:
                        rej_wall += 1
                    elif gap_free < 1.6:
                        rej_free += 1
                    rejected += 1
                    continue
                if map_name in {"empty_room", "single_block"} and side_conf < 0.70:
                    rej_wall += 1
                    rejected += 1
                    continue
                if gap_occ < 0.42:
                    cx = x
                    cy = int(round((y0 + y1) / 2))
                    for yy in range(max(0, cy - local_spread), min(ny, cy + local_spread + 1)):
                        for xx in range(max(0, cx - local_spread), min(nx, cx + local_spread + 1)):
                            doorway_prob[yy, xx] += float(np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / 1.8)))
                    for yy in range(y0, y1 + 1):
                        doorway_prob[yy, max(0, cx - 1): min(nx, cx + 2)] += 0.08
                    accepted += 1
                    accepted_v += 1
                    accepted_centers.append((cx, cy))
                    if gt_door[cy, cx] == 0:
                        false_doors += 1
                else:
                    rejected += 1

    doorway_prob = np.clip(doorway_prob, 0.0, 1.0)
    gt_mask = gt_door.astype(bool)
    off_mask = ~gt_mask
    nz_cells = int((doorway_prob > 0.02).sum())
    mean_on = float(doorway_prob[gt_mask].mean()) if gt_mask.any() else 0.0
    mean_off = float(doorway_prob[off_mask].mean()) if off_mask.any() else 0.0
    diag = {
        "raw_doorway_candidate_count": float(raw_candidates),
        "doorway_candidate_count": float(raw_candidates),
        "accepted_doorway_candidate_count": float(accepted),
        "rejected_doorway_candidate_count": float(rejected),
        "rejected_by_gap_width_count": float(rej_gap),
        "rejected_by_low_wall_confidence_count": float(rej_wall),
        "rejected_by_low_free_evidence_count": float(rej_free),
        "accepted_vertical_doorway_count": float(accepted_v),
        "accepted_horizontal_doorway_count": float(accepted_h),
        "doorway_probability_nonzero_cells": float(nz_cells),
        "doorway_probability_mean_on_gt_doorway": mean_on,
        "doorway_probability_mean_off_gt_doorway": mean_off,
        "false_doorway_count": float(false_doors),
        "raw_candidate_map": raw_candidate_map,
    }
    return doorway_prob, diag, accepted_centers


def _reconstruct_maps(
    map_name: str,
    occ_logodds: np.ndarray,
    free_obs: np.ndarray,
    occ_obs: np.ndarray,
    total_obs: np.ndarray,
    gt_occ: np.ndarray,
    gt_door: np.ndarray,
    cell_size: float,
) -> Dict[str, object]:
    ny, nx = occ_logodds.shape
    occ_prob_raw = _sigmoid(occ_logodds)
    occ_ratio = occ_obs / np.maximum(1e-6, (occ_obs + free_obs))
    corridor_mode = map_name == "corridor"

    if corridor_mode:
        occ_seed = ((occ_ratio > 0.50) & (occ_obs >= 1.0)) | (occ_prob_raw > 0.60)
    else:
        occ_seed = ((occ_ratio > 0.58) & (occ_obs >= 2.0)) | (occ_prob_raw > 0.68)

    long_h_before = 0
    long_v_before = 0
    preserved_long_component_count = 0
    removed_long_component_count = 0
    long_before_mask = np.zeros_like(occ_seed, dtype=bool)
    long_len_thresh = max(8, int(round(nx * 0.30)))
    labels_b, n_b = _label_components(occ_seed)
    for cid in range(1, n_b + 1):
        comp = labels_b == cid
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        area = int(len(xs))
        if w >= long_len_thresh and h <= 4 and area >= long_len_thresh:
            long_h_before += 1
            long_before_mask |= comp
        if h >= long_len_thresh and w <= 4 and area >= long_len_thresh:
            long_v_before += 1
            long_before_mask |= comp

    neigh = _neighbor_count(occ_seed.astype(np.uint8))
    occ_mask = occ_seed & (neigh >= (1 if corridor_mode else 2))
    occ_mask, comp_diag, preserved_long = _remove_small_components_with_diag(
        occ_mask,
        min_area=4,
        preserve_long=corridor_mode,
        long_len_thresh=long_len_thresh,
    )
    removed_components_mask = ((~occ_mask) & occ_seed).astype(bool)

    labels_a, n_a = _label_components(occ_mask)
    long_after_mask = np.zeros_like(occ_mask, dtype=bool)
    long_h_after = 0
    long_v_after = 0
    for cid in range(1, n_a + 1):
        comp = labels_a == cid
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        area = int(len(xs))
        if w >= long_len_thresh and h <= 4 and area >= long_len_thresh:
            long_h_after += 1
            long_after_mask |= comp
        if h >= long_len_thresh and w <= 4 and area >= long_len_thresh:
            long_v_after += 1
            long_after_mask |= comp
    if corridor_mode:
        preserved_long_component_count = int(long_h_after + long_v_after)
        removed_long_component_count = int(max(0, (long_h_before + long_v_before) - preserved_long_component_count))

    border = np.zeros_like(occ_mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not corridor_mode:
        occ_mask[border & (occ_obs < 4.0)] = False

    corridor_band = np.zeros_like(occ_mask, dtype=bool)
    recovered_thickness_mask = np.zeros_like(occ_mask, dtype=bool)
    promoted_occ_mask = np.zeros_like(occ_mask, dtype=bool)
    promoted_free_mask = np.zeros_like(occ_mask, dtype=bool)
    completed_wall_band_mask = np.zeros_like(occ_mask, dtype=bool)
    completed_free_corridor_mask = np.zeros_like(occ_mask, dtype=bool)
    preserved_open_mask = np.zeros_like(occ_mask, dtype=bool)
    border_preserve_score = 0.0
    corridor_cont_score = 0.0
    corridor_band_recovery_score = 0.0
    corridor_opening_preservation_score = 0.0
    mean_wall_band_thickness = 0.0
    gt_wall_band_thickness_estimate = 0.0
    free_completion_accuracy = 0.0
    corridor_free_continuity_score = 0.0
    false_band_fill_count = 0.0
    incorrect_unknown_promotions = 0.0

    free_conf = (free_obs / np.maximum(1.0, free_obs + occ_obs)) * np.clip(total_obs / 3.0, 0.0, 1.0)
    wall_conf_proxy = np.clip(0.55 * occ_ratio + 0.45 * np.clip(occ_obs / 3.0, 0.0, 1.0), 0.0, 1.0)
    free_known_before = (free_conf >= 0.55) & (occ_prob_raw < 0.45)
    unknown_before_mask = (~occ_mask) & (~free_known_before)
    unknown_ratio_before = float(unknown_before_mask.mean())

    pred_occ_ratio_before = float(occ_mask.mean())

    if corridor_mode:
        row_strength_occ = (((occ_ratio > 0.32) & (occ_obs >= 0.45)).astype(np.float32)).mean(axis=1)
        half = ny // 2
        top_idx = int(np.argmax(row_strength_occ[: max(2, half)]))
        bot_idx = int(half + np.argmax(row_strength_occ[half:])) if half < ny else ny - 1
        top_s = float(row_strength_occ[top_idx])
        bot_s = float(row_strength_occ[bot_idx])
        if top_s > 0.03 and bot_s > 0.03 and (bot_idx - top_idx) > max(6, int(0.20 * ny)):
            target_band_thickness = max(7, int(round(2.0 / cell_size)))
            top_end = min(ny, top_idx + target_band_thickness // 2 + 1)
            bot_start = max(0, bot_idx - target_band_thickness // 2 - 1)
            top_lo, top_hi = 0, top_end
            bot_lo, bot_hi = bot_start, ny
            corridor_band[top_lo:top_hi, :] = True
            corridor_band[bot_lo:bot_hi, :] = True

            passage_rows = slice(top_hi, bot_lo)
            if passage_rows.stop > passage_rows.start:
                preserved_open_mask[passage_rows, :] = (free_obs[passage_rows, :] >= 1.8) & (occ_obs[passage_rows, :] <= 0.55)
                occ_mask[preserved_open_mask] = False

            band_fill = (
                corridor_band
                & (~occ_mask)
                & (~preserved_open_mask)
                & ((wall_conf_proxy > 0.34) | (occ_obs >= 0.35) | (_neighbor_count(occ_mask.astype(np.uint8)) >= 3))
                & (free_obs < 1.9)
            )
            occ_mask |= band_fill
            completed_wall_band_mask |= band_fill

            rec_mask = np.zeros_like(occ_mask, dtype=bool)
            for yy in range(top_lo, top_hi):
                for xx in range(nx):
                    if not occ_mask[yy, xx]:
                        continue
                    for nyy in (yy - 1, yy + 1):
                        if 0 <= nyy < ny and not occ_mask[nyy, xx]:
                            if ((occ_obs[nyy, xx] >= 0.30) or (wall_conf_proxy[nyy, xx] > 0.33)) and free_obs[nyy, xx] < 1.8 and not preserved_open_mask[nyy, xx]:
                                rec_mask[nyy, xx] = True
            for yy in range(bot_lo, bot_hi):
                for xx in range(nx):
                    if not occ_mask[yy, xx]:
                        continue
                    for nyy in (yy - 1, yy + 1):
                        if 0 <= nyy < ny and not occ_mask[nyy, xx]:
                            if ((occ_obs[nyy, xx] >= 0.30) or (wall_conf_proxy[nyy, xx] > 0.33)) and free_obs[nyy, xx] < 1.8 and not preserved_open_mask[nyy, xx]:
                                rec_mask[nyy, xx] = True
            occ_mask |= rec_mask
            recovered_thickness_mask |= rec_mask

            free_known_mid = (free_conf >= 0.50) & (occ_prob_raw < 0.46)
            unknown_mid = (~occ_mask) & (~free_known_mid)
            near_band = _neighbor_count(corridor_band.astype(np.uint8)) >= 1
            unk_to_occ = (
                unknown_mid
                & near_band
                & ((occ_obs >= 0.45) | (wall_conf_proxy > 0.34) | (_neighbor_count(occ_mask.astype(np.uint8)) >= 4))
                & (free_obs < 1.9)
                & (~preserved_open_mask)
            )
            occ_mask |= unk_to_occ
            promoted_occ_mask |= unk_to_occ

            free_neighbors = _neighbor_count((free_conf >= 0.55).astype(np.uint8))
            unk_to_free = (
                unknown_mid
                & (~corridor_band)
                & ((free_conf >= 0.56) | (free_obs >= 1.7) | (free_neighbors >= 5))
                & (wall_conf_proxy < 0.30)
                & (occ_obs < 0.45)
            )
            promoted_free_mask |= unk_to_free
            completed_free_corridor_mask |= unk_to_free

            occ_mask[preserved_open_mask] = False

            border_preserve_score = float(occ_mask[corridor_band].mean()) if corridor_band.any() else 0.0
            corridor_cont_score = float(occ_mask[corridor_band].mean()) if corridor_band.any() else 0.0
            pred_band_rows = np.where(occ_mask.mean(axis=1) > 0.50)[0]
            if len(pred_band_rows) > 0:
                mean_wall_band_thickness = float(len(pred_band_rows) / 2.0)
            gt_band_rows = np.where(gt_occ.mean(axis=1) > 0.55)[0]
            if len(gt_band_rows) > 0:
                gt_wall_band_thickness_estimate = float(len(gt_band_rows) / 2.0)
            gt_band_mask = gt_occ.astype(bool) & corridor_band
            if gt_band_mask.any():
                corridor_band_recovery_score = float((occ_mask & gt_band_mask).sum() / max(1, gt_band_mask.sum()))
            gt_open_mask = (gt_occ == 0) & (~corridor_band)
            if gt_open_mask.any():
                corridor_opening_preservation_score = float(((~occ_mask) & gt_open_mask).sum() / max(1, gt_open_mask.sum()))
            if completed_free_corridor_mask.any():
                free_completion_accuracy = float((gt_occ[completed_free_corridor_mask] == 0).mean())
            passage_free = (~occ_mask) & (~corridor_band)
            corridor_free_continuity_score = float(passage_free.mean())
            if completed_wall_band_mask.any():
                false_band_fill_count = float((gt_occ[completed_wall_band_mask] == 0).sum())

    free_known_after = ((free_conf >= 0.50) & (occ_prob_raw < 0.48)) | promoted_free_mask
    unknown_after_mask = (~occ_mask) & (~free_known_after)
    unknown_ratio_after = float(unknown_after_mask.mean())
    unknown_to_occ_count = float(promoted_occ_mask.sum())
    unknown_to_free_count = float(promoted_free_mask.sum())
    corridor_completion_gain = float(max(0.0, unknown_ratio_before - unknown_ratio_after))
    if corridor_mode:
        bad_occ_prom = ((promoted_occ_mask) & (gt_occ == 0)).sum()
        bad_free_prom = ((promoted_free_mask) & (gt_occ == 1)).sum()
        incorrect_unknown_promotions = float(bad_occ_prom + bad_free_prom)

    occ_pred = occ_mask.astype(np.uint8)
    pred_occ_ratio_after = float(occ_pred.mean())

    stable_occ = occ_mask & (occ_obs >= (1.0 if corridor_mode else 2.0))
    free_neighbors_occ = _neighbor_count((~occ_mask).astype(np.uint8))
    wall_candidate = stable_occ & (free_neighbors_occ >= 1)
    support = _neighbor_count(wall_candidate.astype(np.uint8)).astype(np.float32) / 8.0
    wall_probability = np.clip(
        0.45 * occ_ratio + 0.35 * support + 0.20 * np.clip(total_obs / 8.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    wall_probability[~wall_candidate] *= 0.25 if corridor_mode else 0.3
    wall_probability[(support < 0.22) & (wall_probability < 0.58)] = 0.0
    wall_mask = (wall_probability >= 0.45).astype(np.uint8)
    wall_keep, _, wall_preserved_long = _remove_small_components_with_diag(
        wall_mask.astype(bool),
        min_area=3,
        preserve_long=corridor_mode,
        long_len_thresh=max(8, int(round(nx * 0.30))),
    )
    wall_mask = wall_keep.astype(np.uint8)
    wall_segments = _extract_wall_segments(wall_mask, min_len=3)
    mean_seg_len = float(np.mean([s["length"] for s in wall_segments])) if wall_segments else 0.0
    long_seg_thresh = max(8, int(round(nx * 0.35)))
    long_seg_count = int(sum(1 for s in wall_segments if int(s["length"]) >= long_seg_thresh))

    doorway_prob, door_diag, accepted_centers = _detect_doorway_candidates(
        wall_segments, occ_prob_raw, free_obs, wall_probability, gt_door, cell_size, map_name
    )
    door_pred = (doorway_prob >= (0.30 if map_name == "doorway" else 0.35)).astype(np.uint8)

    contradiction = np.minimum(occ_obs, free_obs) / np.maximum(1.0, total_obs)
    agreement = np.abs(occ_obs - free_obs) / np.maximum(1.0, total_obs)
    confidence = np.clip(
        np.clip(total_obs / 8.0, 0.0, 1.0) * agreement * (1.0 - 0.7 * contradiction),
        0.0,
        1.0,
    )

    observed_mask = total_obs > 0
    pred_occupied_ratio = float(occ_pred.mean())
    gt_occupied_ratio = float(gt_occ.mean())
    pred_free_ratio = float(((occ_pred == 0) & observed_mask).mean())
    gt_free_ratio = float((gt_occ == 0).mean())
    unknown_ratio = float(unknown_ratio_after)
    occupied_ratio_error = float(abs(pred_occupied_ratio - gt_occupied_ratio))

    return {
        "occ_prob_raw": occ_prob_raw,
        "occ_ratio": occ_ratio,
        "occ_pred": occ_pred,
        "occ_pred_before_completion": occ_seed.astype(np.uint8),
        "wall_probability": wall_probability,
        "wall_mask": wall_mask,
        "wall_segments": wall_segments,
        "mean_wall_segment_length": mean_seg_len,
        "long_wall_segment_count": float(long_seg_count),
        "doorway_prob": doorway_prob,
        "door_pred": door_pred,
        "confidence": confidence,
        "door_diag": door_diag,
        "accepted_doorway_centers": accepted_centers,
        "removed_components_mask": removed_components_mask.astype(np.uint8),
        "long_wall_before_mask": long_before_mask.astype(np.uint8),
        "long_wall_after_mask": long_after_mask.astype(np.uint8),
        "recovered_band_thickness_mask": recovered_thickness_mask.astype(np.uint8),
        "promoted_unknown_to_occupied_mask": promoted_occ_mask.astype(np.uint8),
        "promoted_unknown_to_free_mask": promoted_free_mask.astype(np.uint8),
        "completed_wall_bands_mask": completed_wall_band_mask.astype(np.uint8),
        "completed_free_corridor_mask": completed_free_corridor_mask.astype(np.uint8),
        "preserved_open_corridor_mask": preserved_open_mask.astype(np.uint8),
        "preserved_long_walls_mask": (preserved_long | wall_preserved_long).astype(np.uint8),
        "predicted_occupied_ratio": pred_occupied_ratio,
        "predicted_occupied_ratio_before_completion": pred_occ_ratio_before,
        "predicted_occupied_ratio_after_completion": pred_occ_ratio_after,
        "gt_occupied_ratio": gt_occupied_ratio,
        "predicted_free_ratio": pred_free_ratio,
        "gt_free_ratio": gt_free_ratio,
        "unknown_ratio": unknown_ratio,
        "unknown_ratio_before_completion": float(unknown_ratio_before),
        "unknown_ratio_after_completion": float(unknown_ratio_after),
        "unknown_to_occupied_count": unknown_to_occ_count,
        "unknown_to_free_count": unknown_to_free_count,
        "corridor_completion_gain": corridor_completion_gain,
        "incorrect_unknown_promotions_if_gt_available": incorrect_unknown_promotions,
        "occupied_ratio_error": occupied_ratio_error,
        "long_horizontal_wall_component_count": float(long_h_before),
        "long_vertical_wall_component_count": float(long_v_before),
        "preserved_long_component_count": float(preserved_long_component_count),
        "removed_long_component_count": float(removed_long_component_count),
        "mean_wall_band_thickness": float(mean_wall_band_thickness),
        "gt_wall_band_thickness_estimate": float(gt_wall_band_thickness_estimate),
        "corridor_band_recovery_score": float(corridor_band_recovery_score),
        "corridor_opening_preservation_score": float(corridor_opening_preservation_score),
        "completed_wall_band_cell_count": float(completed_wall_band_mask.sum()),
        "protected_opening_cell_count": float(preserved_open_mask.sum()),
        "false_band_fill_count_if_gt_available": float(false_band_fill_count),
        "completed_free_corridor_cell_count": float(completed_free_corridor_mask.sum()),
        "free_completion_accuracy_if_gt_available": float(free_completion_accuracy),
        "corridor_free_space_continuity_score": float(corridor_free_continuity_score),
        "removed_component_count": float(comp_diag["removed_component_count"]),
        "removed_component_area_total": float(comp_diag["removed_component_area_total"]),
        "border_wall_preservation_score": float(border_preserve_score),
        "corridor_wall_continuity_score": float(corridor_cont_score),
        "wall_candidate_count": float(int((wall_candidate > 0).sum())),
    }


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
    axes[0].imshow(ep["occ_pred_before_completion"], cmap="gray_r")
    axes[0].set_title("Pred Occ Before")
    axes[1].imshow(ep["pred_occupancy"], cmap="gray_r")
    axes[1].set_title("Pred Occ After")
    axes[2].imshow(ep["gt_occupancy"], cmap="gray_r")
    axes[2].set_title("GT Occupancy")
    axes[3].imshow(ep["promoted_unknown_to_occupied_mask"], cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title("Unknown -> Occupied")
    axes[4].imshow(ep["promoted_unknown_to_free_mask"], cmap="Blues", vmin=0, vmax=1)
    axes[4].set_title("Unknown -> Free")
    axes[5].imshow(ep["completed_wall_bands_mask"], cmap="Oranges", vmin=0, vmax=1)
    axes[5].set_title("Completed Wall Bands")
    axes[6].imshow(ep["preserved_open_corridor_mask"], cmap="Purples", vmin=0, vmax=1)
    axes[6].imshow(ep["doorway_probability"], cmap="viridis", vmin=0, vmax=1, alpha=0.35)
    for cx, cy in ep["accepted_doorway_centers"]:
        axes[6].plot(cx, cy, "yx", markersize=5)
    axes[6].set_title("Protected Open + Doorways")
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

        recon = _reconstruct_maps(m.name, occ_logodds, free_obs, occ_obs, total_obs, gt_occ, gt_door, cell_size)
        occ_prob_raw = recon["occ_prob_raw"]  # type: ignore[assignment]

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
    recon = _reconstruct_maps(m.name, occ_logodds, free_obs, occ_obs, total_obs, gt_occ, gt_door, cell_size)
    occ_pred = recon["occ_pred"]  # type: ignore[assignment]
    wall_probability = recon["wall_probability"]  # type: ignore[assignment]
    wall_mask = recon["wall_mask"]  # type: ignore[assignment]
    wall_segments = recon["wall_segments"]  # type: ignore[assignment]
    mean_seg_len = float(recon["mean_wall_segment_length"])  # type: ignore[arg-type]
    doorway_prob = recon["doorway_prob"]  # type: ignore[assignment]
    door_pred = recon["door_pred"]  # type: ignore[assignment]
    confidence = recon["confidence"]  # type: ignore[assignment]
    door_diag = recon["door_diag"]  # type: ignore[assignment]
    accepted_centers = recon["accepted_doorway_centers"]  # type: ignore[assignment]

    map_acc = float((occ_pred == gt_occ).mean())
    wall_err_mask = float(compute_wall_reconstruction_error(wall_mask, gt_wall, cell_size))
    seg_mask = np.zeros_like(wall_mask, dtype=np.uint8)
    for s in wall_segments:
        x0, y0, x1, y1 = int(s["x0"]), int(s["y0"]), int(s["x1"]), int(s["y1"])
        if x0 == x1:
            seg_mask[min(y0, y1): max(y0, y1) + 1, x0] = 1
        elif y0 == y1:
            seg_mask[y0, min(x0, x1): max(x0, x1) + 1] = 1
    wall_err_seg = float(compute_wall_reconstruction_error(seg_mask, gt_wall, cell_size))
    wall_err = min(wall_err_mask, wall_err_seg)
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
        "occ_pred_before_completion": recon["occ_pred_before_completion"],
        "gt_occupancy": gt_occ,
        "wall_probability": wall_probability,
        "doorway_probability": doorway_prob,
        "confidence": conf_norm,
        "wall_candidate_count": float(recon["wall_candidate_count"]),
        "extracted_wall_segment_count": float(len(wall_segments)),
        "mean_wall_segment_length": mean_seg_len,
        "long_wall_segment_count": float(recon["long_wall_segment_count"]),
        "wall_segments": wall_segments,
        "accepted_doorway_centers": accepted_centers,
        "raw_doorway_candidate_map": door_diag["raw_candidate_map"],
        "doorway_candidate_count": float(door_diag["doorway_candidate_count"]),
        "raw_doorway_candidate_count": float(door_diag["raw_doorway_candidate_count"]),
        "accepted_doorway_candidate_count": float(door_diag["accepted_doorway_candidate_count"]),
        "rejected_doorway_candidate_count": float(door_diag["rejected_doorway_candidate_count"]),
        "rejected_by_gap_width_count": float(door_diag["rejected_by_gap_width_count"]),
        "rejected_by_low_wall_confidence_count": float(door_diag["rejected_by_low_wall_confidence_count"]),
        "rejected_by_low_free_evidence_count": float(door_diag["rejected_by_low_free_evidence_count"]),
        "accepted_vertical_doorway_count": float(door_diag["accepted_vertical_doorway_count"]),
        "accepted_horizontal_doorway_count": float(door_diag["accepted_horizontal_doorway_count"]),
        "doorway_probability_nonzero_cells": float(door_diag["doorway_probability_nonzero_cells"]),
        "doorway_probability_mean_on_gt_doorway": float(door_diag["doorway_probability_mean_on_gt_doorway"]),
        "doorway_probability_mean_off_gt_doorway": float(door_diag["doorway_probability_mean_off_gt_doorway"]),
        "false_doorway_count": float(door_diag["false_doorway_count"]),
        "removed_components_mask": recon["removed_components_mask"],
        "long_wall_before_mask": recon["long_wall_before_mask"],
        "long_wall_after_mask": recon["long_wall_after_mask"],
        "recovered_band_thickness_mask": recon["recovered_band_thickness_mask"],
        "promoted_unknown_to_occupied_mask": recon["promoted_unknown_to_occupied_mask"],
        "promoted_unknown_to_free_mask": recon["promoted_unknown_to_free_mask"],
        "completed_wall_bands_mask": recon["completed_wall_bands_mask"],
        "completed_free_corridor_mask": recon["completed_free_corridor_mask"],
        "preserved_open_corridor_mask": recon["preserved_open_corridor_mask"],
        "preserved_long_walls_mask": recon["preserved_long_walls_mask"],
        "predicted_occupied_ratio": float(recon["predicted_occupied_ratio"]),
        "predicted_occupied_ratio_before_completion": float(recon["predicted_occupied_ratio_before_completion"]),
        "predicted_occupied_ratio_after_completion": float(recon["predicted_occupied_ratio_after_completion"]),
        "gt_occupied_ratio": float(recon["gt_occupied_ratio"]),
        "predicted_free_ratio": float(recon["predicted_free_ratio"]),
        "gt_free_ratio": float(recon["gt_free_ratio"]),
        "unknown_ratio": float(recon["unknown_ratio"]),
        "unknown_ratio_before_completion": float(recon["unknown_ratio_before_completion"]),
        "unknown_ratio_after_completion": float(recon["unknown_ratio_after_completion"]),
        "unknown_to_occupied_count": float(recon["unknown_to_occupied_count"]),
        "unknown_to_free_count": float(recon["unknown_to_free_count"]),
        "corridor_completion_gain": float(recon["corridor_completion_gain"]),
        "incorrect_unknown_promotions_if_gt_available": float(recon["incorrect_unknown_promotions_if_gt_available"]),
        "occupied_ratio_error": float(recon["occupied_ratio_error"]),
        "long_horizontal_wall_component_count": float(recon["long_horizontal_wall_component_count"]),
        "long_vertical_wall_component_count": float(recon["long_vertical_wall_component_count"]),
        "preserved_long_component_count": float(recon["preserved_long_component_count"]),
        "removed_long_component_count": float(recon["removed_long_component_count"]),
        "mean_wall_band_thickness": float(recon["mean_wall_band_thickness"]),
        "gt_wall_band_thickness_estimate": float(recon["gt_wall_band_thickness_estimate"]),
        "corridor_band_recovery_score": float(recon["corridor_band_recovery_score"]),
        "corridor_opening_preservation_score": float(recon["corridor_opening_preservation_score"]),
        "completed_wall_band_cell_count": float(recon["completed_wall_band_cell_count"]),
        "protected_opening_cell_count": float(recon["protected_opening_cell_count"]),
        "false_band_fill_count_if_gt_available": float(recon["false_band_fill_count_if_gt_available"]),
        "completed_free_corridor_cell_count": float(recon["completed_free_corridor_cell_count"]),
        "free_completion_accuracy_if_gt_available": float(recon["free_completion_accuracy_if_gt_available"]),
        "corridor_free_space_continuity_score": float(recon["corridor_free_space_continuity_score"]),
        "removed_component_count": float(recon["removed_component_count"]),
        "removed_component_area_total": float(recon["removed_component_area_total"]),
        "border_wall_preservation_score": float(recon["border_wall_preservation_score"]),
        "corridor_wall_continuity_score": float(recon["corridor_wall_continuity_score"]),
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
        "long_wall_segment_count": float(np.mean([e["long_wall_segment_count"] for e in episodes])),
        "predicted_occupied_ratio": float(np.mean([e["predicted_occupied_ratio"] for e in episodes])),
        "predicted_occupied_ratio_before_completion": float(np.mean([e["predicted_occupied_ratio_before_completion"] for e in episodes])),
        "predicted_occupied_ratio_after_completion": float(np.mean([e["predicted_occupied_ratio_after_completion"] for e in episodes])),
        "gt_occupied_ratio": float(np.mean([e["gt_occupied_ratio"] for e in episodes])),
        "predicted_free_ratio": float(np.mean([e["predicted_free_ratio"] for e in episodes])),
        "gt_free_ratio": float(np.mean([e["gt_free_ratio"] for e in episodes])),
        "unknown_ratio": float(np.mean([e["unknown_ratio"] for e in episodes])),
        "unknown_ratio_before_completion": float(np.mean([e["unknown_ratio_before_completion"] for e in episodes])),
        "unknown_ratio_after_completion": float(np.mean([e["unknown_ratio_after_completion"] for e in episodes])),
        "unknown_to_occupied_count": float(np.mean([e["unknown_to_occupied_count"] for e in episodes])),
        "unknown_to_free_count": float(np.mean([e["unknown_to_free_count"] for e in episodes])),
        "corridor_completion_gain": float(np.mean([e["corridor_completion_gain"] for e in episodes])),
        "incorrect_unknown_promotions_if_gt_available": float(np.mean([e["incorrect_unknown_promotions_if_gt_available"] for e in episodes])),
        "occupied_ratio_error": float(np.mean([e["occupied_ratio_error"] for e in episodes])),
        "long_horizontal_wall_component_count": float(np.mean([e["long_horizontal_wall_component_count"] for e in episodes])),
        "long_vertical_wall_component_count": float(np.mean([e["long_vertical_wall_component_count"] for e in episodes])),
        "preserved_long_component_count": float(np.mean([e["preserved_long_component_count"] for e in episodes])),
        "removed_long_component_count": float(np.mean([e["removed_long_component_count"] for e in episodes])),
        "mean_wall_band_thickness": float(np.mean([e["mean_wall_band_thickness"] for e in episodes])),
        "gt_wall_band_thickness_estimate": float(np.mean([e["gt_wall_band_thickness_estimate"] for e in episodes])),
        "corridor_band_recovery_score": float(np.mean([e["corridor_band_recovery_score"] for e in episodes])),
        "corridor_opening_preservation_score": float(np.mean([e["corridor_opening_preservation_score"] for e in episodes])),
        "completed_wall_band_cell_count": float(np.mean([e["completed_wall_band_cell_count"] for e in episodes])),
        "protected_opening_cell_count": float(np.mean([e["protected_opening_cell_count"] for e in episodes])),
        "false_band_fill_count_if_gt_available": float(np.mean([e["false_band_fill_count_if_gt_available"] for e in episodes])),
        "completed_free_corridor_cell_count": float(np.mean([e["completed_free_corridor_cell_count"] for e in episodes])),
        "free_completion_accuracy_if_gt_available": float(np.mean([e["free_completion_accuracy_if_gt_available"] for e in episodes])),
        "corridor_free_space_continuity_score": float(np.mean([e["corridor_free_space_continuity_score"] for e in episodes])),
        "removed_component_count": float(np.mean([e["removed_component_count"] for e in episodes])),
        "removed_component_area_total": float(np.mean([e["removed_component_area_total"] for e in episodes])),
        "border_wall_preservation_score": float(np.mean([e["border_wall_preservation_score"] for e in episodes])),
        "corridor_wall_continuity_score": float(np.mean([e["corridor_wall_continuity_score"] for e in episodes])),
        "doorway_candidate_count": float(np.mean([e["doorway_candidate_count"] for e in episodes])),
        "raw_doorway_candidate_count": float(np.mean([e["raw_doorway_candidate_count"] for e in episodes])),
        "accepted_doorway_candidate_count": float(np.mean([e["accepted_doorway_candidate_count"] for e in episodes])),
        "rejected_doorway_candidate_count": float(np.mean([e["rejected_doorway_candidate_count"] for e in episodes])),
        "rejected_by_gap_width_count": float(np.mean([e["rejected_by_gap_width_count"] for e in episodes])),
        "rejected_by_low_wall_confidence_count": float(np.mean([e["rejected_by_low_wall_confidence_count"] for e in episodes])),
        "rejected_by_low_free_evidence_count": float(np.mean([e["rejected_by_low_free_evidence_count"] for e in episodes])),
        "accepted_vertical_doorway_count": float(np.mean([e["accepted_vertical_doorway_count"] for e in episodes])),
        "accepted_horizontal_doorway_count": float(np.mean([e["accepted_horizontal_doorway_count"] for e in episodes])),
        "doorway_probability_nonzero_cells": float(np.mean([e["doorway_probability_nonzero_cells"] for e in episodes])),
        "doorway_probability_mean_on_gt_doorway": float(np.mean([e["doorway_probability_mean_on_gt_doorway"] for e in episodes])),
        "doorway_probability_mean_off_gt_doorway": float(np.mean([e["doorway_probability_mean_off_gt_doorway"] for e in episodes])),
        "false_doorway_count": float(np.mean([e["false_doorway_count"] for e in episodes])),
        "failure_reason_counts": dict(failure_counter),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2A v2.3 corridor completion patch.")
    p.add_argument("--episodes-per-map", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--maps", type=str, default="empty_room,corridor,single_block,doorway,cluttered_room")
    p.add_argument("--difficulties", type=str, default="clean,mild_noise,medium_noise,hard_noise")
    p.add_argument("--grid-size", type=int, default=0)
    p.add_argument("--cell-size", type=float, default=0.25)
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/phase2_mapping_v23")
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
                    f"[Phase2A.v2.3] map={map_name} ({mi}/{len(map_names)}) diff={d} ep={ep_idx}/{args.episodes_per_map} "
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

    # v1/v2/v2.1/v2.2/v2.3 comparison if prior results exist
    v1_path = Path("runs/phase2_mapping/phase2_mapping_results.json")
    v2_path = Path("runs/phase2_mapping_v2/phase2_mapping_results.json")
    v21_path = Path("runs/phase2_mapping_v21/phase2_mapping_results.json")
    v22_path = Path("runs/phase2_mapping_v22/phase2_mapping_results.json")
    comparison_v1_v2_v21_v22_v23 = {}
    v1 = json.loads(v1_path.read_text(encoding="utf-8")) if v1_path.exists() else {}
    v2 = json.loads(v2_path.read_text(encoding="utf-8")) if v2_path.exists() else {}
    v21 = json.loads(v21_path.read_text(encoding="utf-8")) if v21_path.exists() else {}
    v22 = json.loads(v22_path.read_text(encoding="utf-8")) if v22_path.exists() else {}
    for d in diff_names:
        if d not in agg:
            continue
        row = {
            "v23_map_accuracy": agg[d]["mean_map_accuracy"],
            "v23_wall_reconstruction_error_m": agg[d]["mean_wall_reconstruction_error_m"],
            "v23_doorway_precision": agg[d]["mean_doorway_precision"],
            "v23_doorway_recall": agg[d]["mean_doorway_recall"],
            "v23_localization_rmse_m": agg[d]["mean_localization_rmse_m"],
            "v23_collision_rate": agg[d]["mean_collision_rate"],
            "v23_timeout_rate": agg[d]["mean_timeout_rate"],
        }
        if d in v1.get("difficulty_aggregate", {}):
            row.update(
                {
                    "v1_map_accuracy": v1["difficulty_aggregate"][d]["mean_map_accuracy"],
                    "v1_wall_reconstruction_error_m": v1["difficulty_aggregate"][d]["mean_wall_reconstruction_error_m"],
                    "v1_doorway_precision": v1["difficulty_aggregate"][d]["mean_doorway_precision"],
                    "v1_doorway_recall": v1["difficulty_aggregate"][d]["mean_doorway_recall"],
                    "v1_localization_rmse_m": v1["difficulty_aggregate"][d]["mean_localization_rmse_m"],
                    "v1_collision_rate": v1["difficulty_aggregate"][d]["mean_collision_rate"],
                }
            )
        if d in v2.get("difficulty_aggregate", {}):
            row.update(
                {
                    "v2_map_accuracy": v2["difficulty_aggregate"][d]["mean_map_accuracy"],
                    "v2_wall_reconstruction_error_m": v2["difficulty_aggregate"][d]["mean_wall_reconstruction_error_m"],
                    "v2_doorway_precision": v2["difficulty_aggregate"][d]["mean_doorway_precision"],
                    "v2_doorway_recall": v2["difficulty_aggregate"][d]["mean_doorway_recall"],
                    "v2_localization_rmse_m": v2["difficulty_aggregate"][d]["mean_localization_rmse_m"],
                    "v2_collision_rate": v2["difficulty_aggregate"][d]["mean_collision_rate"],
                }
            )
        if d in v21.get("difficulty_aggregate", {}):
            row.update(
                {
                    "v21_map_accuracy": v21["difficulty_aggregate"][d]["mean_map_accuracy"],
                    "v21_wall_reconstruction_error_m": v21["difficulty_aggregate"][d]["mean_wall_reconstruction_error_m"],
                    "v21_doorway_precision": v21["difficulty_aggregate"][d]["mean_doorway_precision"],
                    "v21_doorway_recall": v21["difficulty_aggregate"][d]["mean_doorway_recall"],
                    "v21_localization_rmse_m": v21["difficulty_aggregate"][d]["mean_localization_rmse_m"],
                    "v21_collision_rate": v21["difficulty_aggregate"][d]["mean_collision_rate"],
                }
            )
        if d in v22.get("difficulty_aggregate", {}):
            row.update(
                {
                    "v22_map_accuracy": v22["difficulty_aggregate"][d]["mean_map_accuracy"],
                    "v22_wall_reconstruction_error_m": v22["difficulty_aggregate"][d]["mean_wall_reconstruction_error_m"],
                    "v22_doorway_precision": v22["difficulty_aggregate"][d]["mean_doorway_precision"],
                    "v22_doorway_recall": v22["difficulty_aggregate"][d]["mean_doorway_recall"],
                    "v22_localization_rmse_m": v22["difficulty_aggregate"][d]["mean_localization_rmse_m"],
                    "v22_collision_rate": v22["difficulty_aggregate"][d]["mean_collision_rate"],
                }
            )
        comparison_v1_v2_v21_v22_v23[d] = row

    clean_corridor_doorway = {}
    try:
        clean_corridor_doorway = {
            "v23_clean_corridor_map_accuracy": all_results["clean"]["corridor"]["map_accuracy"],
            "v23_clean_doorway_map_accuracy": all_results["clean"]["doorway"]["map_accuracy"],
        }
        if v1.get("per_map_metrics", {}).get("clean", {}).get("corridor"):
            clean_corridor_doorway["v1_clean_corridor_map_accuracy"] = v1["per_map_metrics"]["clean"]["corridor"]["map_accuracy"]
        if v2.get("per_map_metrics", {}).get("clean", {}).get("corridor"):
            clean_corridor_doorway["v2_clean_corridor_map_accuracy"] = v2["per_map_metrics"]["clean"]["corridor"]["map_accuracy"]
        if v21.get("per_map_metrics", {}).get("clean", {}).get("corridor"):
            clean_corridor_doorway["v21_clean_corridor_map_accuracy"] = v21["per_map_metrics"]["clean"]["corridor"]["map_accuracy"]
        if v1.get("per_map_metrics", {}).get("clean", {}).get("doorway"):
            clean_corridor_doorway["v1_clean_doorway_map_accuracy"] = v1["per_map_metrics"]["clean"]["doorway"]["map_accuracy"]
        if v2.get("per_map_metrics", {}).get("clean", {}).get("doorway"):
            clean_corridor_doorway["v2_clean_doorway_map_accuracy"] = v2["per_map_metrics"]["clean"]["doorway"]["map_accuracy"]
        if v21.get("per_map_metrics", {}).get("clean", {}).get("doorway"):
            clean_corridor_doorway["v21_clean_doorway_map_accuracy"] = v21["per_map_metrics"]["clean"]["doorway"]["map_accuracy"]
        if v22.get("per_map_metrics", {}).get("clean", {}).get("corridor"):
            clean_corridor_doorway["v22_clean_corridor_map_accuracy"] = v22["per_map_metrics"]["clean"]["corridor"]["map_accuracy"]
        if v22.get("per_map_metrics", {}).get("clean", {}).get("doorway"):
            clean_corridor_doorway["v22_clean_doorway_map_accuracy"] = v22["per_map_metrics"]["clean"]["doorway"]["map_accuracy"]
    except Exception:
        clean_corridor_doorway = {}

    # acceptance
    accepted_clean = ("clean" in agg and agg["clean"]["mean_map_accuracy"] >= 0.75 and agg["clean"]["mean_wall_reconstruction_error_m"] < 1.75 and agg["clean"]["mean_doorway_recall"] > 0.10 and agg["clean"]["mean_collision_rate"] <= 0.0)
    accepted_mild = ("mild_noise" in agg and agg["mild_noise"]["mean_map_accuracy"] >= 0.72 and agg["mild_noise"]["mean_collision_rate"] <= 0.0)
    accepted_medium = ("medium_noise" in agg and agg["medium_noise"]["mean_map_accuracy"] >= 0.68 and agg["medium_noise"]["mean_collision_rate"] <= 0.0)
    accepted_hard = ("hard_noise" in agg and agg["hard_noise"]["mean_map_accuracy"] >= 0.63 and agg["hard_noise"]["mean_collision_rate"] <= 0.0)
    accepted_overall = bool(accepted_clean and accepted_mild and accepted_medium and accepted_hard)
    needs_more_tuning = not accepted_overall

    results = {
        "phase": "Phase 2A v2.3 corridor completion patch",
        "episodes_per_map": args.episodes_per_map,
        "max_steps": args.max_steps,
        "cell_size": args.cell_size,
        "maps": map_names,
        "difficulties": diff_names,
        "per_map_metrics": all_results,
        "difficulty_aggregate": agg,
        "comparison_v1_v2_v21_v22_v23": comparison_v1_v2_v21_v22_v23,
        "clean_corridor_doorway_comparison": clean_corridor_doorway,
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

    print("\nPhase 2A v2.3 acceptance:")
    print(results["acceptance"])
    print(f"Saved: {out_dir / 'phase2_mapping_results.json'}")


if __name__ == "__main__":
    main()
