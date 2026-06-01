from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import DEFAULT_CELL_SIZE, DEFAULT_HEIGHT_M, DEFAULT_WIDTH_M


@dataclass
class EnvironmentMap:
    name: str
    cell_size: float
    width_m: float
    height_m: float
    ground_truth_occupancy: np.ndarray
    wall_mask: np.ndarray
    doorway_mask: np.ndarray
    free_space_mask: np.ndarray
    start_pose: Tuple[float, float, float]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.ground_truth_occupancy.shape


def world_to_cell(x_m: float, y_m: float, cell_size: float, width_cells: int, height_cells: int) -> Tuple[int, int]:
    cx = int(np.clip(np.floor(x_m / cell_size), 0, width_cells - 1))
    cy = int(np.clip(np.floor(y_m / cell_size), 0, height_cells - 1))
    return cx, cy


def _base_grid(width_m: float, height_m: float, cell_size: float) -> np.ndarray:
    width_cells = int(round(width_m / cell_size))
    height_cells = int(round(height_m / cell_size))
    occ = np.zeros((height_cells, width_cells), dtype=bool)
    occ[0, :] = True
    occ[-1, :] = True
    occ[:, 0] = True
    occ[:, -1] = True
    return occ


def _compute_wall_mask(occ: np.ndarray) -> np.ndarray:
    h, w = occ.shape
    wall = np.zeros_like(occ, dtype=bool)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not occ[y, x]:
                continue
            if not (occ[y - 1, x] and occ[y + 1, x] and occ[y, x - 1] and occ[y, x + 1]):
                wall[y, x] = True
    wall[0, :] = occ[0, :]
    wall[-1, :] = occ[-1, :]
    wall[:, 0] = occ[:, 0]
    wall[:, -1] = occ[:, -1]
    return wall


def _map_empty_room(cell_size: float) -> EnvironmentMap:
    occ = _base_grid(DEFAULT_WIDTH_M, DEFAULT_HEIGHT_M, cell_size)
    wall = _compute_wall_mask(occ)
    doorway = np.zeros_like(occ, dtype=bool)
    free = ~occ
    return EnvironmentMap(
        name="empty_room",
        cell_size=cell_size,
        width_m=DEFAULT_WIDTH_M,
        height_m=DEFAULT_HEIGHT_M,
        ground_truth_occupancy=occ,
        wall_mask=wall,
        doorway_mask=doorway,
        free_space_mask=free,
        start_pose=(2.0, 2.0, 0.0),
    )


def _map_corridor(cell_size: float) -> EnvironmentMap:
    occ = _base_grid(DEFAULT_WIDTH_M, DEFAULT_HEIGHT_M, cell_size)
    h, w = occ.shape
    # Build long top/bottom corridor walls with an opening notch.
    top_band = int(round(2.0 / cell_size))
    bottom_band = int(round(6.0 / cell_size))
    occ[top_band, 1 : w - 1] = True
    occ[bottom_band, 1 : w - 1] = True
    # Opening near right side.
    gap_x0 = int(round(9.5 / cell_size))
    gap_x1 = int(round(10.8 / cell_size))
    occ[top_band, gap_x0:gap_x1] = False
    occ[bottom_band, gap_x0:gap_x1] = False
    wall = _compute_wall_mask(occ)
    doorway = np.zeros_like(occ, dtype=bool)
    doorway[top_band - 1 : top_band + 2, gap_x0:gap_x1] = True
    doorway[bottom_band - 1 : bottom_band + 2, gap_x0:gap_x1] = True
    free = ~occ
    return EnvironmentMap(
        name="corridor",
        cell_size=cell_size,
        width_m=DEFAULT_WIDTH_M,
        height_m=DEFAULT_HEIGHT_M,
        ground_truth_occupancy=occ,
        wall_mask=wall,
        doorway_mask=doorway,
        free_space_mask=free,
        start_pose=(1.2, 4.0, 0.0),
    )


def _map_single_block(cell_size: float) -> EnvironmentMap:
    occ = _base_grid(DEFAULT_WIDTH_M, DEFAULT_HEIGHT_M, cell_size)
    h, w = occ.shape
    cx = w // 2
    cy = h // 2
    bx = int(round(1.5 / cell_size))
    by = int(round(1.1 / cell_size))
    occ[cy - by : cy + by, cx - bx : cx + bx] = True
    wall = _compute_wall_mask(occ)
    doorway = np.zeros_like(occ, dtype=bool)
    free = ~occ
    return EnvironmentMap(
        name="single_block",
        cell_size=cell_size,
        width_m=DEFAULT_WIDTH_M,
        height_m=DEFAULT_HEIGHT_M,
        ground_truth_occupancy=occ,
        wall_mask=wall,
        doorway_mask=doorway,
        free_space_mask=free,
        start_pose=(2.0, 2.0, 0.0),
    )


def _map_doorway(cell_size: float) -> EnvironmentMap:
    occ = _base_grid(DEFAULT_WIDTH_M, DEFAULT_HEIGHT_M, cell_size)
    h, w = occ.shape
    split_x = int(round(6.0 / cell_size))
    occ[1 : h - 1, split_x] = True
    door_y0 = int(round(3.0 / cell_size))
    door_y1 = int(round(5.0 / cell_size))
    occ[door_y0:door_y1, split_x] = False
    wall = _compute_wall_mask(occ)
    doorway = np.zeros_like(occ, dtype=bool)
    doorway[door_y0 - 1 : door_y1 + 1, split_x - 1 : split_x + 2] = True
    free = ~occ
    return EnvironmentMap(
        name="doorway",
        cell_size=cell_size,
        width_m=DEFAULT_WIDTH_M,
        height_m=DEFAULT_HEIGHT_M,
        ground_truth_occupancy=occ,
        wall_mask=wall,
        doorway_mask=doorway,
        free_space_mask=free,
        start_pose=(2.2, 4.0, 0.0),
    )


def _map_cluttered_room(cell_size: float, seed: int) -> EnvironmentMap:
    rng = np.random.default_rng(seed)
    occ = _base_grid(DEFAULT_WIDTH_M, DEFAULT_HEIGHT_M, cell_size)
    h, w = occ.shape
    num_blocks = 14
    for _ in range(num_blocks):
        bw = int(rng.integers(2, 6))
        bh = int(rng.integers(2, 5))
        x0 = int(rng.integers(2, max(3, w - bw - 2)))
        y0 = int(rng.integers(2, max(3, h - bh - 2)))
        occ[y0 : y0 + bh, x0 : x0 + bw] = True
    # Keep a rough center corridor.
    occ[h // 2 - 1 : h // 2 + 2, 2 : w - 2] = False
    wall = _compute_wall_mask(occ)
    doorway = np.zeros_like(occ, dtype=bool)
    free = ~occ
    return EnvironmentMap(
        name="cluttered_room",
        cell_size=cell_size,
        width_m=DEFAULT_WIDTH_M,
        height_m=DEFAULT_HEIGHT_M,
        ground_truth_occupancy=occ,
        wall_mask=wall,
        doorway_mask=doorway,
        free_space_mask=free,
        start_pose=(1.5, 4.0, 0.0),
    )


def get_environment(name: str, cell_size: float = DEFAULT_CELL_SIZE, seed: int = 0) -> EnvironmentMap:
    name = name.strip().lower()
    if name == "empty_room":
        return _map_empty_room(cell_size)
    if name == "corridor":
        return _map_corridor(cell_size)
    if name == "single_block":
        return _map_single_block(cell_size)
    if name == "doorway":
        return _map_doorway(cell_size)
    if name == "cluttered_room":
        return _map_cluttered_room(cell_size, seed=seed)
    raise ValueError(f"Unknown map: {name}")
