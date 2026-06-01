from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # pragma: no cover
    distance_transform_edt = None


RAY_MAX_RANGE = 3.0
RAY_STEP = 0.05

SECTOR_NAMES = ["left", "front_left", "front", "front_right", "right"]
SECTOR_OFFSETS_RAD = {
    "left": math.radians(90.0),
    "front_left": math.radians(40.0),
    "front": 0.0,
    "front_right": math.radians(-40.0),
    "right": math.radians(-90.0),
}

DIFFICULTY_PRESETS: Dict[str, Dict[str, float]] = {
    "clean": {
        "distance_noise_std": 0.0,
        "missed_echo_prob": 0.0,
        "false_echo_prob": 0.0,
        "echo_jitter_std": 0.0,
        "max_range_dropout_prob": 0.0,
        "obstacle_confidence_noise": 0.0,
        "pose_drift_std": 0.0,
    },
    "mild_noise": {
        "distance_noise_std": 0.03,
        "missed_echo_prob": 0.03,
        "false_echo_prob": 0.01,
        "echo_jitter_std": 0.02,
        "max_range_dropout_prob": 0.01,
        "obstacle_confidence_noise": 0.03,
        "pose_drift_std": 0.004,
    },
    "medium_noise": {
        "distance_noise_std": 0.08,
        "missed_echo_prob": 0.08,
        "false_echo_prob": 0.03,
        "echo_jitter_std": 0.05,
        "max_range_dropout_prob": 0.04,
        "obstacle_confidence_noise": 0.08,
        "pose_drift_std": 0.010,
    },
    "hard_noise": {
        "distance_noise_std": 0.13,
        "missed_echo_prob": 0.12,
        "false_echo_prob": 0.06,
        "echo_jitter_std": 0.08,
        "max_range_dropout_prob": 0.08,
        "obstacle_confidence_noise": 0.12,
        "pose_drift_std": 0.020,
    },
}


@dataclass
class Rect:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class MapDef:
    name: str
    width: float
    height: float
    obstacles: List[Rect]
    doorway_regions: Optional[List[Rect]] = None


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def point_in_rect(px: float, py: float, r: Rect) -> bool:
    return (r.xmin <= px <= r.xmax) and (r.ymin <= py <= r.ymax)


def point_in_obstacle(px: float, py: float, m: MapDef, robot_radius: float = 0.0) -> bool:
    if px < robot_radius or py < robot_radius or px > (m.width - robot_radius) or py > (m.height - robot_radius):
        return True
    for r in m.obstacles:
        rr = Rect(r.xmin - robot_radius, r.ymin - robot_radius, r.xmax + robot_radius, r.ymax + robot_radius)
        if point_in_rect(px, py, rr):
            return True
    return False


def raycast_distance(m: MapDef, x: float, y: float, angle: float, max_range: float = RAY_MAX_RANGE, step: float = RAY_STEP) -> float:
    t = 0.0
    while t <= max_range:
        px = x + t * math.cos(angle)
        py = y + t * math.sin(angle)
        if point_in_obstacle(px, py, m, robot_radius=0.0):
            return t
        t += step
    return max_range


def sector_true_distances(m: MapDef, x: float, y: float, heading: float, cone_half_rad: float = math.radians(12.0)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sec in SECTOR_NAMES:
        center = heading + SECTOR_OFFSETS_RAD[sec]
        vals = [
            raycast_distance(m, x, y, center - cone_half_rad),
            raycast_distance(m, x, y, center),
            raycast_distance(m, x, y, center + cone_half_rad),
        ]
        out[sec] = float(min(vals))
    return out


def apply_distance_noise(true_d: float, rng: np.random.Generator, preset: Dict[str, float]) -> float:
    d = true_d + rng.normal(0.0, preset["distance_noise_std"]) + rng.normal(0.0, preset["echo_jitter_std"])
    d = float(np.clip(d, 0.03, RAY_MAX_RANGE))
    if rng.random() < preset["max_range_dropout_prob"]:
        d = RAY_MAX_RANGE
    return d


def simulate_echo_observation(
    m: MapDef,
    x: float,
    y: float,
    heading: float,
    rng: np.random.Generator,
    preset: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    true_d = sector_true_distances(m, x, y, heading)
    obs: Dict[str, Dict[str, float]] = {}
    for sec, d in true_d.items():
        nd = apply_distance_noise(d, rng, preset)
        if rng.random() < preset["missed_echo_prob"]:
            nd = RAY_MAX_RANGE
        p_false = preset["false_echo_prob"]
        if nd >= RAY_MAX_RANGE and rng.random() < p_false:
            nd = float(rng.uniform(0.25, min(2.2, RAY_MAX_RANGE)))
        intensity = float(np.clip(1.0 / (1.0 + nd * nd) + rng.normal(0.0, 0.02 + preset["obstacle_confidence_noise"]), 0.0, 1.0))
        obs[sec] = {"distance": nd, "intensity": intensity, "true_distance": d}
    return obs


def forward_step_for_action(action: str) -> float:
    return {
        "MOVE_FORWARD_FAST": 0.20,
        "MOVE_FORWARD_SLOW": 0.08,
        "PROBE_FORWARD": 0.03,
    }.get(action, 0.0)


def apply_action(
    x: float,
    y: float,
    heading: float,
    action: str,
    m: MapDef,
    robot_radius: float = 0.12,
    turn_deg: float = 15.0,
) -> Tuple[float, float, float, float, bool]:
    if action == "TURN_LEFT":
        return x, y, wrap_angle(heading + math.radians(turn_deg)), 0.0, False
    if action == "TURN_RIGHT":
        return x, y, wrap_angle(heading - math.radians(turn_deg)), 0.0, False
    if action == "SLOW_DOWN_AND_RESAMPLE":
        return x, y, heading, 0.0, False
    dist = forward_step_for_action(action)
    if action == "STOP_OR_REVERSE":
        dist = -0.03
    nseg = max(2, int(abs(dist) / 0.01))
    nx, ny = x, y
    collided = False
    for i in range(1, nseg + 1):
        t = i / nseg
        px = x + t * dist * math.cos(heading)
        py = y + t * dist * math.sin(heading)
        if point_in_obstacle(px, py, m, robot_radius=robot_radius):
            collided = True
            break
        nx, ny = px, py
    moved = float(math.hypot(nx - x, ny - y))
    return nx, ny, heading, moved, collided


def predict_collision(
    m: MapDef,
    x: float,
    y: float,
    heading: float,
    dist: float,
    robot_radius: float,
    margin: float,
) -> bool:
    nseg = max(2, int(abs(dist) / 0.01))
    for i in range(1, nseg + 1):
        t = i / nseg
        px = x + t * dist * math.cos(heading)
        py = y + t * dist * math.sin(heading)
        if point_in_obstacle(px, py, m, robot_radius=robot_radius + margin):
            return True
    return False


def sample_free_pose(m: MapDef, rng: np.random.Generator, robot_radius: float = 0.12) -> Tuple[float, float]:
    for _ in range(2500):
        x = rng.uniform(0.5, m.width - 0.5)
        y = rng.uniform(0.5, m.height - 0.5)
        if not point_in_obstacle(x, y, m, robot_radius=robot_radius):
            return float(x), float(y)
    raise RuntimeError(f"Could not sample free pose in map {m.name}")


def make_maps() -> List[MapDef]:
    return [
        MapDef("empty_room", 10.0, 8.0, [], doorway_regions=[]),
        MapDef(
            "corridor",
            12.0,
            8.0,
            [Rect(0.0, 0.0, 12.0, 2.2), Rect(0.0, 5.8, 12.0, 8.0), Rect(5.5, 2.2, 6.5, 4.0)],
            doorway_regions=[],
        ),
        MapDef("single_block", 10.0, 8.0, [Rect(4.2, 2.8, 5.8, 5.2)], doorway_regions=[]),
        MapDef(
            "doorway",
            12.0,
            8.0,
            [Rect(5.7, 0.0, 6.3, 3.2), Rect(5.7, 4.8, 6.3, 8.0)],
            doorway_regions=[Rect(5.55, 3.2, 6.45, 4.8)],
        ),
        MapDef(
            "cluttered_room",
            12.0,
            9.0,
            [
                Rect(2.0, 2.0, 3.4, 3.2),
                Rect(4.5, 1.0, 6.2, 2.6),
                Rect(7.0, 2.5, 8.8, 4.2),
                Rect(3.0, 5.0, 4.8, 6.7),
                Rect(6.0, 5.7, 7.6, 7.8),
                Rect(9.0, 5.0, 10.8, 7.3),
            ],
            doorway_regions=[],
        ),
    ]


def build_gt_grids(m: MapDef, cell_size: float) -> Dict[str, np.ndarray]:
    nx = int(math.ceil(m.width / cell_size))
    ny = int(math.ceil(m.height / cell_size))
    occ = np.zeros((ny, nx), dtype=np.uint8)
    door = np.zeros((ny, nx), dtype=np.uint8)
    for iy in range(ny):
        for ix in range(nx):
            px = (ix + 0.5) * cell_size
            py = (iy + 0.5) * cell_size
            if point_in_obstacle(px, py, m, robot_radius=0.0):
                occ[iy, ix] = 1
            if m.doorway_regions:
                for dr in m.doorway_regions:
                    if point_in_rect(px, py, dr):
                        door[iy, ix] = 1
                        break
    wall = derive_wall_mask(occ)
    return {"occupancy": occ, "wall": wall, "doorway": door}


def derive_wall_mask(occ: np.ndarray) -> np.ndarray:
    ny, nx = occ.shape
    wall = np.zeros_like(occ, dtype=np.uint8)
    for y in range(ny):
        for x in range(nx):
            if occ[y, x] == 0:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxp, nyp = x + dx, y + dy
                if nxp < 0 or nyp < 0 or nxp >= nx or nyp >= ny or occ[nyp, nxp] == 0:
                    wall[y, x] = 1
                    break
    return wall


def compute_wall_reconstruction_error(pred_wall: np.ndarray, gt_wall: np.ndarray, cell_size: float) -> float:
    pred = pred_wall.astype(bool)
    gt = gt_wall.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if distance_transform_edt is not None:
        dt_gt = distance_transform_edt(~gt)
        dt_pred = distance_transform_edt(~pred)
        e1 = float(dt_gt[pred].mean()) if pred.any() else float(dt_gt.mean())
        e2 = float(dt_pred[gt].mean()) if gt.any() else float(dt_pred.mean())
        return 0.5 * (e1 + e2) * cell_size

    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float(max(len(pred_pts), len(gt_pts)) * cell_size)
    d1 = []
    for p in pred_pts:
        d = np.sqrt(((gt_pts - p) ** 2).sum(axis=1)).min()
        d1.append(d)
    d2 = []
    for g in gt_pts:
        d = np.sqrt(((pred_pts - g) ** 2).sum(axis=1)).min()
        d2.append(d)
    return float(0.5 * (np.mean(d1) + np.mean(d2)) * cell_size)


def doorway_precision_recall(pred_door: np.ndarray, gt_door: np.ndarray) -> Dict[str, float]:
    p = pred_door.astype(bool)
    g = gt_door.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def make_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
