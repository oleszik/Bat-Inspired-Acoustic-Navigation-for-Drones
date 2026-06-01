from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from .mapping_utils import (
        DIFFICULTY_PRESETS,
        SECTOR_NAMES,
        apply_action,
        build_gt_grids,
        compute_wall_reconstruction_error,
        doorway_precision_recall,
        make_maps,
        make_output_dirs,
        predict_collision,
        sample_free_pose,
        simulate_echo_observation,
    )
except ImportError:  # pragma: no cover
    from simulation.phase2_mapping.mapping_utils import (
        DIFFICULTY_PRESETS,
        SECTOR_NAMES,
        apply_action,
        build_gt_grids,
        compute_wall_reconstruction_error,
        doorway_precision_recall,
        make_maps,
        make_output_dirs,
        predict_collision,
        sample_free_pose,
        simulate_echo_observation,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _grid_shape(width: float, height: float, cell_size: float) -> Tuple[int, int]:
    nx = int(math.ceil(width / cell_size))
    ny = int(math.ceil(height / cell_size))
    return ny, nx


def _world_to_cell(x: float, y: float, cell_size: float, nx: int, ny: int) -> Tuple[int, int]:
    cx = int(np.clip(math.floor(x / cell_size), 0, nx - 1))
    cy = int(np.clip(math.floor(y / cell_size), 0, ny - 1))
    return cx, cy


def _cast_update(
    occ_logodds: np.ndarray,
    wall_prob: np.ndarray,
    conf: np.ndarray,
    x: float,
    y: float,
    heading: float,
    angle_offset: float,
    distance: float,
    intensity: float,
    cell_size: float,
) -> None:
    ny, nx = occ_logodds.shape
    angle = heading + angle_offset
    step = max(0.04, cell_size * 0.6)
    t = 0.0
    # free-space along ray
    while t < max(0.0, distance - step):
        px = x + t * math.cos(angle)
        py = y + t * math.sin(angle)
        cx, cy = _world_to_cell(px, py, cell_size, nx, ny)
        occ_logodds[cy, cx] -= 0.06
        conf[cy, cx] += 0.015
        t += step
    # endpoint wall belief
    if distance < 2.95:
        px = x + distance * math.cos(angle)
        py = y + distance * math.sin(angle)
        cx, cy = _world_to_cell(px, py, cell_size, nx, ny)
        occ_logodds[cy, cx] += 0.25 + 0.25 * intensity
        wall_prob[cy, cx] += 0.06 + 0.06 * intensity
        conf[cy, cx] += 0.035


def _estimate_doorway_prob(wall_prob: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    # Gap heuristic: likely doorway where neighboring wall support exists but center wall is weak.
    wp = np.clip(wall_prob, 0.0, 1.0)
    ny, nx = wp.shape
    dmap = np.zeros_like(wp)
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            c = wp[y, x]
            horiz = min(wp[y, x - 1], wp[y, x + 1]) - c
            vert = min(wp[y - 1, x], wp[y + 1, x]) - c
            gap = max(horiz, vert)
            if gap > 0.2:
                dmap[y, x] = gap
    # Confidence suppresses hallucinated doorway cells.
    conf_norm = np.clip(confidence / (confidence.max() + 1e-6), 0.0, 1.0)
    return np.clip(dmap * (0.4 + 0.6 * conf_norm), 0.0, 1.0)


def _choose_safe_action(
    m,
    x: float,
    y: float,
    heading: float,
    obs: Dict[str, Dict[str, float]],
    robot_radius: float,
    safety_margin: float,
) -> str:
    front = obs["front"]["distance"]
    fl = obs["front_left"]["distance"]
    fr = obs["front_right"]["distance"]
    left = obs["left"]["distance"]
    right = obs["right"]["distance"]

    def _safe_forward(dist: float) -> bool:
        return (front > dist) and (fl > 0.15) and (fr > 0.15) and (
            not predict_collision(m, x, y, heading, dist, robot_radius, safety_margin)
        )

    # Safety-first movement policy, secondary for mapping coverage.
    if _safe_forward(0.08):
        return "MOVE_FORWARD_SLOW"
    if _safe_forward(0.03):
        return "PROBE_FORWARD"
    # Turn toward clearer side.
    if left >= right:
        return "TURN_LEFT"
    return "TURN_RIGHT"


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

    occ_logodds = np.zeros((ny, nx), dtype=np.float32)
    wall_prob = np.zeros((ny, nx), dtype=np.float32)
    doorway_prob = np.zeros((ny, nx), dtype=np.float32)
    confidence = np.zeros((ny, nx), dtype=np.float32)

    x, y = sample_free_pose(m, rng, robot_radius=robot_radius)
    heading = float(rng.uniform(-math.pi, math.pi))
    est_x, est_y, est_heading = x, y, heading

    action_counts = Counter()
    true_path = [(x, y)]
    est_path = [(est_x, est_y)]
    loc_err = []
    collisions = 0

    sector_offsets = {
        "left": math.radians(90.0),
        "front_left": math.radians(40.0),
        "front": 0.0,
        "front_right": math.radians(-40.0),
        "right": math.radians(-90.0),
    }

    for _ in range(max_steps):
        obs = simulate_echo_observation(m, x, y, heading, rng, preset)

        # Inverse sensor update into mapping grids.
        for sec in SECTOR_NAMES:
            _cast_update(
                occ_logodds,
                wall_prob,
                confidence,
                est_x,
                est_y,
                est_heading,
                sector_offsets[sec],
                float(obs[sec]["distance"]),
                float(obs[sec]["intensity"]),
                cell_size,
            )

        doorway_prob = _estimate_doorway_prob(wall_prob, confidence)

        action = _choose_safe_action(m, x, y, heading, obs, robot_radius, safety_margin)
        action_counts[action] += 1
        nx_t, ny_t, nh_t, moved, collided = apply_action(x, y, heading, action, m, robot_radius=robot_radius, turn_deg=15.0)
        if collided:
            collisions += 1
            break

        # True pose
        x, y, heading = nx_t, ny_t, nh_t

        # Dead-reckoning estimate with difficulty-dependent drift.
        drift = preset["pose_drift_std"]
        est_x = est_x + moved * math.cos(est_heading) + rng.normal(0.0, drift)
        est_y = est_y + moved * math.sin(est_heading) + rng.normal(0.0, drift)
        if action == "TURN_LEFT":
            est_heading = est_heading + math.radians(15.0) + rng.normal(0.0, 0.5 * drift)
        elif action == "TURN_RIGHT":
            est_heading = est_heading - math.radians(15.0) + rng.normal(0.0, 0.5 * drift)
        est_heading = ((est_heading + math.pi) % (2.0 * math.pi)) - math.pi

        true_path.append((x, y))
        est_path.append((est_x, est_y))
        loc_err.append(math.hypot(est_x - x, est_y - y))

    occ_prob = _sigmoid(occ_logodds)
    occ_pred = (occ_prob >= 0.5).astype(np.uint8)
    wall_pred = (wall_prob >= 0.45).astype(np.uint8)
    door_pred = (doorway_prob >= 0.30).astype(np.uint8)

    gt_occ = gt["occupancy"].astype(np.uint8)
    gt_wall = gt["wall"].astype(np.uint8)
    gt_door = gt["doorway"].astype(np.uint8)

    map_acc = float((occ_pred == gt_occ).mean())
    wall_err = compute_wall_reconstruction_error(wall_pred, gt_wall, cell_size)
    door_metrics = doorway_precision_recall(door_pred, gt_door) if m.name == "doorway" else {"precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}

    loc_rmse = float(np.sqrt(np.mean(np.square(loc_err)))) if loc_err else 0.0

    conf_norm = np.clip(confidence / (confidence.max() + 1e-6), 0.0, 1.0)
    correctness = (occ_pred == gt_occ).astype(np.float32)
    conf_quality = float((conf_norm * correctness).sum() / max(1e-6, conf_norm.sum()))

    return {
        "map_accuracy": map_acc,
        "wall_reconstruction_error_m": float(wall_err),
        "doorway_precision": float(door_metrics["precision"]),
        "doorway_recall": float(door_metrics["recall"]),
        "localization_rmse_m": loc_rmse,
        "map_confidence_quality": conf_quality,
        "collision_count": int(collisions),
        "timeout": int(collisions == 0),
        "steps": int(len(true_path) - 1),
        "action_counts": dict(action_counts),
        "final_observed_proxy": float((confidence > 0.05).mean()),
        "true_path": true_path,
        "est_path": est_path,
        "pred_occupancy": occ_pred,
        "gt_occupancy": gt_occ,
        "wall_prob": np.clip(wall_prob, 0.0, 1.0),
        "doorway_prob": doorway_prob,
        "confidence": conf_norm,
    }


def summarize_episodes(episodes: List[Dict[str, object]]) -> Dict[str, object]:
    action_counter = Counter()
    for e in episodes:
        action_counter.update(e["action_counts"])  # type: ignore[arg-type]
    total_actions = sum(action_counter.values())
    action_dist = {
        a: {"count": int(action_counter.get(a, 0)), "rate": float(action_counter.get(a, 0) / max(1, total_actions))}
        for a in ["MOVE_FORWARD_FAST", "MOVE_FORWARD_SLOW", "PROBE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "SLOW_DOWN_AND_RESAMPLE", "STOP_OR_REVERSE"]
    }
    return {
        "map_accuracy": float(np.mean([e["map_accuracy"] for e in episodes])),
        "wall_reconstruction_error_m": float(np.mean([e["wall_reconstruction_error_m"] for e in episodes])),
        "doorway_precision": float(np.mean([e["doorway_precision"] for e in episodes])),
        "doorway_recall": float(np.mean([e["doorway_recall"] for e in episodes])),
        "localization_rmse_m": float(np.mean([e["localization_rmse_m"] for e in episodes])),
        "map_confidence_quality": float(np.mean([e["map_confidence_quality"] for e in episodes])),
        "collision_rate": float(np.mean([1.0 if e["collision_count"] > 0 else 0.0 for e in episodes])),
        "timeout_rate": float(np.mean([e["timeout"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "mean_observed_proxy": float(np.mean([e["final_observed_proxy"] for e in episodes])),
        "action_distribution": action_dist,
    }


def _save_episode_plots(out_dir: Path, map_name: str, difficulty: str, ep: Dict[str, object]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=130)
    axes = axes.flatten()
    axes[0].imshow(ep["pred_occupancy"], cmap="gray_r")
    axes[0].set_title("Pred Occupancy")
    axes[1].imshow(ep["gt_occupancy"], cmap="gray_r")
    axes[1].set_title("GT Occupancy")
    axes[2].imshow(ep["wall_prob"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Wall Probability")
    axes[3].imshow(ep["doorway_prob"], cmap="viridis", vmin=0.0, vmax=1.0)
    axes[3].set_title("Doorway Probability")
    axes[4].imshow(ep["confidence"], cmap="Blues", vmin=0.0, vmax=1.0)
    axes[4].set_title("Confidence")
    tpath = np.asarray(ep["true_path"], dtype=float)
    epath = np.asarray(ep["est_path"], dtype=float)
    axes[5].plot(tpath[:, 0], tpath[:, 1], label="true")
    axes[5].plot(epath[:, 0], epath[:, 1], label="estimated")
    axes[5].set_title("Trajectory")
    axes[5].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / f"{map_name}_{difficulty}_sample.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2A classical acoustic mapping baseline.")
    parser.add_argument("--episodes-per-map", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--maps", type=str, default="empty_room,corridor,single_block,doorway,cluttered_room")
    parser.add_argument("--difficulties", type=str, default="clean,mild_noise,medium_noise,hard_noise")
    parser.add_argument("--grid-size", type=int, default=0, help="Reserved for future use.")
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--output-dir", type=str, default="runs/phase2_mapping")
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.output_dir)
    make_output_dirs(out_dir)

    maps = {m.name: m for m in make_maps()}
    map_names = [m.strip() for m in args.maps.split(",") if m.strip()]
    diff_names = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    for d in diff_names:
        if d not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {d}")
    for m in map_names:
        if m not in maps:
            raise ValueError(f"Unknown map: {m}")

    rng = np.random.default_rng(args.seed)
    all_results: Dict[str, Dict[str, object]] = defaultdict(dict)

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
                    f"[Phase2A] map={map_name} ({mi}/{len(map_names)}) diff={d} ep={ep_idx}/{args.episodes_per_map} "
                    f"elapsed={elapsed:.1f}s mean_acc={np.mean(acc_so_far):.3f} mean_loc_rmse={np.mean(loc_so_far):.3f}m"
                )
            summary = summarize_episodes(episodes)
            all_results[d][map_name] = summary
            if args.save_plots:
                _save_episode_plots(out_dir, map_name, d, episodes[-1])

    # Aggregate by difficulty.
    difficulty_aggregate = {}
    for d in diff_names:
        vals = [all_results[d][m] for m in map_names]
        difficulty_aggregate[d] = {
            "mean_map_accuracy": float(np.mean([v["map_accuracy"] for v in vals])),
            "mean_wall_reconstruction_error_m": float(np.mean([v["wall_reconstruction_error_m"] for v in vals])),
            "mean_doorway_precision": float(np.mean([v["doorway_precision"] for v in vals])),
            "mean_doorway_recall": float(np.mean([v["doorway_recall"] for v in vals])),
            "mean_localization_rmse_m": float(np.mean([v["localization_rmse_m"] for v in vals])),
            "mean_map_confidence_quality": float(np.mean([v["map_confidence_quality"] for v in vals])),
            "mean_collision_rate": float(np.mean([v["collision_rate"] for v in vals])),
            "mean_timeout_rate": float(np.mean([v["timeout_rate"] for v in vals])),
        }

    # Acceptance.
    def _acc_ok(diff: str, thresh: float) -> bool:
        if diff not in difficulty_aggregate:
            return False
        return difficulty_aggregate[diff]["mean_map_accuracy"] >= thresh

    accepted_mapping_clean = _acc_ok("clean", 0.75)
    accepted_mapping_mild = _acc_ok("mild_noise", 0.72)
    accepted_mapping_medium = _acc_ok("medium_noise", 0.68)
    accepted_mapping_hard = _acc_ok("hard_noise", 0.63)
    all_collision_ok = all(v["mean_collision_rate"] <= 0.02 for v in difficulty_aggregate.values())
    accepted_mapping_overall = bool(
        accepted_mapping_clean and accepted_mapping_mild and accepted_mapping_medium and accepted_mapping_hard and all_collision_ok
    )
    needs_more_mapping_tuning = not accepted_mapping_overall

    results = {
        "phase": "Phase 2A classical acoustic mapping baseline",
        "episodes_per_map": args.episodes_per_map,
        "max_steps": args.max_steps,
        "cell_size": args.cell_size,
        "maps": map_names,
        "difficulties": diff_names,
        "per_map_metrics": all_results,
        "difficulty_aggregate": difficulty_aggregate,
        "acceptance": {
            "accepted_mapping_clean": bool(accepted_mapping_clean),
            "accepted_mapping_mild": bool(accepted_mapping_mild),
            "accepted_mapping_medium": bool(accepted_mapping_medium),
            "accepted_mapping_hard": bool(accepted_mapping_hard),
            "accepted_mapping_overall": bool(accepted_mapping_overall),
            "needs_more_mapping_tuning": bool(needs_more_mapping_tuning),
        },
    }

    (out_dir / "phase2_mapping_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "per_map_metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\nPhase 2A acceptance:")
    print(results["acceptance"])
    print(f"Saved: {out_dir / 'phase2_mapping_results.json'}")


if __name__ == "__main__":
    main()
