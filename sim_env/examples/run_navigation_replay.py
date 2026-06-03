from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_env.acoustic_world import (  # noqa: E402
    AcousticAgent,
    AcousticSensor,
    choose_action_debug,
    dummy_predict_maps,
    get_environment,
    get_mapper_config,
    get_navigation_config,
    load_mapper_manifest,
    load_navigation_manifest,
    render_simulation_state,
)
from sim_env.acoustic_world.environments import world_to_cell  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step-by-step mapper-guided navigation replay debugger.")
    p.add_argument("--map", type=str, default="doorway")
    p.add_argument("--difficulty", type=str, default="clean")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument(
        "--policy-mode",
        type=str,
        default="simple",
        choices=[
            "simple",
            "accepted_v3_like",
            "stabilized_frontier",
            "route_committed",
            "coverage_sweep",
            "adaptive_hybrid",
        ],
    )
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--save-frames", action="store_true")
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--output-dir", type=str, default="sim_env/outputs")
    return p.parse_args()


def mark_observed(env, observed: np.ndarray, agent: AcousticAgent, sensor_obs: dict) -> None:
    h, w = env.shape
    for ang, dist in zip(sensor_obs["ray_angles_rad"], sensor_obs["ray_distances_m"]):
        t = 0.0
        while t <= float(dist):
            px = agent.x + t * np.cos(float(ang))
            py = agent.y + t * np.sin(float(ang))
            cx, cy = world_to_cell(px, py, env.cell_size, w, h)
            observed[cy, cx] = True
            t += max(0.04, 0.75 * env.cell_size)


def _save_gif(frame_paths: List[Path], gif_path: Path, duration_ms: int = 90) -> bool:
    if Image is None or not frame_paths:
        return False
    imgs = []
    for p in frame_paths:
        imgs.append(Image.open(p).convert("RGB"))
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for im in imgs:
        im.close()
    return True


def _compute_coverage(env, observed: np.ndarray) -> float:
    free_cells = env.free_space_mask.astype(bool)
    observed_free = np.logical_and(observed, free_cells)
    return float(observed_free.sum() / max(1, free_cells.sum()))


def _to_jsonable_target(target: Tuple[int, int] | None) -> List[int] | None:
    if target is None:
        return None
    return [int(target[0]), int(target[1])]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = get_environment(args.map, seed=args.seed)
    agent = AcousticAgent(*env.start_pose)
    sensor = AcousticSensor.default()

    mapper_manifest = load_mapper_manifest()
    nav_manifest = load_navigation_manifest()
    mapper_cfg = get_mapper_config(mapper_manifest)
    nav_cfg = get_navigation_config(nav_manifest)
    print(f"Loaded mapper manifest: {mapper_cfg.get('accepted_model_name')}")
    print(f"Loaded navigation manifest: {nav_cfg.get('accepted_model_name')}")

    observed = np.zeros(env.shape, dtype=bool)
    accepted_doorway_cells: List[Tuple[int, int]] = []
    rejected_doorway_cells: List[Tuple[int, int]] = []
    candidate_doorway_cells: List[Tuple[int, int]] = []
    frame_paths: List[Path] = []
    step_metrics: List[Dict[str, object]] = []
    action_counter: Counter[str] = Counter()
    visited_cell_counts: Dict[Tuple[int, int], int] = {}
    repeated_step_count = 0
    total_distance_traveled = 0.0
    collision_count = 0
    fake_doorway_approach_count = 0
    accepted_doorway_decision_count = 0
    rejected_doorway_decision_count = 0
    frontier_switch_count = 0
    frontier_commit_success_count = 0
    stagnation_event_count = 0
    stagnation_active = False
    prev_target: Tuple[int, int] | None = None
    prev_coverage = 0.0
    prev_confidence_mean = 0.0
    coverage_history: List[float] = []
    steps_targeting_unknown_frontier = 0
    steps_safe_forward_bias = 0
    high_confidence_revisit_count = 0
    nav_state: Dict[str, object] = {}

    out_root = Path(args.output_dir)
    run_name = f"navigation_replay_{args.map}_{args.difficulty}_{args.policy_mode}_seed{args.seed}"
    replay_dir = out_root / run_name
    frames_dir = replay_dir / "frames"
    replay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    start_cell = agent.cell(env)
    visited_cell_counts[start_cell] = 1

    for step in range(args.steps):
        obs = sensor.sense(env, agent.x, agent.y, agent.theta, args.difficulty, rng)
        mark_observed(env, observed, agent, obs)
        pred = dummy_predict_maps(env, observed, rng)
        coverage = _compute_coverage(env, observed)
        confidence_mean = float(np.mean(pred["confidence_map"]))
        confidence_gain = confidence_mean - prev_confidence_mean if step > 0 else confidence_mean

        action, target, dbg = choose_action_debug(
            env=env,
            agent=agent,
            sensor_obs=obs,
            predicted_maps=pred,
            policy_mode=args.policy_mode,
            nav_state=nav_state,
            coverage=coverage,
        )
        nav_state = dbg.get("nav_state", nav_state)
        action_counter[action] += 1

        if target is not None and not bool(observed[target[1], target[0]]):
            steps_targeting_unknown_frontier += 1
        if dbg.get("action_reason") in ("safe_forward_frontier_progress", "safe_forward_no_target"):
            steps_safe_forward_bias += 1
        if prev_target is not None and target is not None and prev_target != target:
            frontier_switch_count += 1
        if prev_target is not None and target is not None and prev_target == target and action in (
            "move_forward_slow",
            "probe_forward",
        ):
            frontier_commit_success_count += 1
        prev_target = target

        if dbg.get("doorway_candidate"):
            c = dbg.get("doorway_candidate_cell")
            if isinstance(c, tuple):
                candidate_doorway_cells.append(c)
                if dbg.get("doorway_accepted"):
                    accepted_doorway_cells.append(c)
                    accepted_doorway_decision_count += 1
                    y0, y1 = max(0, c[1] - 1), min(env.shape[0], c[1] + 2)
                    x0, x1 = max(0, c[0] - 1), min(env.shape[1], c[0] + 2)
                    local_true_door = env.doorway_mask[y0:y1, x0:x1]
                    if not bool(np.any(local_true_door)):
                        fake_doorway_approach_count += 1
                else:
                    rejected_doorway_cells.append(c)
                    rejected_doorway_decision_count += 1

        prev_pos = (agent.x, agent.y)
        agent.apply_action(env, action)
        dx = float(agent.x - prev_pos[0])
        dy = float(agent.y - prev_pos[1])
        step_distance = float(np.hypot(dx, dy))
        total_distance_traveled += step_distance
        if action in ("move_forward_slow", "probe_forward", "stop_or_reverse") and step_distance < 1e-6:
            collision_count += 1

        cell = agent.cell(env)
        visit_count = visited_cell_counts.get(cell, 0)
        if visit_count > 0:
            repeated_step_count += 1
            if confidence_mean > 0.70:
                high_confidence_revisit_count += 1
        visited_cell_counts[cell] = visit_count + 1
        repeated_cell_ratio = float(repeated_step_count / max(1, step + 1))

        coverage_gain = coverage - prev_coverage if step > 0 else coverage
        coverage_history.append(coverage)
        if step >= 20:
            old_coverage = coverage_history[step - 20]
            stagnating_now = (coverage - old_coverage) < 0.005
            if stagnating_now and not stagnation_active:
                stagnation_event_count += 1
                stagnation_active = True
            elif not stagnating_now:
                stagnation_active = False
        prev_coverage = coverage
        prev_confidence_mean = confidence_mean

        doorway_state = (
            "accepted"
            if bool(dbg.get("doorway_accepted"))
            else ("rejected" if bool(dbg.get("doorway_candidate")) else "none")
        )
        step_metrics.append(
            {
                "step_index": int(step),
                "policy_mode": str(args.policy_mode),
                "chosen_action": str(dbg.get("chosen_action", action)),
                "action_reason": str(dbg.get("action_reason", "")),
                "collision_risk": bool(dbg.get("collision_risk", False)),
                "coverage": float(coverage),
                "confidence_mean": float(confidence_mean),
                "confidence_gain": float(confidence_gain),
                "frontier_target": _to_jsonable_target(target),
                "doorway_candidate_count": int(dbg.get("doorway_candidate_count", 0)),
                "doorway_accepted_count": int(dbg.get("doorway_accepted_count", 0)),
                "doorway_rejected_count": int(dbg.get("doorway_rejected_count", 0)),
                "distance_traveled": float(step_distance),
                "repeated_cell_ratio": float(repeated_cell_ratio),
                "doorway_state": doorway_state,
                "confidence_gain_estimate": float(dbg.get("confidence_gain_estimate", 0.0)),
                "target_frontier_score": float(dbg.get("target_frontier_score", 0.0)),
                "stagnation_recovery_active": bool(dbg.get("stagnation_recovery_active", False)),
                "current_committed_frontier_target": _to_jsonable_target(dbg.get("current_committed_frontier_target")),
                "frontier_commitment_remaining_steps": int(dbg.get("frontier_commitment_remaining_steps", 0)),
                "frontier_switch_reason": str(dbg.get("frontier_switch_reason", "")),
                "doorway_rejection_reason": str(dbg.get("doorway_rejection_reason", "none")),
                "doorway_acceptance_reason": str(dbg.get("doorway_acceptance_reason", "none")),
                "safe_forward_chosen_count": int(dbg.get("safe_forward_chosen_count", 0)),
                "safe_forward_blocked_count": int(dbg.get("safe_forward_blocked_count", 0)),
                "distance_to_target": dbg.get("distance_to_target"),
                "distance_to_target_delta": float(dbg.get("distance_to_target_delta", 0.0)),
                "progress_toward_target_count": int(dbg.get("progress_toward_target_count", 0)),
                "no_progress_count": int(dbg.get("no_progress_count", 0)),
                "blocked_step_count": int(dbg.get("blocked_step_count", 0)),
                "route_commit_remaining": int(dbg.get("route_commit_remaining", 0)),
                "route_target_switch_reason": str(dbg.get("route_target_switch_reason", "")),
                "doorway_target_selected_count": int(dbg.get("doorway_target_selected_count", 0)),
                "doorway_target_reached_count": int(dbg.get("doorway_target_reached_count", 0)),
                "sweep_heading_rad": float(dbg.get("sweep_heading_rad", 0.0)),
                "sweep_burst_remaining": int(dbg.get("sweep_burst_remaining", 0)),
                "sweep_last_reason": str(dbg.get("sweep_last_reason", "")),
                "sweep_direction_score": float(dbg.get("sweep_direction_score", 0.0)),
                "local_obstacle_density": float(dbg.get("local_obstacle_density", 0.0)),
                "local_free_space_ratio": float(dbg.get("local_free_space_ratio", 0.0)),
                "local_wall_structure_score": float(dbg.get("local_wall_structure_score", 0.0)),
                "local_doorway_score": float(dbg.get("local_doorway_score", 0.0)),
                "local_clutter_score": float(dbg.get("local_clutter_score", 0.0)),
                "local_corridor_score": float(dbg.get("local_corridor_score", 0.0)),
                "adaptive_selected_subpolicy": str(dbg.get("adaptive_selected_subpolicy", "")),
                "adaptive_switch_reason": str(dbg.get("adaptive_switch_reason", "")),
                "adaptive_switch_count": int(dbg.get("adaptive_switch_count", 0)),
                "steps_in_stabilized_frontier": int(dbg.get("steps_in_stabilized_frontier", 0)),
                "steps_in_coverage_sweep": int(dbg.get("steps_in_coverage_sweep", 0)),
            }
        )

        if args.save_frames:
            frame_path = frames_dir / f"frame_{step:04d}.png"
            low_conf = dbg.get("low_confidence_frontier_cells", [])
            low_conf = low_conf if isinstance(low_conf, list) else []
            # Keep plots readable by downsampling frontier overlay.
            if len(low_conf) > 500:
                low_conf = low_conf[:: max(1, len(low_conf) // 500)]
            overlay_text = (
                f"mode={args.policy_mode}  step={step}  action={dbg.get('chosen_action')}  cov={coverage:.3f}\n"
                f"conf={confidence_mean:.3f} (gain={confidence_gain:+.3f})  "
                f"collision_risk={'YES' if dbg.get('collision_risk', False) else 'no'}\n"
                f"doorway={doorway_state}  cand/acc/rej="
                f"{int(dbg.get('doorway_candidate_count', 0))}/"
                f"{int(dbg.get('doorway_accepted_count', 0))}/"
                f"{int(dbg.get('doorway_rejected_count', 0))}\n"
                f"frontier_score={float(dbg.get('target_frontier_score', 0.0)):.2f}  "
                f"stagnation_recovery={'ON' if dbg.get('stagnation_recovery_active', False) else 'off'}\n"
                f"commit_target={dbg.get('current_committed_frontier_target')} "
                f"commit_left={int(dbg.get('frontier_commitment_remaining_steps', 0))} "
                f"switch={dbg.get('frontier_switch_reason', '')}\n"
                f"safe_fwd(chosen/blocked)="
                f"{int(dbg.get('safe_forward_chosen_count', 0))}/"
                f"{int(dbg.get('safe_forward_blocked_count', 0))} "
                f"door_rej={dbg.get('doorway_rejection_reason', 'none')}\n"
                f"route_tgt={dbg.get('target_frontier')} "
                f"dist={float(dbg.get('distance_to_target') or 0.0):.2f}m "
                f"dDist={float(dbg.get('distance_to_target_delta', 0.0)):+.3f} "
                f"replan={dbg.get('route_target_switch_reason', 'none')}\n"
                f"sweep_hdg={float(dbg.get('sweep_heading_rad', 0.0)):+.2f} "
                f"burst={int(dbg.get('sweep_burst_remaining', 0))} "
                f"reason={dbg.get('sweep_last_reason', '')} "
                f"dir_score={float(dbg.get('sweep_direction_score', 0.0)):.2f}\n"
                f"adaptive_sub={dbg.get('adaptive_selected_subpolicy', args.policy_mode)} "
                f"switch={dbg.get('adaptive_switch_reason', 'none')} "
                f"sw_count={int(dbg.get('adaptive_switch_count', 0))}\n"
                f"structure clutter={float(dbg.get('local_clutter_score', 0.0)):.2f} "
                f"corridor={float(dbg.get('local_corridor_score', 0.0)):.2f} "
                f"doorway={float(dbg.get('local_doorway_score', 0.0)):.2f}"
            )
            render_simulation_state(
                env=env,
                agent=agent,
                sensor_obs=obs,
                predicted_maps=pred,
                selected_target=target,
                low_conf_frontiers=low_conf,
                doorway_candidates=candidate_doorway_cells[-40:],
                accepted_doorway_candidates=accepted_doorway_cells[-40:],
                rejected_doorway_candidates=rejected_doorway_cells[-40:],
                collision_risk=bool(dbg.get("collision_risk", False)),
                action_text=f"action={dbg.get('chosen_action')} reason={dbg.get('action_reason')}",
                overlay_text=overlay_text,
                step_idx=step,
                title=f"Navigation Replay :: {args.map} :: {args.difficulty}",
                save_path=frame_path,
            )
            frame_paths.append(frame_path)

    gif_path = replay_dir / f"{run_name}.gif"
    gif_saved = False
    if args.save_gif and args.save_frames:
        gif_saved = _save_gif(frame_paths, gif_path)

    final_coverage = _compute_coverage(env, observed)
    final_confidence_mean = float(prev_confidence_mean)
    final_metrics: Dict[str, object] = {
        "map_name": str(args.map),
        "difficulty": str(args.difficulty),
        "step_count": int(args.steps),
        "policy_mode": str(args.policy_mode),
        "accepted_mapper_manifest_name": str(mapper_cfg.get("accepted_model_name", "unknown")),
        "accepted_navigation_manifest_name": str(nav_cfg.get("accepted_model_name", "unknown")),
        "final_agent_pose": {
            "x": float(agent.x),
            "y": float(agent.y),
            "theta": float(agent.theta),
        },
        "final_coverage": float(final_coverage),
        "collision_count": int(collision_count),
        "fake_doorway_approach_count": int(fake_doorway_approach_count),
        "accepted_doorway_decision_count": int(accepted_doorway_decision_count),
        "rejected_doorway_decision_count": int(rejected_doorway_decision_count),
        "mean_confidence": float(final_confidence_mean),
        "total_distance_traveled": float(total_distance_traveled),
        "action_distribution": {k: int(v) for k, v in sorted(action_counter.items())},
        "number_of_frontier_switches": int(frontier_switch_count),
        "number_of_stagnation_events": int(stagnation_event_count),
        "coverage_gain_per_100_steps": float(
            (final_coverage - coverage_history[0]) * 100.0 / max(1, args.steps)
            if coverage_history
            else 0.0
        ),
        "frontier_commit_success_count": int(frontier_commit_success_count),
        "revisit_rate": float(repeated_step_count / max(1, args.steps)),
        "high_confidence_revisit_ratio": float(high_confidence_revisit_count / max(1, args.steps)),
        "stagnation_event_count": int(nav_state.get("stagnation_event_count", 0)),
        "forced_frontier_switch_count": int(nav_state.get("forced_frontier_switch_count", 0)),
        "frontier_blacklist_count": int(len(nav_state.get("frontier_blacklist", set()))),
        "safe_forward_preference_count": int(nav_state.get("safe_forward_preference_count", 0)),
        "safe_forward_chosen_count": int(nav_state.get("safe_forward_chosen_count", 0)),
        "safe_forward_blocked_count": int(nav_state.get("safe_forward_blocked_count", 0)),
        "doorway_for_coverage_count": int(nav_state.get("doorway_for_coverage_count", 0)),
        "frontier_commitment_count": int(nav_state.get("frontier_commitment_count", 0)),
        "premature_frontier_switch_count": int(nav_state.get("premature_frontier_switch_count", 0)),
        "target_reached_count": int(nav_state.get("target_reached_count", 0)),
        "route_commitment_count": int(nav_state.get("route_commitment_count", 0)),
        "route_replan_count": int(nav_state.get("route_replan_count", 0)),
        "progress_toward_target_count": int(nav_state.get("progress_toward_target_count", 0)),
        "no_progress_count": int(nav_state.get("no_progress_count", 0)),
        "blocked_step_count": int(nav_state.get("blocked_step_count", 0)),
        "doorway_target_selected_count": int(nav_state.get("doorway_target_selected_count", 0)),
        "doorway_target_reached_count": int(nav_state.get("doorway_target_reached_count", 0)),
        "doorway_acceptance_reason": str(nav_state.get("doorway_acceptance_reason", "none")),
        "visited_cell_ratio": float(len(visited_cell_counts) / max(1, env.shape[0] * env.shape[1])),
        "high_revisit_cell_count": int(nav_state.get("high_revisit_cell_count", 0)),
        "revisit_penalty_weight": float(nav_state.get("revisit_penalty_weight", 0.0)),
        "doorway_rejection_reason_counts": dict(nav_state.get("doorway_rejection_reasons", {})),
        "sweep_forward_burst_count": int(nav_state.get("sweep_forward_burst_count", 0)),
        "sweep_blocked_turn_count": int(nav_state.get("sweep_blocked_turn_count", 0)),
        "sweep_scan_turn_count": int(nav_state.get("sweep_scan_turn_count", 0)),
        "sweep_direction_change_count": int(nav_state.get("sweep_direction_change_count", 0)),
        "forward_safe_count": int(nav_state.get("forward_safe_count", 0)),
        "forward_blocked_count": int(nav_state.get("forward_blocked_count", 0)),
        "low_confidence_direction_score": float(
            float(nav_state.get("low_confidence_direction_score_sum", 0.0))
            / max(1, int(nav_state.get("direction_score_count", 0)))
        ),
        "unvisited_direction_score": float(
            float(nav_state.get("unvisited_direction_score_sum", 0.0))
            / max(1, int(nav_state.get("direction_score_count", 0)))
        ),
        "oscillation_prevented_count": int(nav_state.get("oscillation_prevented_count", 0)),
        "local_obstacle_density": float(nav_state.get("local_obstacle_density", 0.0)),
        "local_free_space_ratio": float(nav_state.get("local_free_space_ratio", 0.0)),
        "local_wall_structure_score": float(nav_state.get("local_wall_structure_score", 0.0)),
        "local_doorway_score": float(nav_state.get("local_doorway_score", 0.0)),
        "local_clutter_score": float(nav_state.get("local_clutter_score", 0.0)),
        "local_corridor_score": float(nav_state.get("local_corridor_score", 0.0)),
        "adaptive_selected_subpolicy": str(nav_state.get("adaptive_selected_subpolicy", args.policy_mode)),
        "adaptive_switch_reason": str(nav_state.get("adaptive_switch_reason", "none")),
        "adaptive_switch_count": int(nav_state.get("adaptive_switch_count", 0)),
        "adaptive_switch_reason_counts": dict(nav_state.get("adaptive_switch_reason_counts", {})),
        "steps_in_stabilized_frontier": int(nav_state.get("steps_in_stabilized_frontier", 0)),
        "steps_in_coverage_sweep": int(nav_state.get("steps_in_coverage_sweep", 0)),
        "adaptive_min_mode_steps": int(nav_state.get("adaptive_min_mode_steps", 40)),
        "adaptive_coverage_gain_window": int(nav_state.get("adaptive_coverage_gain_window", 50)),
        "mean_distance_to_target": float(
            np.mean([float(m["distance_to_target"]) for m in step_metrics if m.get("distance_to_target") is not None])
            if any(m.get("distance_to_target") is not None for m in step_metrics)
            else 0.0
        ),
        "mean_heading_error": float(
            float(nav_state.get("mean_heading_error_accum", 0.0)) / max(1, int(nav_state.get("heading_error_count", 0)))
        ),
        "mean_distance_to_selected_frontier": float(
            np.mean(
                [
                    np.hypot(
                        (m["frontier_target"][0] + 0.5) * env.cell_size - agent.trajectory[min(i, len(agent.trajectory) - 1)][0],
                        (m["frontier_target"][1] + 0.5) * env.cell_size - agent.trajectory[min(i, len(agent.trajectory) - 1)][1],
                    )
                    for i, m in enumerate(step_metrics)
                    if m["frontier_target"] is not None
                ]
            )
            if any(m["frontier_target"] is not None for m in step_metrics)
            else 0.0
        ),
        "percent_steps_targeting_unknown_frontier": float(steps_targeting_unknown_frontier / max(1, args.steps)),
        "percent_steps_safe_forward_bias_active": float(steps_safe_forward_bias / max(1, args.steps)),
    }

    metrics_payload = {
        "run_name": run_name,
        "map": args.map,
        "difficulty": args.difficulty,
        "seed": int(args.seed),
        "steps": int(args.steps),
        "save_frames": bool(args.save_frames),
        "save_gif": bool(args.save_gif),
        "frames_saved": int(len(frame_paths)) if args.save_frames else 0,
        "gif_saved": bool(gif_saved),
        "final_metrics": final_metrics,
        "per_step_metrics": step_metrics,
    }
    metrics_path = replay_dir / "replay_metrics.json"
    summary_path = replay_dir / "replay_summary.txt"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    summary_lines = [
        f"run_name: {run_name}",
        f"map: {args.map}",
        f"difficulty: {args.difficulty}",
        f"policy_mode: {args.policy_mode}",
        f"steps: {args.steps}",
        f"accepted_mapper_manifest_name: {final_metrics['accepted_mapper_manifest_name']}",
        f"accepted_navigation_manifest_name: {final_metrics['accepted_navigation_manifest_name']}",
        f"final_agent_pose: {json.dumps(final_metrics['final_agent_pose'])}",
        f"final_coverage: {final_metrics['final_coverage']:.4f}",
        f"collision_count: {final_metrics['collision_count']}",
        f"fake_doorway_approach_count: {final_metrics['fake_doorway_approach_count']}",
        f"accepted_doorway_decision_count: {final_metrics['accepted_doorway_decision_count']}",
        f"rejected_doorway_decision_count: {final_metrics['rejected_doorway_decision_count']}",
        f"mean_confidence: {final_metrics['mean_confidence']:.4f}",
        f"total_distance_traveled: {final_metrics['total_distance_traveled']:.4f}",
        f"action_distribution: {json.dumps(final_metrics['action_distribution'])}",
        f"number_of_frontier_switches: {final_metrics['number_of_frontier_switches']}",
        f"number_of_stagnation_events: {final_metrics['number_of_stagnation_events']}",
        f"stagnation_event_count: {final_metrics['stagnation_event_count']}",
        f"forced_frontier_switch_count: {final_metrics['forced_frontier_switch_count']}",
        f"frontier_blacklist_count: {final_metrics['frontier_blacklist_count']}",
        f"high_confidence_revisit_ratio: {final_metrics['high_confidence_revisit_ratio']:.4f}",
        f"safe_forward_preference_count: {final_metrics['safe_forward_preference_count']}",
        f"safe_forward_chosen_count: {final_metrics['safe_forward_chosen_count']}",
        f"safe_forward_blocked_count: {final_metrics['safe_forward_blocked_count']}",
        f"doorway_for_coverage_count: {final_metrics['doorway_for_coverage_count']}",
        f"frontier_commitment_count: {final_metrics['frontier_commitment_count']}",
        f"premature_frontier_switch_count: {final_metrics['premature_frontier_switch_count']}",
        f"target_reached_count: {final_metrics['target_reached_count']}",
        f"route_commitment_count: {final_metrics['route_commitment_count']}",
        f"route_replan_count: {final_metrics['route_replan_count']}",
        f"progress_toward_target_count: {final_metrics['progress_toward_target_count']}",
        f"no_progress_count: {final_metrics['no_progress_count']}",
        f"blocked_step_count: {final_metrics['blocked_step_count']}",
        f"doorway_target_selected_count: {final_metrics['doorway_target_selected_count']}",
        f"doorway_target_reached_count: {final_metrics['doorway_target_reached_count']}",
        f"doorway_acceptance_reason: {final_metrics['doorway_acceptance_reason']}",
        f"visited_cell_ratio: {final_metrics['visited_cell_ratio']:.4f}",
        f"high_revisit_cell_count: {final_metrics['high_revisit_cell_count']}",
        f"revisit_penalty_weight: {final_metrics['revisit_penalty_weight']:.4f}",
        f"doorway_rejection_reason_counts: {json.dumps(final_metrics['doorway_rejection_reason_counts'])}",
        f"sweep_forward_burst_count: {final_metrics['sweep_forward_burst_count']}",
        f"sweep_blocked_turn_count: {final_metrics['sweep_blocked_turn_count']}",
        f"sweep_scan_turn_count: {final_metrics['sweep_scan_turn_count']}",
        f"sweep_direction_change_count: {final_metrics['sweep_direction_change_count']}",
        f"forward_safe_count: {final_metrics['forward_safe_count']}",
        f"forward_blocked_count: {final_metrics['forward_blocked_count']}",
        f"low_confidence_direction_score: {final_metrics['low_confidence_direction_score']:.4f}",
        f"unvisited_direction_score: {final_metrics['unvisited_direction_score']:.4f}",
        f"oscillation_prevented_count: {final_metrics['oscillation_prevented_count']}",
        f"local_obstacle_density: {final_metrics['local_obstacle_density']:.4f}",
        f"local_free_space_ratio: {final_metrics['local_free_space_ratio']:.4f}",
        f"local_wall_structure_score: {final_metrics['local_wall_structure_score']:.4f}",
        f"local_doorway_score: {final_metrics['local_doorway_score']:.4f}",
        f"local_clutter_score: {final_metrics['local_clutter_score']:.4f}",
        f"local_corridor_score: {final_metrics['local_corridor_score']:.4f}",
        f"adaptive_selected_subpolicy: {final_metrics['adaptive_selected_subpolicy']}",
        f"adaptive_switch_reason: {final_metrics['adaptive_switch_reason']}",
        f"adaptive_switch_count: {final_metrics['adaptive_switch_count']}",
        f"adaptive_switch_reason_counts: {json.dumps(final_metrics['adaptive_switch_reason_counts'])}",
        f"steps_in_stabilized_frontier: {final_metrics['steps_in_stabilized_frontier']}",
        f"steps_in_coverage_sweep: {final_metrics['steps_in_coverage_sweep']}",
        f"adaptive_min_mode_steps: {final_metrics['adaptive_min_mode_steps']}",
        f"adaptive_coverage_gain_window: {final_metrics['adaptive_coverage_gain_window']}",
        f"mean_distance_to_target: {final_metrics['mean_distance_to_target']:.4f}",
        f"mean_heading_error: {final_metrics['mean_heading_error']:.4f}",
        f"revisit_rate: {final_metrics['revisit_rate']:.4f}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Replay output directory: {replay_dir}")
    if args.save_frames:
        print(f"Saved frames: {len(frame_paths)} at {frames_dir}")
    if args.save_gif:
        print(f"Saved GIF: {gif_path}" if gif_saved else "GIF not saved (Pillow missing or no frames).")
    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved summary TXT: {summary_path}")
    print(
        "Replay includes trajectory, acoustic rays, confidence-map growth, frontier targets, "
        "doorway candidate decisions, and collision-risk checks."
    )


if __name__ == "__main__":
    main()
