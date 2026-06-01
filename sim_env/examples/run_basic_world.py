from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_env.acoustic_world import AcousticAgent, AcousticSensor, get_environment, render_simulation_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run basic sim_env world demo.")
    p.add_argument("--map", type=str, default="doorway")
    p.add_argument("--difficulty", type=str, default="clean")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--output-dir", type=str, default="sim_env/outputs")
    return p.parse_args()


def choose_basic_action(sensor_obs: dict) -> str:
    d = sensor_obs["ray_distances_m"]
    front = float(d[len(d) // 2])
    left = float(np.mean(d[:3]))
    right = float(np.mean(d[-3:]))
    if front < 0.12:
        return "turn_left" if left > right else "turn_right"
    if front > 0.24:
        return "move_forward_slow"
    if front > 0.14:
        return "probe_forward"
    return "turn_left" if left > right else "turn_right"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = get_environment(args.map, seed=args.seed)
    agent = AcousticAgent(*env.start_pose)
    sensor = AcousticSensor.default()

    observed = np.zeros(env.shape, dtype=bool)
    last_obs = None
    for _ in range(args.steps):
        obs = sensor.sense(env, agent.x, agent.y, agent.theta, args.difficulty, rng)
        last_obs = obs
        action = choose_basic_action(obs)
        agent.apply_action(env, action)
        cx, cy = agent.cell(env)
        observed[cy, cx] = True

    pred = {
        "occupancy_prob": env.ground_truth_occupancy.astype(float),
        "wall_prob": env.wall_mask.astype(float),
        "doorway_prob": env.doorway_mask.astype(float),
        "free_prob": env.free_space_mask.astype(float),
        "confidence_map": observed.astype(float),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"basic_world_{args.map}_{args.difficulty}.png"
    render_simulation_state(
        env=env,
        agent=agent,
        sensor_obs=last_obs if last_obs is not None else sensor.sense(env, agent.x, agent.y, agent.theta, args.difficulty, rng),
        predicted_maps=pred,
        selected_target=None,
        title=f"Basic World :: {args.map} :: {args.difficulty}",
        save_path=out_path if args.save_plots else None,
    )
    print(f"Demo finished. Plot saved: {out_path}" if args.save_plots else "Demo finished.")


if __name__ == "__main__":
    main()
