from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim_env.acoustic_world import (
    AcousticAgent,
    AcousticSensor,
    dummy_predict_maps,
    get_environment,
    load_mapper_manifest,
    render_simulation_state,
)
from sim_env.acoustic_world.environments import world_to_cell


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run mapper bridge demo.")
    p.add_argument("--map", type=str, default="doorway")
    p.add_argument("--difficulty", type=str, default="clean")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--save-plots", action="store_true")
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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = get_environment(args.map, seed=args.seed)
    agent = AcousticAgent(*env.start_pose)
    sensor = AcousticSensor.default()
    mapper_manifest = load_mapper_manifest()
    print(f"Loaded mapper manifest: {mapper_manifest.get('accepted_model_name', 'unknown')}")

    observed = np.zeros(env.shape, dtype=bool)
    pred = None
    last_obs = None
    for step in range(args.steps):
        obs = sensor.sense(env, agent.x, agent.y, agent.theta, args.difficulty, rng)
        last_obs = obs
        mark_observed(env, observed, agent, obs)
        pred = dummy_predict_maps(env, observed, rng)
        # Light motion to sweep environment.
        if step % 8 in (0, 1, 2, 3):
            agent.apply_action(env, "move_forward_slow")
        else:
            agent.apply_action(env, "turn_left")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mapper_demo_{args.map}_{args.difficulty}.png"
    render_simulation_state(
        env=env,
        agent=agent,
        sensor_obs=last_obs if last_obs is not None else sensor.sense(env, agent.x, agent.y, agent.theta, args.difficulty, rng),
        predicted_maps=pred if pred is not None else dummy_predict_maps(env, observed, rng),
        title=f"Mapper Demo :: {args.map} :: {args.difficulty}",
        save_path=out_path if args.save_plots else None,
    )
    print(f"Demo finished. Plot saved: {out_path}" if args.save_plots else "Demo finished.")


if __name__ == "__main__":
    main()
