from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .agent import AcousticAgent
from .environments import EnvironmentMap


def render_simulation_state(
    env: EnvironmentMap,
    agent: AcousticAgent,
    sensor_obs: Dict[str, np.ndarray],
    predicted_maps: Dict[str, np.ndarray],
    selected_target: Optional[Tuple[int, int]] = None,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=130)
    ax = axes.flatten()

    gt_occ = env.ground_truth_occupancy.astype(float)
    ax[0].imshow(gt_occ, cmap="gray_r", vmin=0.0, vmax=1.0)
    ax[0].set_title("Ground Truth Occupancy")

    ax[1].imshow(predicted_maps["occupancy_prob"], cmap="gray_r", vmin=0.0, vmax=1.0)
    ax[1].set_title("Pred Occupancy Prob")

    ax[2].imshow(predicted_maps["wall_prob"], cmap="magma", vmin=0.0, vmax=1.0)
    ax[2].set_title("Wall Probability")

    ax[3].imshow(predicted_maps["doorway_prob"], cmap="viridis", vmin=0.0, vmax=1.0)
    ax[3].set_title("Doorway Probability")

    ax[4].imshow(predicted_maps["confidence_map"], cmap="Blues", vmin=0.0, vmax=1.0)
    ax[4].set_title("Confidence Map")

    # Trajectory and rays.
    ax[5].imshow(gt_occ, cmap="gray_r", alpha=0.35)
    traj = np.asarray(agent.trajectory, dtype=float)
    if len(traj) > 1:
        ax[5].plot(traj[:, 0] / env.cell_size, traj[:, 1] / env.cell_size, c="tab:blue", lw=1.6, label="trajectory")
    ax[5].scatter([agent.x / env.cell_size], [agent.y / env.cell_size], c="red", s=30, label="agent")

    ray_angles = sensor_obs["ray_angles_rad"]
    ray_dist = sensor_obs["ray_distances_m"]
    for ang, dist in zip(ray_angles, ray_dist):
        px = agent.x + float(dist) * np.cos(float(ang))
        py = agent.y + float(dist) * np.sin(float(ang))
        ax[5].plot(
            [agent.x / env.cell_size, px / env.cell_size],
            [agent.y / env.cell_size, py / env.cell_size],
            c="cyan",
            alpha=0.35,
            lw=0.8,
        )

    if selected_target is not None:
        tx, ty = selected_target
        ax[5].scatter([tx], [ty], c="gold", s=45, marker="*", label="frontier target")
    ax[5].set_title("Agent + Rays + Target")
    ax[5].legend(fontsize=7, loc="upper right")

    ax[6].plot(sensor_obs["echo_timing_vector"], c="tab:purple", lw=1.2)
    ax[6].set_title("Echo Timing Vector")
    ax[6].set_ylim(0.0, 1.05)

    ax[7].plot(sensor_obs["echo_intensity_vector"], c="tab:green", lw=1.2)
    ax[7].set_title("Echo Intensity Vector")
    ax[7].set_ylim(0.0, 1.05)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(title if title else f"sim_env :: {env.name}", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    plt.close(fig)
