from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .agent import AcousticAgent
from .environments import EnvironmentMap


def render_simulation_state(
    env: EnvironmentMap,
    agent: AcousticAgent,
    sensor_obs: Dict[str, np.ndarray],
    predicted_maps: Dict[str, np.ndarray],
    selected_target: Optional[Tuple[int, int]] = None,
    low_conf_frontiers: Optional[List[Tuple[int, int]]] = None,
    doorway_candidates: Optional[List[Tuple[int, int]]] = None,
    accepted_doorway_candidates: Optional[List[Tuple[int, int]]] = None,
    rejected_doorway_candidates: Optional[List[Tuple[int, int]]] = None,
    collision_risk: Optional[bool] = None,
    action_text: str = "",
    overlay_text: str = "",
    step_idx: Optional[int] = None,
    title: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=130)
    # Keep subplot geometry fixed across frames for stable replay panel sizes.
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.04, top=0.90, wspace=0.10, hspace=0.18)
    ax = axes.flatten()
    h, w = env.shape

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

    if low_conf_frontiers:
        f = np.asarray(low_conf_frontiers, dtype=float)
        ax[5].scatter(f[:, 0], f[:, 1], c="yellow", s=8, alpha=0.35, marker=".", label="low-conf frontiers")
    if doorway_candidates:
        dc = np.asarray(doorway_candidates, dtype=float)
        ax[5].scatter(dc[:, 0], dc[:, 1], c="orange", s=22, alpha=0.55, marker="o", label="doorway cand.")
    if accepted_doorway_candidates:
        ac = np.asarray(accepted_doorway_candidates, dtype=float)
        ax[5].scatter(ac[:, 0], ac[:, 1], c="lime", s=26, alpha=0.8, marker="o", label="doorway accepted")
    if rejected_doorway_candidates:
        rc = np.asarray(rejected_doorway_candidates, dtype=float)
        ax[5].scatter(rc[:, 0], rc[:, 1], c="red", s=22, alpha=0.8, marker="x", label="doorway rejected")
    if selected_target is not None:
        tx, ty = selected_target
        ax[5].scatter([tx], [ty], c="gold", s=45, marker="*", label="frontier target")
    ax[5].set_title("Agent + Rays + Target")
    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=1.6, label="trajectory"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=6, label="agent"),
        Line2D([0], [0], color="cyan", lw=1.0, label="acoustic rays"),
        Line2D([0], [0], marker=".", color="yellow", linestyle="None", markersize=6, label="low-conf frontiers"),
        Line2D([0], [0], marker="o", color="orange", linestyle="None", markersize=6, label="doorway cand."),
        Line2D([0], [0], marker="o", color="lime", linestyle="None", markersize=6, label="doorway accepted"),
        Line2D([0], [0], marker="x", color="red", linestyle="None", markersize=6, label="doorway rejected"),
        Line2D([0], [0], marker="*", color="gold", linestyle="None", markersize=8, label="frontier target"),
    ]
    ax[5].legend(handles=legend_handles, fontsize=7, loc="upper right")
    if overlay_text:
        ax[5].text(
            0.01,
            0.99,
            overlay_text,
            transform=ax[5].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 3, "edgecolor": "none"},
        )

    ax[6].plot(sensor_obs["echo_timing_vector"], c="tab:purple", lw=1.2)
    ax[6].set_title("Echo Timing Vector")
    ax[6].set_ylim(0.0, 1.05)

    ax[7].plot(sensor_obs["echo_intensity_vector"], c="tab:green", lw=1.2)
    ax[7].set_title("Echo Intensity Vector")
    ax[7].set_ylim(0.0, 1.05)
    ax[6].set_xlim(0, max(1, len(sensor_obs["echo_timing_vector"]) - 1))
    ax[7].set_xlim(0, max(1, len(sensor_obs["echo_intensity_vector"]) - 1))

    for i in range(6):
        ax[i].set_xlim(-0.5, w - 0.5)
        ax[i].set_ylim(h - 0.5, -0.5)
        ax[i].set_aspect("equal", adjustable="box")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    # Keep title static to avoid any frame-dependent layout drift.
    fig.suptitle((title if title else f"sim_env :: {env.name}"), fontsize=12)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
