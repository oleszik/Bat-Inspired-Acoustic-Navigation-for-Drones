import numpy as np

from sim_env.acoustic_world import AcousticAgent, choose_action_simple, get_environment


def _dummy_obs(front: float, left: float, right: float, n: int = 9):
    d = np.full((n,), front, dtype=np.float32)
    d[:3] = left
    d[-3:] = right
    return {
        "ray_angles_rad": np.linspace(-1.0, 1.0, n, dtype=np.float32),
        "ray_distances_m": d,
        "ray_intensities": np.full((n,), 0.5, dtype=np.float32),
        "echo_timing_vector": np.zeros((128,), dtype=np.float32),
        "echo_intensity_vector": np.zeros((128,), dtype=np.float32),
        "multichannel_features": np.zeros((2, 128), dtype=np.float32),
    }


def _dummy_maps(env):
    shape = env.shape
    return {
        "occupancy_prob": env.ground_truth_occupancy.astype(float),
        "wall_prob": env.wall_mask.astype(float),
        "doorway_prob": np.zeros(shape, dtype=float),
        "free_prob": env.free_space_mask.astype(float),
        "confidence_map": np.full(shape, 0.5, dtype=float),
    }


def test_navigation_turns_when_front_blocked():
    env = get_environment("doorway", seed=1)
    agent = AcousticAgent(*env.start_pose)
    obs = _dummy_obs(front=0.08, left=0.40, right=0.10)
    action, _ = choose_action_simple(env, agent, obs, _dummy_maps(env))
    assert action in {"turn_left", "turn_right"}


def test_navigation_can_move_forward_when_safe():
    env = get_environment("empty_room", seed=1)
    agent = AcousticAgent(*env.start_pose)
    obs = _dummy_obs(front=1.50, left=1.20, right=1.20)
    action, _ = choose_action_simple(env, agent, obs, _dummy_maps(env))
    assert action in {"move_forward_slow", "probe_forward", "turn_left", "turn_right"}
