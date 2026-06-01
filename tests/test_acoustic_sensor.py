import numpy as np

from sim_env.acoustic_world import AcousticSensor, get_environment


def test_sensor_output_shapes_and_ranges():
    rng = np.random.default_rng(42)
    env = get_environment("doorway", seed=42)
    sensor = AcousticSensor.default()
    x, y, th = env.start_pose
    obs = sensor.sense(env, x, y, th, "clean", rng)

    assert obs["echo_timing_vector"].shape == (sensor.echo_bins,)
    assert obs["echo_intensity_vector"].shape == (sensor.echo_bins,)
    assert obs["multichannel_features"].shape == (2, sensor.echo_bins)
    assert obs["ray_distances_m"].shape[0] == len(sensor.ray_angles_deg)
    assert np.all(obs["ray_distances_m"] >= 0.0)
    assert np.all(obs["ray_distances_m"] <= sensor.max_range_m + 1e-6)
    assert np.all(obs["ray_intensities"] >= 0.0)
    assert np.all(obs["ray_intensities"] <= 1.0)
