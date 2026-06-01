from sim_env.acoustic_world import get_environment


def test_environment_maps_have_consistent_shapes():
    for name in ["empty_room", "corridor", "single_block", "doorway", "cluttered_room"]:
        env = get_environment(name, seed=123)
        h, w = env.shape
        assert h > 0 and w > 0
        assert env.wall_mask.shape == env.ground_truth_occupancy.shape
        assert env.doorway_mask.shape == env.ground_truth_occupancy.shape
        assert env.free_space_mask.shape == env.ground_truth_occupancy.shape


def test_free_space_is_inverse_of_occupancy():
    env = get_environment("doorway", seed=7)
    assert ((~env.ground_truth_occupancy) == env.free_space_mask).all()
