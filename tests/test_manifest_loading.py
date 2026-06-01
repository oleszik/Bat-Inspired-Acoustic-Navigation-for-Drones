from sim_env.acoustic_world import load_mapper_manifest, load_navigation_manifest


def test_mapper_manifest_loads():
    manifest = load_mapper_manifest()
    assert isinstance(manifest, dict)
    assert manifest.get("accepted_model_name") == "phase2c5_hybrid_acoustic_mapper"


def test_navigation_manifest_loads():
    manifest = load_navigation_manifest()
    assert isinstance(manifest, dict)
    assert manifest.get("accepted_model_name") == "phase2d_mapper_guided_navigation_v3"
