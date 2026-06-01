import numpy as np

from sim_env.acoustic_world import binary_metrics


def test_binary_metrics_basic_case():
    gt = np.array([[1, 0], [1, 0]], dtype=bool)
    pred = np.array([[1, 0], [0, 0]], dtype=bool)
    m = binary_metrics(pred, gt)
    assert abs(m["precision"] - 1.0) < 1e-6
    assert abs(m["recall"] - 0.5) < 1e-6
    assert m["f1"] > 0.0
    assert m["iou"] > 0.0
    assert m["accuracy"] > 0.0
