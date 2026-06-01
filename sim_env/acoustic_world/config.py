from __future__ import annotations

DEFAULT_CELL_SIZE = 0.25
DEFAULT_WIDTH_M = 12.0
DEFAULT_HEIGHT_M = 8.0

SPEED_OF_SOUND_M_S = 343.0
MAX_SENSOR_RANGE_M = 3.5
ECHO_VECTOR_BINS = 128

DEFAULT_RAY_ANGLES_DEG = [-70, -40, -20, -8, 0, 8, 20, 40, 70]

DIFFICULTY_PRESETS = {
    "clean": {
        "distance_noise_std": 0.00,
        "intensity_noise_std": 0.02,
        "dropout_prob": 0.00,
    },
    "mild_noise": {
        "distance_noise_std": 0.02,
        "intensity_noise_std": 0.05,
        "dropout_prob": 0.02,
    },
    "medium_noise": {
        "distance_noise_std": 0.05,
        "intensity_noise_std": 0.08,
        "dropout_prob": 0.05,
    },
    "hard_noise": {
        "distance_noise_std": 0.08,
        "intensity_noise_std": 0.12,
        "dropout_prob": 0.10,
    },
}
