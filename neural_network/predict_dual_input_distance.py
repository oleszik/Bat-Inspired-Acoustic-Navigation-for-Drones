"""
Predict obstacle and distance using the dual-input regression model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_utils import preprocess_pil_image
from dual_input_model import AcousticDualInputRegressionCNN


DEFAULT_RUN_NAME = "hard_v1_dual"
BASE_CHECKPOINT_DIR = Path("neural_network/checkpoints")
IMAGE_SIZE = 128
CORR_LEN = 512


def normalize_corr_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != CORR_LEN:
        raise ValueError(f"Correlation vector length must be {CORR_LEN}, got {x.size}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(x.mean())
    std = float(x.std())
    if std > 1e-6:
        x = (x - mean) / std
    else:
        x = x - mean
    return x.astype(np.float32)


def load_model(checkpoint_path: Path, device: torch.device) -> AcousticDualInputRegressionCNN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AcousticDualInputRegressionCNN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with dual-input acoustic regression model.")
    parser.add_argument("spectrogram_path", type=str, help="Path to spectrogram image.")
    parser.add_argument("correlation_path", type=str, help="Path to correlation .npy feature.")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME, help="Run name for checkpoint folder.")
    args = parser.parse_args()

    spec_path = Path(args.spectrogram_path)
    corr_path = Path(args.correlation_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"Spectrogram not found: {spec_path}")
    if not corr_path.exists():
        raise FileNotFoundError(f"Correlation feature not found: {corr_path}")

    checkpoint_path = BASE_CHECKPOINT_DIR / args.run_name / "acoustic_dual_input_regression_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path=checkpoint_path, device=device)

    image = Image.open(spec_path).convert("RGB")
    image_tensor = preprocess_pil_image(image, image_size=IMAGE_SIZE).unsqueeze(0).to(device)

    corr_vec = np.load(corr_path)
    corr_vec = normalize_corr_vector(corr_vec)
    corr_tensor = torch.from_numpy(corr_vec).unsqueeze(0).to(device)

    with torch.no_grad():
        obstacle_logit, distance_pred_m = model(image_tensor, corr_tensor)
        obstacle_prob = float(torch.sigmoid(obstacle_logit).item())
        pred_has_obstacle = int(obstacle_prob >= 0.5)
        pred_distance_m = float(distance_pred_m.item())

    print(f"Spectrogram: {spec_path}")
    print(f"Correlation: {corr_path}")
    print(f"Run name: {args.run_name}")
    print(f"Obstacle probability: {obstacle_prob:.4f}")
    print(f"Predicted has_obstacle: {pred_has_obstacle}")
    print(f"Predicted distance (m): {pred_distance_m:.4f}")
    print(f"Predicted distance (cm): {pred_distance_m * 100.0:.2f}")


if __name__ == "__main__":
    main()
