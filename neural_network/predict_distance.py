"""
Run single-image prediction with the trained acoustic regression CNN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from data_utils import preprocess_pil_image
from regression_model import AcousticRegressionCNN


DEFAULT_RUN_NAME = "clean"
BASE_CHECKPOINT_DIR = Path("neural_network/checkpoints")
IMAGE_SIZE = 128


def load_model(device: torch.device, checkpoint_path: Path) -> AcousticRegressionCNN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AcousticRegressionCNN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict obstacle and distance for one spectrogram image.")
    parser.add_argument("image_path", type=str, help="Path to spectrogram image.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Run name used for checkpoint subfolder.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    checkpoint_path = BASE_CHECKPOINT_DIR / args.run_name / "acoustic_regression_cnn_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, checkpoint_path=checkpoint_path)

    image = Image.open(image_path).convert("RGB")
    x = preprocess_pil_image(image, image_size=IMAGE_SIZE).unsqueeze(0).to(device)

    with torch.no_grad():
        obstacle_logit, distance_pred_m = model(x)
        obstacle_prob = torch.sigmoid(obstacle_logit).item()
        has_obstacle_pred = int(obstacle_prob >= 0.5)
        pred_distance_m = float(distance_pred_m.item())

    print(f"Image: {image_path}")
    print(f"Run name: {args.run_name}")
    print(f"Obstacle probability: {obstacle_prob:.4f}")
    print(f"Predicted has_obstacle: {has_obstacle_pred}")
    print(f"Predicted distance (m): {pred_distance_m:.4f}")
    print(f"Predicted distance (cm): {pred_distance_m * 100.0:.2f}")


if __name__ == "__main__":
    main()
