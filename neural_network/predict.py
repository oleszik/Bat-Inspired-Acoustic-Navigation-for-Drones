"""
Predict class for one spectrogram image using a trained AcousticCNN model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data_utils import preprocess_pil_image
from model import AcousticCNN


CHECKPOINT_PATH = Path("neural_network/checkpoints/acoustic_cnn_best.pt")
CLASS_MAP_PATH = Path("neural_network/checkpoints/class_to_idx.json")
IMAGE_SIZE = 128


def load_model(num_classes: int, device: torch.device) -> AcousticCNN:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = AcousticCNN(num_classes=num_classes).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict class for one spectrogram image.")
    parser.add_argument("image_path", type=str, help="Path to the spectrogram image.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(f"Class mapping not found: {CLASS_MAP_PATH}")

    with CLASS_MAP_PATH.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(num_classes=len(class_to_idx), device=device)

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess_pil_image(image, image_size=IMAGE_SIZE).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_class = idx_to_class[int(pred_idx.item())]
    conf_pct = float(confidence.item()) * 100.0

    print(f"Image: {image_path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {conf_pct:.2f}%")


if __name__ == "__main__":
    main()
