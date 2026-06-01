"""
Evaluate AcousticCNN on the synthetic spectrogram test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from data_utils import SimpleImageFolder
from model import AcousticCNN


DATA_ROOT = Path("datasets/synthetic_echoes/spectrograms_echo_focus")
CHECKPOINT_DIR = Path("neural_network/checkpoints")
RESULTS_DIR = Path("neural_network/results")

BEST_MODEL_PATH = CHECKPOINT_DIR / "acoustic_cnn_best.pt"
CLASS_MAP_PATH = CHECKPOINT_DIR / "class_to_idx.json"
SPLIT_META_PATH = CHECKPOINT_DIR / "split_indices.json"

RESULTS_JSON_PATH = RESULTS_DIR / "cnn_test_results.json"
CONFUSION_PNG_PATH = RESULTS_DIR / "confusion_matrix.png"

IMAGE_SIZE = 128
BATCH_SIZE = 32
SEED = 42


def build_test_subset(dataset: SimpleImageFolder) -> Subset:
    """
    Use saved split indices from training if available.
    Fallback: deterministic 70/15/15 split with the same seed.
    """
    if SPLIT_META_PATH.exists():
        with SPLIT_META_PATH.open("r", encoding="utf-8") as f:
            split_data = json.load(f)
        return Subset(dataset, split_data["test_indices"])

    total = len(dataset)
    train_len = int(0.70 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len
    generator = torch.Generator().manual_seed(SEED)
    _, _, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)
    return test_set


def load_model(num_classes: int, device: torch.device) -> AcousticCNN:
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model = AcousticCNN(num_classes=num_classes).to(device)

    # Support checkpoint dict (recommended) or raw state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_confusion_matrix_png(confusion: np.ndarray, class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("CNN Confusion Matrix (Test Set)")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    fig.savefig(CONFUSION_PNG_PATH)
    plt.close(fig)


def main() -> None:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {BEST_MODEL_PATH}")
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing class mapping: {CLASS_MAP_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CLASS_MAP_PATH.open("r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    dataset = SimpleImageFolder(root=DATA_ROOT, image_size=IMAGE_SIZE)
    test_set = build_test_subset(dataset)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(num_classes=len(class_names), device=device)

    total = 0
    correct = 0
    class_correct = np.zeros(len(class_names), dtype=np.int64)
    class_total = np.zeros(len(class_names), dtype=np.int64)
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            for true_idx, pred_idx in zip(labels_np, preds_np):
                class_total[true_idx] += 1
                class_correct[true_idx] += int(true_idx == pred_idx)
                confusion[true_idx, pred_idx] += 1

    test_accuracy = correct / max(total, 1)
    per_class_accuracy = {}
    for idx, class_name in enumerate(class_names):
        acc = class_correct[idx] / max(class_total[idx], 1)
        per_class_accuracy[class_name] = acc

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print("\nPer-class accuracy:")
    for class_name in class_names:
        print(f"  {class_name}: {per_class_accuracy[class_name] * 100:.2f}%")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion)

    save_confusion_matrix_png(confusion, class_names)

    results = {
        "test_accuracy": test_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "class_names": class_names,
        "confusion_matrix": confusion.tolist(),
        "num_test_samples": int(total),
    }
    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results JSON: {RESULTS_JSON_PATH}")
    print(f"Saved confusion matrix image: {CONFUSION_PNG_PATH}")


if __name__ == "__main__":
    main()
