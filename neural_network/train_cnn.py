"""
Train AcousticCNN on synthetic echo-focused spectrograms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import SimpleImageFolder, make_splits
from model import AcousticCNN


DATA_ROOT = Path("datasets/synthetic_echoes/spectrograms_echo_focus")
CHECKPOINT_DIR = Path("neural_network/checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "acoustic_cnn_best.pt"
CLASS_MAP_PATH = CHECKPOINT_DIR / "class_to_idx.json"
SPLIT_META_PATH = CHECKPOINT_DIR / "split_indices.json"

IMAGE_SIZE = 128
BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-3
SEED = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_split_metadata(train_set, val_set, test_set) -> None:
    """
    Save split indices so evaluation can use the exact same test subset.
    """
    split_data = {
        "seed": SEED,
        "train_indices": list(train_set.indices),
        "val_indices": list(val_set.indices),
        "test_indices": list(test_set.indices),
    }
    with SPLIT_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AcousticCNN on synthetic spectrograms.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate for Adam.")
    args = parser.parse_args()

    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SimpleImageFolder(root=DATA_ROOT, image_size=IMAGE_SIZE)
    train_set, val_set, test_set = make_splits(dataset, seed=SEED)
    save_split_metadata(train_set, val_set, test_set)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    with CLASS_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {dataset.classes}")
    print(f"Dataset size: total={len(dataset)}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    model = AcousticCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            train_total += labels.size(0)

        train_loss = running_train_loss / max(train_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )

        # Save model with best validation accuracy.
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(dataset.classes),
                    "image_size": IMAGE_SIZE,
                },
                BEST_MODEL_PATH,
            )
            print(f"  Saved new best model -> {BEST_MODEL_PATH}")

    print(f"Training complete. Best validation accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
