"""
Train a two-head CNN for:
1) obstacle detection
2) continuous distance regression (for obstacle samples only)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from data_utils import preprocess_pil_image
from regression_model import AcousticRegressionCNN


DEFAULT_DATASET_ROOT = Path("datasets/synthetic_echoes_regression")
DEFAULT_RUN_NAME = "clean"
BASE_CHECKPOINT_DIR = Path("neural_network/checkpoints")

IMAGE_SIZE = 128
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

# Safety clamp range for distance labels (meters).
DISTANCE_MIN_M = 0.0
DISTANCE_MAX_M = 5.0


class RegressionEchoDataset(Dataset):
    """
    Dataset built from labels.csv.

    - has_obstacle is the binary target.
    - distance_m is only valid when has_obstacle == 1.
    """

    def __init__(self, dataset_root: Path, labels_csv: Path, image_size: int = 128) -> None:
        self.dataset_root = dataset_root
        self.labels_csv = labels_csv
        self.image_size = image_size
        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict]:
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Missing labels CSV: {self.labels_csv}")

        rows: list[dict] = []
        with self.labels_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                has_obstacle = int(r["has_obstacle"])
                distance_raw = (r.get("distance_m") or "").strip()
                distance_m = float("nan")

                if has_obstacle == 1:
                    try:
                        distance_m = float(distance_raw)
                    except ValueError:
                        # Invalid labels are made safe; distance loss will still be masked
                        # by has_obstacle and this value is clamped.
                        distance_m = 0.0

                rows.append(
                    {
                        "spectrogram_path": r["spectrogram_path"],
                        "has_obstacle": has_obstacle,
                        "distance_m": distance_m,
                    }
                )

        if not rows:
            raise ValueError(f"No rows loaded from {self.labels_csv}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = self.dataset_root / row["spectrogram_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing spectrogram image: {image_path}")

        image = Image.open(image_path)
        image_tensor = preprocess_pil_image(image, image_size=self.image_size)

        has_obstacle = float(row["has_obstacle"])
        distance_m = row["distance_m"]
        if math.isnan(distance_m):
            distance_m = 0.0

        distance_m = float(max(DISTANCE_MIN_M, min(DISTANCE_MAX_M, distance_m)))

        return (
            image_tensor,
            torch.tensor(has_obstacle, dtype=torch.float32),
            torch.tensor(distance_m, dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_splits(dataset: Dataset, seed: int):
    total = len(dataset)
    train_len = int(0.70 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=g)


def save_split_indices(train_set, val_set, test_set, split_path: Path, seed: int) -> None:
    split_data = {
        "seed": seed,
        "train_indices": list(train_set.indices),
        "val_indices": list(val_set.indices),
        "test_indices": list(test_set.indices),
    }
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)


def compute_distance_metrics(
    distance_pred_m: torch.Tensor,
    distance_true_m: torch.Tensor,
    has_obstacle: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """
    Compute masked distance loss/metrics for wall samples only.
    """
    mask = has_obstacle > 0.5
    if mask.any():
        diff = distance_pred_m[mask] - distance_true_m[mask]
        mse = torch.mean(diff * diff)
        mae_cm = torch.mean(torch.abs(diff)).item() * 100.0
        rmse_cm = torch.sqrt(mse).item() * 100.0
        return mse, mae_cm, rmse_cm

    zero = distance_pred_m.new_tensor(0.0)
    return zero, float("nan"), float("nan")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    bce_loss: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    running_total_loss = 0.0
    total_samples = 0
    obstacle_correct = 0

    # Accumulate wall-only errors for dataset-level MAE/RMSE.
    abs_errors_cm: list[float] = []
    squared_errors_cm: list[float] = []

    with torch.no_grad():
        for images, has_obstacle, distance_m in loader:
            images = images.to(device)
            has_obstacle = has_obstacle.to(device)
            distance_m = distance_m.to(device)

            obstacle_logit, distance_pred_m = model(images)

            obstacle_loss = bce_loss(obstacle_logit, has_obstacle)
            distance_loss, _, _ = compute_distance_metrics(distance_pred_m, distance_m, has_obstacle)
            total_loss = obstacle_loss + distance_loss

            running_total_loss += total_loss.item() * images.size(0)
            total_samples += images.size(0)

            obstacle_prob = torch.sigmoid(obstacle_logit)
            obstacle_pred = (obstacle_prob >= 0.5).float()
            obstacle_correct += int((obstacle_pred == has_obstacle).sum().item())

            mask = has_obstacle > 0.5
            if mask.any():
                diff_cm = (distance_pred_m[mask] - distance_m[mask]).abs() * 100.0
                abs_errors_cm.extend(diff_cm.cpu().tolist())
                squared_errors_cm.extend((diff_cm * diff_cm).cpu().tolist())

    avg_loss = running_total_loss / max(total_samples, 1)
    obstacle_acc = obstacle_correct / max(total_samples, 1)

    if abs_errors_cm:
        mae_cm = float(sum(abs_errors_cm) / len(abs_errors_cm))
        rmse_cm = float(math.sqrt(sum(squared_errors_cm) / len(squared_errors_cm)))
    else:
        mae_cm = float("nan")
        rmse_cm = float("nan")

    return avg_loss, obstacle_acc, mae_cm, rmse_cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-head acoustic regression CNN.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset root folder containing labels.csv and spectrograms/.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Run name used for checkpoint subfolder.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    dataset_root = Path(args.dataset_root)
    labels_csv = dataset_root / "labels.csv"
    run_checkpoint_dir = BASE_CHECKPOINT_DIR / args.run_name
    best_model_path = run_checkpoint_dir / "acoustic_regression_cnn_best.pt"
    split_path = run_checkpoint_dir / "regression_split_indices.json"
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = RegressionEchoDataset(dataset_root, labels_csv, image_size=IMAGE_SIZE)
    train_set, val_set, test_set = make_splits(dataset, seed=SEED)
    save_split_indices(train_set, val_set, test_set, split_path=split_path, seed=SEED)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Run name: {args.run_name}")
    print(f"Dataset size: total={len(dataset)}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    model = AcousticRegressionCNN().to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, has_obstacle, distance_m in train_loader:
            images = images.to(device)
            has_obstacle = has_obstacle.to(device)
            distance_m = distance_m.to(device)

            optimizer.zero_grad()

            obstacle_logit, distance_pred_m = model(images)
            obstacle_loss = bce_loss(obstacle_logit, has_obstacle)
            distance_loss, _, _ = compute_distance_metrics(distance_pred_m, distance_m, has_obstacle)
            total_loss = obstacle_loss + distance_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss, val_obstacle_acc, val_mae_cm, val_rmse_cm = evaluate(model, val_loader, bce_loss, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_obstacle_acc={val_obstacle_acc * 100:.2f}% | "
            f"val_distance_mae={val_mae_cm:.2f} cm | "
            f"val_distance_rmse={val_rmse_cm:.2f} cm"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                },
                best_model_path,
            )
            print(f"  Saved best model -> {best_model_path}")

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
