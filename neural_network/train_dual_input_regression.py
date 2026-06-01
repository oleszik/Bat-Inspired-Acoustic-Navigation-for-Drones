"""
Train dual-input acoustic regression model on hard_v1 dataset.

Uses:
- spectrogram image input
- matched-filter correlation vector input
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from data_utils import preprocess_pil_image
from dual_input_model import AcousticDualInputRegressionCNN


DEFAULT_DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
DEFAULT_LABELS_FILE = "labels_with_correlation.csv"
DEFAULT_RUN_NAME = "hard_v1_dual"

BASE_CHECKPOINT_DIR = Path("neural_network/checkpoints")

IMAGE_SIZE = 128
CORR_LEN = 512
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

DISTANCE_MIN_M = 0.0
DISTANCE_MAX_M = 5.0


def normalize_corr_vector(x: np.ndarray) -> np.ndarray:
    """
    Safe per-sample normalization for correlation vector.
    """
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


class DualInputEchoDataset(Dataset):
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
                        distance_m = 0.0

                rows.append(
                    {
                        "spectrogram_path": r["spectrogram_path"],
                        "correlation_path": r["correlation_path"],
                        "has_obstacle": has_obstacle,
                        "distance_m": distance_m,
                    }
                )

        if not rows:
            raise ValueError(f"No rows found in {self.labels_csv}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        image_path = self.dataset_root / row["spectrogram_path"]
        corr_path = self.dataset_root / row["correlation_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing spectrogram image: {image_path}")
        if not corr_path.exists():
            raise FileNotFoundError(f"Missing correlation feature: {corr_path}")

        image = Image.open(image_path)
        image_tensor = preprocess_pil_image(image, image_size=self.image_size)

        corr_vec = np.load(corr_path)
        corr_vec = normalize_corr_vector(corr_vec)
        corr_tensor = torch.from_numpy(corr_vec)

        has_obstacle = float(row["has_obstacle"])
        distance_m = row["distance_m"]
        if math.isnan(distance_m):
            distance_m = 0.0
        distance_m = float(max(DISTANCE_MIN_M, min(DISTANCE_MAX_M, distance_m)))

        return (
            image_tensor,
            corr_tensor,
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


def compute_distance_mse(distance_pred_m: torch.Tensor, distance_true_m: torch.Tensor, has_obstacle: torch.Tensor):
    mask = has_obstacle > 0.5
    if mask.any():
        diff = distance_pred_m[mask] - distance_true_m[mask]
        return torch.mean(diff * diff)
    return distance_pred_m.new_tensor(0.0)


def evaluate(model: nn.Module, loader: DataLoader, bce_loss: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct = 0
    tp = 0
    fn = 0
    abs_errors_cm: list[float] = []
    sq_errors_cm: list[float] = []

    with torch.no_grad():
        for spec_img, corr_vec, has_obstacle, distance_m in loader:
            spec_img = spec_img.to(device)
            corr_vec = corr_vec.to(device)
            has_obstacle = has_obstacle.to(device)
            distance_m = distance_m.to(device)

            obstacle_logit, distance_pred_m = model(spec_img, corr_vec)
            obstacle_loss = bce_loss(obstacle_logit, has_obstacle)
            distance_loss = compute_distance_mse(distance_pred_m, distance_m, has_obstacle)
            total_loss = obstacle_loss + distance_loss

            running_loss += total_loss.item() * spec_img.size(0)
            total_samples += spec_img.size(0)

            prob = torch.sigmoid(obstacle_logit)
            pred = (prob >= 0.5).float()
            correct += int((pred == has_obstacle).sum().item())
            tp += int(((pred == 1) & (has_obstacle == 1)).sum().item())
            fn += int(((pred == 0) & (has_obstacle == 1)).sum().item())

            wall_mask = has_obstacle > 0.5
            if wall_mask.any():
                err_cm = (distance_pred_m[wall_mask] - distance_m[wall_mask]) * 100.0
                abs_errors_cm.extend(torch.abs(err_cm).cpu().tolist())
                sq_errors_cm.extend((err_cm * err_cm).cpu().tolist())

    val_loss = running_loss / max(total_samples, 1)
    val_acc = correct / max(total_samples, 1)
    val_recall = tp / max(tp + fn, 1)

    if abs_errors_cm:
        val_mae_cm = float(sum(abs_errors_cm) / len(abs_errors_cm))
        val_rmse_cm = float(math.sqrt(sum(sq_errors_cm) / len(sq_errors_cm)))
    else:
        val_mae_cm = float("nan")
        val_rmse_cm = float("nan")

    return val_loss, val_acc, val_recall, val_mae_cm, val_rmse_cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-input acoustic regression model.")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--labels-file", type=str, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    set_seed(SEED)

    dataset_root = Path(args.dataset_root)
    labels_csv = dataset_root / args.labels_file
    run_checkpoint_dir = BASE_CHECKPOINT_DIR / args.run_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_checkpoint_dir / "acoustic_dual_input_regression_best.pt"
    split_path = run_checkpoint_dir / "regression_split_indices.json"

    dataset = DualInputEchoDataset(dataset_root=dataset_root, labels_csv=labels_csv, image_size=IMAGE_SIZE)
    train_set, val_set, test_set = make_splits(dataset, seed=SEED)
    save_split_indices(train_set, val_set, test_set, split_path=split_path, seed=SEED)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Run name: {args.run_name}")
    print(f"Dataset size: total={len(dataset)}, train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    model = AcousticDualInputRegressionCNN().to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for spec_img, corr_vec, has_obstacle, distance_m in train_loader:
            spec_img = spec_img.to(device)
            corr_vec = corr_vec.to(device)
            has_obstacle = has_obstacle.to(device)
            distance_m = distance_m.to(device)

            optimizer.zero_grad()

            obstacle_logit, distance_pred_m = model(spec_img, corr_vec)
            obstacle_loss = bce_loss(obstacle_logit, has_obstacle)
            distance_loss = compute_distance_mse(distance_pred_m, distance_m, has_obstacle)
            loss = obstacle_loss + distance_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spec_img.size(0)
            seen += spec_img.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss, val_acc, val_recall, val_mae_cm, val_rmse_cm = evaluate(model, val_loader, bce_loss, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_obstacle_acc={val_acc * 100:.2f}% | "
            f"val_obstacle_recall={val_recall * 100:.2f}% | "
            f"val_distance_mae={val_mae_cm:.2f} cm | "
            f"val_distance_rmse={val_rmse_cm:.2f} cm"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "corr_len": CORR_LEN,
                },
                best_model_path,
            )
            print(f"  Saved best model -> {best_model_path}")

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
