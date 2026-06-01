"""
Evaluate the trained two-head acoustic regression CNN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from data_utils import preprocess_pil_image
from regression_model import AcousticRegressionCNN


DEFAULT_DATASET_ROOT = Path("datasets/synthetic_echoes_regression")
DEFAULT_RUN_NAME = "clean"
BASE_CHECKPOINT_DIR = Path("neural_network/checkpoints")
BASE_RESULTS_DIR = Path("neural_network/results")

IMAGE_SIZE = 128
BATCH_SIZE = 32
DISTANCE_MIN_M = 0.0
DISTANCE_MAX_M = 5.0


class RegressionEchoDataset(Dataset):
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
                        "has_obstacle": has_obstacle,
                        "distance_m": distance_m,
                    }
                )
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


def load_model(device: torch.device, best_model_path: Path) -> AcousticRegressionCNN:
    if not best_model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model = AcousticRegressionCNN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-head acoustic regression CNN.")
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
        help="Run name used for checkpoint and result subfolders.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    labels_csv = dataset_root / "labels.csv"
    run_checkpoint_dir = BASE_CHECKPOINT_DIR / args.run_name
    best_model_path = run_checkpoint_dir / "acoustic_regression_cnn_best.pt"
    split_path = run_checkpoint_dir / "regression_split_indices.json"
    run_results_dir = BASE_RESULTS_DIR / args.run_name
    results_json_path = run_results_dir / "regression_test_results.json"
    scatter_path = run_results_dir / "regression_true_vs_predicted.png"
    hist_path = run_results_dir / "regression_error_histogram.png"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split indices: {split_path}")

    run_results_dir.mkdir(parents=True, exist_ok=True)

    dataset = RegressionEchoDataset(dataset_root, labels_csv, image_size=IMAGE_SIZE)
    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_indices = split_data["test_indices"]
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, best_model_path=best_model_path)
    print(f"Dataset root: {dataset_root}")
    print(f"Run name: {args.run_name}")

    tp = fp = tn = fn = 0
    true_dist_m: list[float] = []
    pred_dist_m: list[float] = []
    abs_err_cm: list[float] = []
    sq_err_cm: list[float] = []

    with torch.no_grad():
        for images, has_obstacle, distance_m in test_loader:
            images = images.to(device)
            has_obstacle = has_obstacle.to(device)
            distance_m = distance_m.to(device)

            obstacle_logit, distance_pred_m = model(images)
            obstacle_prob = torch.sigmoid(obstacle_logit)
            obstacle_pred = (obstacle_prob >= 0.5).float()

            # Obstacle confusion counts.
            tp += int(((obstacle_pred == 1) & (has_obstacle == 1)).sum().item())
            fp += int(((obstacle_pred == 1) & (has_obstacle == 0)).sum().item())
            tn += int(((obstacle_pred == 0) & (has_obstacle == 0)).sum().item())
            fn += int(((obstacle_pred == 0) & (has_obstacle == 1)).sum().item())

            # Distance metrics: wall samples only.
            wall_mask = has_obstacle > 0.5
            if wall_mask.any():
                y_true = distance_m[wall_mask]
                y_pred = distance_pred_m[wall_mask]
                err_cm = (y_pred - y_true) * 100.0

                true_dist_m.extend(y_true.cpu().tolist())
                pred_dist_m.extend(y_pred.cpu().tolist())
                abs_err_cm.extend(torch.abs(err_cm).cpu().tolist())
                sq_err_cm.extend((err_cm * err_cm).cpu().tolist())

    total = tp + tn + fp + fn
    obstacle_accuracy = (tp + tn) / max(total, 1)
    obstacle_precision = tp / max(tp + fp, 1)
    obstacle_recall = tp / max(tp + fn, 1)

    if abs_err_cm:
        distance_mae_cm = float(sum(abs_err_cm) / len(abs_err_cm))
        distance_rmse_cm = float(math.sqrt(sum(sq_err_cm) / len(sq_err_cm)))
        max_abs_error_cm = float(max(abs_err_cm))
    else:
        distance_mae_cm = float("nan")
        distance_rmse_cm = float("nan")
        max_abs_error_cm = float("nan")

    print(f"Obstacle accuracy: {obstacle_accuracy * 100:.2f}%")
    print(f"Obstacle precision: {obstacle_precision * 100:.2f}%")
    print(f"Obstacle recall: {obstacle_recall * 100:.2f}%")
    print(f"Distance MAE (wall only): {distance_mae_cm:.2f} cm")
    print(f"Distance RMSE (wall only): {distance_rmse_cm:.2f} cm")
    print(f"Max abs distance error (wall only): {max_abs_error_cm:.2f} cm")

    # Scatter plot: true vs predicted for wall samples.
    plt.figure(figsize=(6, 6), dpi=140)
    plt.scatter(true_dist_m, pred_dist_m, s=12, alpha=0.65)
    if true_dist_m:
        lo = min(true_dist_m)
        hi = max(true_dist_m)
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="red", linewidth=1.2, label="Ideal y=x")
        plt.legend()
    plt.xlabel("True distance (m)")
    plt.ylabel("Predicted distance (m)")
    plt.title("Regression: True vs Predicted Distance")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    # Error histogram in centimeters.
    plt.figure(figsize=(7, 4), dpi=140)
    plt.hist(abs_err_cm, bins=40, color="tab:blue", alpha=0.85)
    plt.xlabel("Absolute distance error (cm)")
    plt.ylabel("Count")
    plt.title("Regression Distance Error Histogram (Wall Samples)")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    results = {
        "num_test_samples": total,
        "num_wall_samples_for_distance_metrics": len(abs_err_cm),
        "obstacle_accuracy": obstacle_accuracy,
        "obstacle_precision": obstacle_precision,
        "obstacle_recall": obstacle_recall,
        "distance_mae_cm_wall_only": distance_mae_cm,
        "distance_rmse_cm_wall_only": distance_rmse_cm,
        "max_abs_distance_error_cm_wall_only": max_abs_error_cm,
        "confusion_counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results JSON: {results_json_path}")
    print(f"Saved scatter plot: {scatter_path}")
    print(f"Saved error histogram: {hist_path}")


if __name__ == "__main__":
    main()
