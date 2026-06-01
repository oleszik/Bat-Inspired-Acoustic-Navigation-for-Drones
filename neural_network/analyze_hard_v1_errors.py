"""
Analyze hard_v1 regression failure modes without retraining.

This script:
1) loads the hard_v1 test split,
2) runs predictions with the saved model checkpoint,
3) saves per-sample error diagnostics,
4) prints grouped summary statistics,
5) saves diagnostic plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from data_utils import preprocess_pil_image
from regression_model import AcousticRegressionCNN


DEFAULT_DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
DEFAULT_RUN_NAME = "hard_v1"
BASE_CHECKPOINT_DIR = Path("neural_network/checkpoints")
BASE_RESULTS_DIR = Path("neural_network/results")

IMAGE_SIZE = 128


def load_model(checkpoint_path: Path, device: torch.device) -> AcousticRegressionCNN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AcousticRegressionCNN().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def print_grouped_means(df_wall: pd.DataFrame) -> None:
    print("\nMean error by noise_level bins (cm):")
    noise_bins = [0.0, 0.003, 0.006, 0.010, 0.020, 0.040, np.inf]
    noise_labels = ["0-0.003", "0.003-0.006", "0.006-0.010", "0.010-0.020", "0.020-0.040", "0.040+"]
    noise_group = df_wall.copy()
    noise_group["noise_bin"] = pd.cut(
        noise_group["noise_level"], bins=noise_bins, labels=noise_labels, include_lowest=True, right=False
    )
    print(noise_group.groupby("noise_bin", dropna=False, observed=False)["absolute_distance_error_cm"].mean())

    print("\nMean error by reflection_strength bins (cm):")
    refl_bins = [0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.20, np.inf]
    refl_labels = ["0-0.10", "0.10-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.20", "1.20+"]
    refl_group = df_wall.copy()
    refl_group["reflection_bin"] = pd.cut(
        refl_group["reflection_strength"], bins=refl_bins, labels=refl_labels, include_lowest=True, right=False
    )
    print(refl_group.groupby("reflection_bin", dropna=False, observed=False)["absolute_distance_error_cm"].mean())

    print("\nMean error with vs without secondary reflection (cm):")
    print(df_wall.groupby("has_secondary_reflection")["absolute_distance_error_cm"].mean())

    print("\nMean error by distance range (cm):")
    dist_bins = [0.10, 0.50, 1.00, 1.50, 2.00, 2.50]
    dist_labels = ["0.10-0.50 m", "0.50-1.00 m", "1.00-1.50 m", "1.50-2.00 m", "2.00-2.50 m"]
    dist_group = df_wall.copy()
    dist_group["distance_range"] = pd.cut(
        dist_group["true_distance_m"], bins=dist_bins, labels=dist_labels, include_lowest=True, right=False
    )
    print(dist_group.groupby("distance_range", dropna=False, observed=False)["absolute_distance_error_cm"].mean())


def save_plots(df: pd.DataFrame, df_wall: pd.DataFrame, results_dir: Path) -> None:
    # error_cm vs noise_level
    plt.figure(figsize=(6.5, 4.2), dpi=140)
    plt.scatter(df_wall["noise_level"], df_wall["absolute_distance_error_cm"], s=12, alpha=0.55)
    plt.xlabel("noise_level")
    plt.ylabel("absolute_distance_error_cm")
    plt.title("Error vs Noise Level (Wall Samples)")
    plt.tight_layout()
    plt.savefig(results_dir / "hard_v1_error_vs_noise_level.png")
    plt.close()

    # error_cm vs reflection_strength
    plt.figure(figsize=(6.5, 4.2), dpi=140)
    plt.scatter(df_wall["reflection_strength"], df_wall["absolute_distance_error_cm"], s=12, alpha=0.55)
    plt.xlabel("reflection_strength")
    plt.ylabel("absolute_distance_error_cm")
    plt.title("Error vs Reflection Strength (Wall Samples)")
    plt.tight_layout()
    plt.savefig(results_dir / "hard_v1_error_vs_reflection_strength.png")
    plt.close()

    # error_cm vs true_distance_m
    plt.figure(figsize=(6.5, 4.2), dpi=140)
    plt.scatter(df_wall["true_distance_m"], df_wall["absolute_distance_error_cm"], s=12, alpha=0.55)
    plt.xlabel("true_distance_m")
    plt.ylabel("absolute_distance_error_cm")
    plt.title("Error vs True Distance (Wall Samples)")
    plt.tight_layout()
    plt.savefig(results_dir / "hard_v1_error_vs_true_distance.png")
    plt.close()

    # predicted obstacle probability vs reflection_strength
    plt.figure(figsize=(6.5, 4.2), dpi=140)
    colors = np.where(df["has_obstacle"] == 1, "tab:blue", "tab:orange")
    plt.scatter(df["reflection_strength"], df["predicted_obstacle_probability"], c=colors, s=12, alpha=0.55)
    plt.xlabel("reflection_strength")
    plt.ylabel("predicted_obstacle_probability")
    plt.title("Predicted Obstacle Probability vs Reflection Strength")
    plt.tight_layout()
    plt.savefig(results_dir / "hard_v1_obstacle_prob_vs_reflection_strength.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hard_v1 regression errors.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset root containing labels.csv and spectrograms/.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Run name used for checkpoint and output folders.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    labels_csv = dataset_root / "labels.csv"
    checkpoint_dir = BASE_CHECKPOINT_DIR / args.run_name
    checkpoint_path = checkpoint_dir / "acoustic_regression_cnn_best.pt"
    split_path = checkpoint_dir / "regression_split_indices.json"
    results_dir = BASE_RESULTS_DIR / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    df_labels = pd.read_csv(labels_csv)
    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_indices = split_data["test_indices"]
    df_test = df_labels.iloc[test_indices].copy().reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path=checkpoint_path, device=device)

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for _, row in df_test.iterrows():
            image_path = dataset_root / str(row["spectrogram_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing spectrogram image: {image_path}")

            image = Image.open(image_path).convert("RGB")
            x = preprocess_pil_image(image, image_size=IMAGE_SIZE).unsqueeze(0).to(device)

            obstacle_logit, distance_pred_m = model(x)
            obstacle_prob = float(torch.sigmoid(obstacle_logit).item())
            pred_has_obstacle = int(obstacle_prob >= 0.5)
            pred_distance_m = float(distance_pred_m.item())

            has_obstacle = int(row["has_obstacle"])
            true_distance = row["distance_m"]
            if has_obstacle == 1 and pd.notna(true_distance):
                abs_error_cm = abs(pred_distance_m - float(true_distance)) * 100.0
                true_distance_val = float(true_distance)
            else:
                abs_error_cm = np.nan
                true_distance_val = np.nan

            rows.append(
                {
                    "filename": row["filename"],
                    "has_obstacle": has_obstacle,
                    "true_distance_m": true_distance_val,
                    "predicted_obstacle_probability": obstacle_prob,
                    "predicted_has_obstacle": pred_has_obstacle,
                    "predicted_distance_m": pred_distance_m,
                    "absolute_distance_error_cm": abs_error_cm,
                    "reflection_strength": row["reflection_strength"],
                    "noise_level": row["noise_level"],
                    "surface_absorption": row["surface_absorption"],
                    "has_secondary_reflection": row["has_secondary_reflection"],
                    "secondary_delay_ms": row["secondary_delay_ms"],
                    "secondary_strength": row["secondary_strength"],
                }
            )

    df = pd.DataFrame(rows)

    # Save full per-sample results.
    per_sample_csv = results_dir / "hard_v1_per_sample_errors.csv"
    df.to_csv(per_sample_csv, index=False)

    # Save top-30 worst wall errors.
    df_wall = df[df["has_obstacle"] == 1].copy()
    top30 = df_wall.sort_values("absolute_distance_error_cm", ascending=False).head(30)
    top30_csv = results_dir / "hard_v1_top30_worst_errors.csv"
    top30.to_csv(top30_csv, index=False)

    # Save false negatives (true wall but predicted no obstacle).
    false_negatives = df[(df["has_obstacle"] == 1) & (df["predicted_has_obstacle"] == 0)].copy()
    false_neg_csv = results_dir / "hard_v1_false_negatives.csv"
    false_negatives.to_csv(false_neg_csv, index=False)

    # Print requested summary statistics.
    print_grouped_means(df_wall)
    print(f"\nNumber of false negatives: {len(false_negatives)}")

    # Save requested plots.
    save_plots(df=df, df_wall=df_wall, results_dir=results_dir)

    print(f"\nSaved per-sample CSV: {per_sample_csv}")
    print(f"Saved top-30 worst CSV: {top30_csv}")
    print(f"Saved false negatives CSV: {false_neg_csv}")
    print(f"Saved plots in: {results_dir}")


if __name__ == "__main__":
    main()
