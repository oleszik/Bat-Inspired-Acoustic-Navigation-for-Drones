"""
Train a feature-based echo-validity classifier.

valid_echo label:
- obstacle_easy, obstacle_weak -> 1
- clear_* classes -> 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT = Path("datasets/clear_space_diagnostic_v1")
FEATURES_CSV = DATASET_ROOT / "clear_space_features.csv"

CHECKPOINT_DIR = Path("neural_network/checkpoints/echo_validity")
MODEL_PATH = CHECKPOINT_DIR / "echo_validity_classifier_best.pt"
NORM_PATH = CHECKPOINT_DIR / "feature_normalization.json"
SPLIT_PATH = CHECKPOINT_DIR / "split_indices.json"

FEATURE_COLS = [
    "strongest_peak_value",
    "strongest_peak_delay_ms",
    "first_noise_floor_peak_value",
    "first_noise_floor_peak_delay_ms",
    "peak_snr",
    "peak_prominence",
    "peak_width",
    "noise_floor",
    "num_peaks",
]

POS_CLASSES = {"obstacle_easy", "obstacle_weak"}

BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
LR = 1e-3
SEED = 42


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class EchoValidityMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_split_indices(n: int, seed: int):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def eval_binary(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pred = (torch.sigmoid(logits) >= 0.5).float()
            tp += int(((pred == 1) & (y == 1)).sum().item())
            fp += int(((pred == 1) & (y == 0)).sum().item())
            tn += int(((pred == 0) & (y == 0)).sum().item())
            fn += int(((pred == 0) & (y == 1)).sum().item())

    loss = total_loss / max(total, 1)
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fnr = fn / max(tp + fn, 1)
    return {
        "loss": loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "fnr": fnr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train echo-validity feature classifier.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Missing feature CSV: {FEATURES_CSV}\nRun signal_processing/evaluate_clear_space_features_v1.py first."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    if "class_name" not in df.columns:
        raise ValueError("clear_space_features.csv must contain class_name column.")

    y = df["class_name"].apply(lambda c: 1 if c in POS_CLASSES else 0).to_numpy(dtype=np.float32)
    x_df = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")

    train_idx, val_idx, test_idx = make_split_indices(len(df), SEED)
    with SPLIT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "test_indices": test_idx.tolist(),
            },
            f,
            indent=2,
        )

    # Train-only normalization.
    train_x = x_df.iloc[train_idx].copy()
    mean = train_x.mean(axis=0, skipna=True)
    std = train_x.std(axis=0, skipna=True, ddof=0).mask(lambda s: s < 1e-6, 1.0)

    x_filled = x_df.fillna(mean)
    x_norm = (x_filled - mean) / std
    x_np = np.nan_to_num(x_norm.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    with NORM_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLS,
                "mean": {k: float(mean[k]) for k in FEATURE_COLS},
                "std": {k: float(std[k]) for k in FEATURE_COLS},
            },
            f,
            indent=2,
        )

    train_ds = FeatureDataset(x_np[train_idx], y[train_idx])
    val_ds = FeatureDataset(x_np[val_idx], y[val_idx])
    test_ds = FeatureDataset(x_np[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset size: total={len(df)}, train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = EchoValidityMLP(in_dim=len(FEATURE_COLS)).to(device)

    # Slightly weight positive class to reduce false negatives on obstacle echoes.
    pos_count = float(y[train_idx].sum())
    neg_count = float(len(train_idx) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_key = (float("inf"), float("inf"), float("inf"))  # (fnr, -recall -> via 1-recall, loss)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * x_batch.size(0)
            seen += x_batch.size(0)

        train_loss = running / max(seen, 1)
        val = eval_binary(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | val_acc={val['accuracy'] * 100:.2f}% | "
            f"val_precision={val['precision'] * 100:.2f}% | val_recall={val['recall'] * 100:.2f}% | "
            f"val_fnr={val['fnr'] * 100:.2f}%"
        )

        key = (val["fnr"], 1.0 - val["recall"], val["loss"])
        if key < best_key:
            best_key = key
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": FEATURE_COLS,
                    "selection_key": {"val_fnr": val["fnr"], "val_recall": val["recall"], "val_loss": val["loss"]},
                },
                MODEL_PATH,
            )
            print(f"  Saved best model -> {MODEL_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    main()
