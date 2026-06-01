"""
Evaluate echo-validity feature classifier on held-out test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT = Path("datasets/clear_space_diagnostic_v1")
FEATURES_CSV = DATASET_ROOT / "clear_space_features.csv"

CHECKPOINT_DIR = Path("neural_network/checkpoints/echo_validity")
MODEL_PATH = CHECKPOINT_DIR / "echo_validity_classifier_best.pt"
NORM_PATH = CHECKPOINT_DIR / "feature_normalization.json"
SPLIT_PATH = CHECKPOINT_DIR / "split_indices.json"

RESULT_DIR = Path("neural_network/results/echo_validity")
RESULT_JSON = RESULT_DIR / "echo_validity_results.json"
CONF_PNG = RESULT_DIR / "echo_validity_confusion_matrix.png"
HIST_PNG = RESULT_DIR / "echo_validity_score_histogram.png"

POS_CLASSES = {"obstacle_easy", "obstacle_weak"}


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


def main() -> None:
    for p in [FEATURES_CSV, MODEL_PATH, NORM_PATH, SPLIT_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    with NORM_PATH.open("r", encoding="utf-8") as f:
        norm = json.load(f)
    with SPLIT_PATH.open("r", encoding="utf-8") as f:
        split = json.load(f)
    feature_cols = norm["feature_columns"]
    mean = pd.Series(norm["mean"], dtype=float)
    std = pd.Series(norm["std"], dtype=float).mask(lambda s: s < 1e-6, 1.0)

    y = df["class_name"].apply(lambda c: 1 if c in POS_CLASSES else 0).to_numpy(dtype=np.float32)
    x_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(mean)
    x_norm = (x_df - mean) / std
    x_np = np.nan_to_num(x_norm.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    test_idx = np.asarray(split["test_indices"], dtype=np.int64)
    test_ds = FeatureDataset(x_np[test_idx], y[test_idx])
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = EchoValidityMLP(in_dim=len(feature_cols)).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    probs_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            y_all.append(yb.numpy())

    probs = np.concatenate(probs_all)
    y_true = np.concatenate(y_all).astype(int)
    y_pred = (probs >= 0.5).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total = len(y_true)
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"False positive rate: {fpr * 100:.2f}%")
    print(f"False negative rate: {fnr * 100:.2f}%")
    print(f"Confusion matrix [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")

    # Confusion matrix plot.
    conf = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=140)
    im = ax.imshow(conf, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xticks([0, 1], labels=["Pred Clear/Noise", "Pred Valid Echo"])
    ax.set_yticks([0, 1], labels=["True Clear/Noise", "True Valid Echo"])
    ax.set_title("Echo Validity Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=11, color="black")
    fig.tight_layout()
    fig.savefig(CONF_PNG)
    plt.close(fig)

    # Score histogram.
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=140)
    ax.hist(probs[y_true == 0], bins=40, alpha=0.7, label="True clear/noise (0)")
    ax.hist(probs[y_true == 1], bins=40, alpha=0.7, label="True valid echo (1)")
    ax.set_xlabel("Predicted valid-echo probability")
    ax.set_ylabel("Count")
    ax.set_title("Echo Validity Score Histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HIST_PNG)
    plt.close(fig)

    result = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "num_test_samples": int(total),
    }
    with RESULT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results JSON: {RESULT_JSON}")
    print(f"Saved confusion matrix plot: {CONF_PNG}")
    print(f"Saved score histogram: {HIST_PNG}")


if __name__ == "__main__":
    main()
