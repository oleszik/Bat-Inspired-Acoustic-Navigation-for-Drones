from __future__ import annotations

from typing import Dict

import numpy as np


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    tn = float(np.logical_and(~pred_b, ~gt_b).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1.0, tp + fp + fn + tn)
    iou = tp / max(1.0, tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "iou": iou,
    }
