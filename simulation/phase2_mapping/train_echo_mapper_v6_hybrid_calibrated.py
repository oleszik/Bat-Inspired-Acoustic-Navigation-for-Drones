from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .echo_dataset import EchoMappingNPZDataset
    from .train_echo_mapper_v4 import EchoMapperV2 as V4Model
    from .train_echo_mapper_v5 import EchoMapperV3 as V5Model
except ImportError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.phase2_mapping.echo_dataset import EchoMappingNPZDataset  # type: ignore
    from simulation.phase2_mapping.train_echo_mapper_v4 import EchoMapperV2 as V4Model  # type: ignore
    from simulation.phase2_mapping.train_echo_mapper_v5 import EchoMapperV3 as V5Model  # type: ignore


NON_DOORWAY_MAPS = ["empty_room", "corridor", "single_block", "cluttered_room"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2C.5 hybrid calibrated mapper: v4/v5 checkpoint fusion and threshold calibration."
    )
    parser.add_argument("--data-dir", type=str, default="datasets/phase2_echo_mapping")
    parser.add_argument("--v4-run-dir", type=str, default="runs/phase2_echo_mapper_v4")
    parser.add_argument("--v5-run-dir", type=str, default="runs/phase2_echo_mapper_v5")
    parser.add_argument("--output-dir", type=str, default="runs/phase2_echo_mapper_v6_hybrid_calibrated")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument(
        "--threshold-candidates",
        type=str,
        default="0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--doorway-fp-lambda", type=float, default=0.35)
    parser.add_argument("--doorway-recall-floor", type=float, default=0.70)
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def binary_stats(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    tp = float(np.logical_and(pred, tgt).sum())
    fp = float(np.logical_and(pred, np.logical_not(tgt)).sum())
    fn = float(np.logical_and(np.logical_not(pred), tgt).sum())
    tn = float(np.logical_and(np.logical_not(pred), np.logical_not(tgt)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    iou = tp / max(1.0, tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "iou": iou,
    }


def _shift_no_wrap(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(a)
    y_src_start = max(0, -dy)
    y_src_end = a.shape[1] - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end = a.shape[2] - max(0, dx)

    y_dst_start = max(0, dy)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)

    if y_src_end > y_src_start and x_src_end > x_src_start:
        out[:, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = a[:, y_src_start:y_src_end, x_src_start:x_src_end]
    return out


def apply_doorway_structural_gating(
    door_prob_2d: np.ndarray,
    wall_prob_2d: np.ndarray,
    free_prob_2d: np.ndarray,
    occ_prob_2d: np.ndarray,
    patch_size: int,
    gate_strength: float,
    min_gate: float,
    structure_weight: float,
) -> np.ndarray:
    n = door_prob_2d.shape[0]
    d = door_prob_2d.reshape(n, patch_size, patch_size)
    w = wall_prob_2d.reshape(n, patch_size, patch_size)
    f = free_prob_2d.reshape(n, patch_size, patch_size)
    o = occ_prob_2d.reshape(n, patch_size, patch_size)

    h1 = np.minimum(_shift_no_wrap(w, 0, -1), _shift_no_wrap(w, 0, 1))
    h2 = np.minimum(_shift_no_wrap(w, 0, -2), _shift_no_wrap(w, 0, 2))
    v1 = np.minimum(_shift_no_wrap(w, -1, 0), _shift_no_wrap(w, 1, 0))
    v2 = np.minimum(_shift_no_wrap(w, -2, 0), _shift_no_wrap(w, 2, 0))
    wall_support = np.maximum.reduce([h1, h2, v1, v2])

    neigh = [
        _shift_no_wrap(w, -1, -1),
        _shift_no_wrap(w, -1, 0),
        _shift_no_wrap(w, -1, 1),
        _shift_no_wrap(w, 0, -1),
        _shift_no_wrap(w, 0, 1),
        _shift_no_wrap(w, 1, -1),
        _shift_no_wrap(w, 1, 0),
        _shift_no_wrap(w, 1, 1),
    ]
    local_wall = np.mean(np.stack(neigh, axis=0), axis=0)

    support_gate = np.clip((wall_support - 0.25) / 0.45, 0.0, 1.0)
    free_gate = np.clip((f - 0.35) / 0.45, 0.0, 1.0)
    occ_gate = np.clip((0.65 - o) / 0.65, 0.0, 1.0)
    structure_score = np.clip((0.6 * support_gate + 0.2 * free_gate + 0.2 * occ_gate), 0.0, 1.0)

    open_empty = ((f > 0.80) & (o < 0.18) & (local_wall < 0.10)).astype(np.float32)
    isolated_object_edge = ((o > 0.55) & (wall_support < 0.20)).astype(np.float32)
    suppress = np.clip(0.75 * open_empty + 0.45 * isolated_object_edge, 0.0, 1.0)
    structure_score = np.clip(structure_score * (1.0 - structure_weight * suppress), 0.0, 1.0)

    base_gate = min_gate + (1.0 - min_gate) * structure_score
    gate = (1.0 - gate_strength) + gate_strength * base_gate
    return np.clip(d * gate, 0.0, 1.0).reshape(door_prob_2d.shape[0], -1)


def non_doorway_fp_rates(
    door_pred_flat: np.ndarray,
    door_gt_flat: np.ndarray,
    map_idx: np.ndarray,
    map_names: List[str],
) -> Tuple[Dict[str, float], float]:
    fpr_by_map: Dict[str, float] = {}
    vals = []
    for name in NON_DOORWAY_MAPS:
        if name not in map_names:
            continue
        idx = map_names.index(name)
        mask_map = map_idx == idx
        if not np.any(mask_map):
            fpr_by_map[name] = 0.0
            vals.append(0.0)
            continue
        gt_neg = door_gt_flat[mask_map].sum(axis=1) == 0
        if np.any(gt_neg):
            pred_pos = door_pred_flat[mask_map][gt_neg].any(axis=1)
            fpr = float(pred_pos.mean())
        else:
            fpr = 0.0
        fpr_by_map[name] = fpr
        vals.append(fpr)
    avg_fpr = float(np.mean(vals)) if vals else 0.0
    return fpr_by_map, avg_fpr


def collect_model_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    use_soft_gating: bool,
    gate_strength: float,
    min_gate: float,
    structure_weight: float,
    has_context_head: bool,
) -> Dict[str, np.ndarray]:
    model.eval()
    occ, wall, door, free = [], [], [], []
    occ_t, wall_t, door_t, free_t = [], [], [], []
    map_idx, diff_idx = [], []
    context_prob = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch["signal"], batch["meta"])

            occ_prob = torch.sigmoid(out["occupancy_logits"]).cpu().numpy()
            wall_prob = torch.sigmoid(out["wall_logits"]).cpu().numpy()
            door_prob = torch.sigmoid(out["doorway_logits"]).cpu().numpy()
            free_prob = torch.sigmoid(out["free_logits"]).cpu().numpy()
            if has_context_head and ("doorway_context_logit" in out):
                cprob = torch.sigmoid(out["doorway_context_logit"]).cpu().numpy()
            else:
                cprob = np.ones((door_prob.shape[0],), dtype=np.float32)

            occ.append(occ_prob)
            wall.append(wall_prob)
            door.append(door_prob)
            free.append(free_prob)
            context_prob.append(cprob)

            occ_t.append(batch["occupancy"].cpu().numpy() >= 0.5)
            wall_t.append(batch["wall"].cpu().numpy() >= 0.5)
            door_t.append(batch["doorway"].cpu().numpy() >= 0.5)
            free_t.append(batch["free"].cpu().numpy() >= 0.5)
            map_idx.append(batch["map_index"].cpu().numpy())
            diff_idx.append(batch["difficulty_index"].cpu().numpy())

    out_np = {
        "occ_prob": np.concatenate(occ, axis=0),
        "wall_prob": np.concatenate(wall, axis=0),
        "door_prob_raw": np.concatenate(door, axis=0),
        "free_prob": np.concatenate(free, axis=0),
        "context_prob": np.concatenate(context_prob, axis=0),
        "occ_t": np.concatenate(occ_t, axis=0),
        "wall_t": np.concatenate(wall_t, axis=0),
        "door_t": np.concatenate(door_t, axis=0),
        "free_t": np.concatenate(free_t, axis=0),
        "map_idx": np.concatenate(map_idx, axis=0),
        "diff_idx": np.concatenate(diff_idx, axis=0),
    }

    if use_soft_gating:
        out_np["door_prob_soft"] = apply_doorway_structural_gating(
            door_prob_2d=out_np["door_prob_raw"],
            wall_prob_2d=out_np["wall_prob"],
            free_prob_2d=out_np["free_prob"],
            occ_prob_2d=out_np["occ_prob"],
            patch_size=patch_size,
            gate_strength=gate_strength,
            min_gate=min_gate,
            structure_weight=structure_weight,
        )
    else:
        out_np["door_prob_soft"] = out_np["door_prob_raw"].copy()

    out_np["door_prob_context"] = out_np["door_prob_raw"] * out_np["context_prob"][:, None]
    out_np["door_prob_final"] = out_np["door_prob_soft"] * out_np["context_prob"][:, None]
    return out_np


def scan_threshold(
    prob: np.ndarray,
    tgt: np.ndarray,
    thresholds: Iterable[float],
    metric: str,
) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5
    best_metric = -1.0
    best_stats: Dict[str, float] = {}
    for t in thresholds:
        s = binary_stats(prob >= t, tgt)
        val = float(s[metric])
        if val > best_metric:
            best_metric = val
            best_t = float(t)
            best_stats = s
    return best_t, best_stats


def doorway_candidate_score(
    pred: np.ndarray,
    door_t: np.ndarray,
    map_idx: np.ndarray,
    diff_idx: np.ndarray,
    map_names: List[str],
    diff_names: List[str],
    fp_lambda: float,
    recall_floor: float,
) -> Dict[str, float]:
    stats = binary_stats(pred, door_t)
    fpr_by_map, avg_fpr = non_doorway_fp_rates(pred, door_t, map_idx, map_names)

    door_map_f1 = 0.0
    if "doorway" in map_names:
        didx = map_names.index("doorway")
        m = map_idx == didx
        if np.any(m):
            door_map_f1 = binary_stats(pred[m], door_t[m])["f1"]

    hard_f1 = 0.0
    if "hard_noise" in diff_names:
        hidx = diff_names.index("hard_noise")
        m = diff_idx == hidx
        if np.any(m):
            hard_f1 = binary_stats(pred[m], door_t[m])["f1"]

    score = (
        1.0 * stats["f1"]
        + 0.40 * door_map_f1
        + 0.35 * hard_f1
        - fp_lambda * avg_fpr
    )
    if stats["recall"] < recall_floor:
        score -= (recall_floor - stats["recall"]) * 2.0

    return {
        "precision": stats["precision"],
        "recall": stats["recall"],
        "f1": stats["f1"],
        "doorway_map_f1": door_map_f1,
        "hard_noise_f1": hard_f1,
        "fp_empty_room": fpr_by_map.get("empty_room", 0.0),
        "fp_corridor": fpr_by_map.get("corridor", 0.0),
        "fp_single_block": fpr_by_map.get("single_block", 0.0),
        "fp_cluttered_room": fpr_by_map.get("cluttered_room", 0.0),
        "avg_non_doorway_fp_rate": avg_fpr,
        "combined_score": score,
    }


def select_best_doorway_mode(
    candidates: Dict[str, np.ndarray],
    door_t: np.ndarray,
    map_idx: np.ndarray,
    diff_idx: np.ndarray,
    map_names: List[str],
    diff_names: List[str],
    thresholds: Iterable[float],
    fp_lambda: float,
    recall_floor: float,
) -> Tuple[str, float, Dict[str, float], List[Dict[str, float]]]:
    best_mode = "v5_final"
    best_thr = 0.7
    best_row: Dict[str, float] = {}
    best_score = -1e9
    table: List[Dict[str, float]] = []

    for mode, prob in candidates.items():
        for thr in thresholds:
            pred = prob >= thr
            row = doorway_candidate_score(
                pred=pred,
                door_t=door_t,
                map_idx=map_idx,
                diff_idx=diff_idx,
                map_names=map_names,
                diff_names=diff_names,
                fp_lambda=fp_lambda,
                recall_floor=recall_floor,
            )
            row.update({"mode": mode, "threshold": float(thr)})
            table.append(row)
            if row["combined_score"] > best_score:
                best_score = row["combined_score"]
                best_mode = mode
                best_thr = float(thr)
                best_row = row

    return best_mode, best_thr, best_row, table


def evaluate_combined(
    occ_prob: np.ndarray,
    wall_prob: np.ndarray,
    door_prob: np.ndarray,
    free_prob: np.ndarray,
    occ_t: np.ndarray,
    wall_t: np.ndarray,
    door_t: np.ndarray,
    free_t: np.ndarray,
    map_idx: np.ndarray,
    diff_idx: np.ndarray,
    map_names: List[str],
    diff_names: List[str],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    occ_pred = occ_prob >= thresholds["occupancy"]
    wall_pred = wall_prob >= thresholds["wall"]
    door_pred = door_prob >= thresholds["doorway"]
    free_pred = free_prob >= thresholds["free"]

    occ = binary_stats(occ_pred, occ_t)
    wall = binary_stats(wall_pred, wall_t)
    door = binary_stats(door_pred, door_t)
    free = binary_stats(free_pred, free_t)

    by_map: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(map_names):
        m = map_idx == i
        if not np.any(m):
            continue
        by_map[name] = {
            "occupancy_iou": binary_stats(occ_pred[m], occ_t[m])["iou"],
            "wall_f1": binary_stats(wall_pred[m], wall_t[m])["f1"],
            "doorway_f1": binary_stats(door_pred[m], door_t[m])["f1"],
            "doorway_precision": binary_stats(door_pred[m], door_t[m])["precision"],
            "doorway_recall": binary_stats(door_pred[m], door_t[m])["recall"],
        }

    by_diff: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(diff_names):
        m = diff_idx == i
        if not np.any(m):
            continue
        by_diff[name] = {
            "occupancy_iou": binary_stats(occ_pred[m], occ_t[m])["iou"],
            "wall_f1": binary_stats(wall_pred[m], wall_t[m])["f1"],
            "doorway_f1": binary_stats(door_pred[m], door_t[m])["f1"],
            "doorway_precision": binary_stats(door_pred[m], door_t[m])["precision"],
            "doorway_recall": binary_stats(door_pred[m], door_t[m])["recall"],
        }

    fpr, avg_fpr = non_doorway_fp_rates(door_pred, door_t, map_idx, map_names)
    doorway_map_f1 = by_map.get("doorway", {}).get("doorway_f1", 0.0)
    hard_f1 = by_diff.get("hard_noise", {}).get("doorway_f1", 0.0)

    promotion_score = (
        1.0 * occ["iou"]
        + 1.0 * wall["f1"]
        + 1.4 * door["f1"]
        + 0.7 * doorway_map_f1
        + 0.7 * hard_f1
        - 0.9 * avg_fpr
    )

    return {
        "overall": {
            "occupancy_iou": occ["iou"],
            "wall_f1": wall["f1"],
            "doorway_precision": door["precision"],
            "doorway_recall": door["recall"],
            "doorway_f1": door["f1"],
            "free_space_accuracy": free["accuracy"],
            "promotion_score": promotion_score,
        },
        "by_map": by_map,
        "by_difficulty": by_diff,
        "doorway_false_positive_rate_empty_room": fpr.get("empty_room", 0.0),
        "doorway_false_positive_rate_corridor": fpr.get("corridor", 0.0),
        "doorway_false_positive_rate_single_block": fpr.get("single_block", 0.0),
        "doorway_false_positive_rate_cluttered_room": fpr.get("cluttered_room", 0.0),
        "avg_non_doorway_fp_rate": avg_fpr,
        "doorway_map_f1": doorway_map_f1,
        "hard_noise_doorway_f1": hard_f1,
    }


def acceptance_flags(metrics: Dict[str, object]) -> Dict[str, bool]:
    ov = metrics["overall"]  # type: ignore[index]
    return {
        "doorway_f1_ge_0_50": float(ov["doorway_f1"]) >= 0.50,
        "hard_noise_doorway_f1_ge_0_42": float(metrics.get("hard_noise_doorway_f1", 0.0)) >= 0.42,
        "occupancy_iou_ge_0_68": float(ov["occupancy_iou"]) >= 0.68,
        "wall_f1_ge_0_50": float(ov["wall_f1"]) >= 0.50,
        "empty_room_fp_le_0_15": float(metrics.get("doorway_false_positive_rate_empty_room", 1.0)) <= 0.15,
        "corridor_fp_le_0_05": float(metrics.get("doorway_false_positive_rate_corridor", 1.0)) <= 0.05,
    }


def safe_read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_plots(out_dir: Path, v4: Dict[str, object], v5: Dict[str, object], v6: Dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = ["occ_iou", "wall_f1", "door_f1", "door_map_f1", "hard_door_f1"]
    v4_vals = [
        v4.get("occupancy_iou", 0.0),
        v4.get("wall_f1", 0.0),
        v4.get("doorway_f1", 0.0),
        v4.get("doorway_map_f1", 0.0),
        v4.get("hard_noise_doorway_f1", 0.0),
    ]
    v5_vals = [
        v5.get("occupancy_iou", 0.0),
        v5.get("wall_f1", 0.0),
        v5.get("doorway_f1", 0.0),
        v5.get("doorway_map_f1", 0.0),
        v5.get("hard_noise_doorway_f1", 0.0),
    ]
    v6_vals = [
        v6.get("occupancy_iou", 0.0),
        v6.get("wall_f1", 0.0),
        v6.get("doorway_f1", 0.0),
        v6.get("doorway_map_f1", 0.0),
        v6.get("hard_noise_doorway_f1", 0.0),
    ]

    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(10, 4), dpi=120)
    plt.bar(x - w, v4_vals, width=w, label="v4")
    plt.bar(x, v5_vals, width=w, label="v5")
    plt.bar(x + w, v6_vals, width=w, label="v6")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.title("Phase 2C Mapper Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_metrics_bar.png")
    plt.close()

    fp_labels = ["empty", "corridor", "single", "clutter"]
    v4_fp = [
        v4.get("doorway_false_positive_rate_empty_room", 0.0),
        v4.get("doorway_false_positive_rate_corridor", 0.0),
        v4.get("doorway_false_positive_rate_single_block", 0.0),
        v4.get("doorway_false_positive_rate_cluttered_room", 0.0),
    ]
    v5_fp = [
        v5.get("doorway_false_positive_rate_empty_room", 0.0),
        v5.get("doorway_false_positive_rate_corridor", 0.0),
        v5.get("doorway_false_positive_rate_single_block", 0.0),
        v5.get("doorway_false_positive_rate_cluttered_room", 0.0),
    ]
    v6_fp = [
        v6.get("doorway_false_positive_rate_empty_room", 0.0),
        v6.get("doorway_false_positive_rate_corridor", 0.0),
        v6.get("doorway_false_positive_rate_single_block", 0.0),
        v6.get("doorway_false_positive_rate_cluttered_room", 0.0),
    ]

    x = np.arange(len(fp_labels))
    plt.figure(figsize=(8.5, 4), dpi=120)
    plt.bar(x - w, v4_fp, width=w, label="v4")
    plt.bar(x, v5_fp, width=w, label="v5")
    plt.bar(x + w, v6_fp, width=w, label="v6")
    plt.xticks(x, fp_labels)
    plt.ylim(0.0, 0.8)
    plt.title("Doorway False-Positive Activation by Map")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_doorway_fp_bar.png")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    v4_dir = Path(args.v4_run_dir)
    v5_dir = Path(args.v5_run_dir)

    thresholds = sorted(set(float(x.strip()) for x in args.threshold_candidates.split(",") if x.strip()))
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    thresholds = sorted(set(thresholds))

    v4_cfg = safe_read_json(v4_dir / "config.json")
    v5_cfg = safe_read_json(v5_dir / "config.json")

    val_ds = EchoMappingNPZDataset(data_dir / "val.npz", patch_size=args.patch_size)
    test_ds = EchoMappingNPZDataset(data_dir / "test.npz", patch_size=args.patch_size)
    map_names = val_ds.map_names
    diff_names = val_ds.difficulty_names

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    v4_model = V4Model(
        in_channels=val_ds.in_channels,
        n_bins=val_ds.n_bins,
        patch_size=args.patch_size,
        hidden_dim=int(v4_cfg.get("hidden_dim", 320)),
        meta_dim=8,
        use_visibility_head=bool(v4_cfg.get("use_visibility_head", True)),
        use_pose_head=bool(v4_cfg.get("use_pose_head", True)),
    ).to(device)
    v4_model.load_state_dict(torch.load(v4_dir / "best_model.pt", map_location=device))

    v5_model = V5Model(
        in_channels=val_ds.in_channels,
        n_bins=val_ds.n_bins,
        patch_size=args.patch_size,
        hidden_dim=int(v5_cfg.get("hidden_dim", 320)),
        meta_dim=8,
        use_visibility_head=bool(v5_cfg.get("use_visibility_head", True)),
        use_pose_head=bool(v5_cfg.get("use_pose_head", True)),
        use_doorway_context_head=bool(v5_cfg.get("use_doorway_context_head", True)),
    ).to(device)
    v5_model.load_state_dict(torch.load(v5_dir / "best_model.pt", map_location=device))

    v4_val = collect_model_outputs(
        model=v4_model,
        loader=val_loader,
        device=device,
        patch_size=args.patch_size,
        use_soft_gating=bool(v4_cfg.get("use_soft_doorway_gating", True)),
        gate_strength=float(v4_cfg.get("doorway_gate_strength", 0.5)),
        min_gate=float(v4_cfg.get("doorway_min_gate", 0.35)),
        structure_weight=float(v4_cfg.get("doorway_structure_weight", 0.5)),
        has_context_head=False,
    )
    v5_val = collect_model_outputs(
        model=v5_model,
        loader=val_loader,
        device=device,
        patch_size=args.patch_size,
        use_soft_gating=bool(v5_cfg.get("use_soft_doorway_gating", True)),
        gate_strength=float(v5_cfg.get("doorway_gate_strength", 0.5)),
        min_gate=float(v5_cfg.get("doorway_min_gate", 0.35)),
        structure_weight=float(v5_cfg.get("doorway_structure_weight", 0.5)),
        has_context_head=bool(v5_cfg.get("use_doorway_context_head", True)),
    )

    v4_test = collect_model_outputs(
        model=v4_model,
        loader=test_loader,
        device=device,
        patch_size=args.patch_size,
        use_soft_gating=bool(v4_cfg.get("use_soft_doorway_gating", True)),
        gate_strength=float(v4_cfg.get("doorway_gate_strength", 0.5)),
        min_gate=float(v4_cfg.get("doorway_min_gate", 0.35)),
        structure_weight=float(v4_cfg.get("doorway_structure_weight", 0.5)),
        has_context_head=False,
    )
    v5_test = collect_model_outputs(
        model=v5_model,
        loader=test_loader,
        device=device,
        patch_size=args.patch_size,
        use_soft_gating=bool(v5_cfg.get("use_soft_doorway_gating", True)),
        gate_strength=float(v5_cfg.get("doorway_gate_strength", 0.5)),
        min_gate=float(v5_cfg.get("doorway_min_gate", 0.35)),
        structure_weight=float(v5_cfg.get("doorway_structure_weight", 0.5)),
        has_context_head=bool(v5_cfg.get("use_doorway_context_head", True)),
    )

    occ_sources_val = {
        "v4_occ": v4_val["occ_prob"],
        "v5_occ": v5_val["occ_prob"],
    }
    wall_sources_val = {
        "v4_wall": v4_val["wall_prob"],
        "v5_wall": v5_val["wall_prob"],
    }
    free_sources_val = {
        "v4_free": v4_val["free_prob"],
        "v5_free": v5_val["free_prob"],
    }

    occ_choice = []
    for mode, prob in occ_sources_val.items():
        t, s = scan_threshold(prob, v4_val["occ_t"], thresholds, metric="iou")
        occ_choice.append((mode, t, s))
    occ_mode, occ_thr, occ_stats = max(occ_choice, key=lambda x: x[2]["iou"])

    wall_choice = []
    for mode, prob in wall_sources_val.items():
        t, s = scan_threshold(prob, v4_val["wall_t"], thresholds, metric="f1")
        wall_choice.append((mode, t, s))
    wall_mode, wall_thr, wall_stats = max(wall_choice, key=lambda x: x[2]["f1"])

    free_choice = []
    for mode, prob in free_sources_val.items():
        t, s = scan_threshold(prob, v4_val["free_t"], thresholds, metric="accuracy")
        free_choice.append((mode, t, s))
    free_mode, free_thr, free_stats = max(free_choice, key=lambda x: x[2]["accuracy"])

    door_candidates_val = {
        "v4_raw": v4_val["door_prob_raw"],
        "v4_soft": v4_val["door_prob_soft"],
        "v5_raw": v5_val["door_prob_raw"],
        "v5_soft": v5_val["door_prob_soft"],
        "v5_context": v5_val["door_prob_context"],
        "v5_final": v5_val["door_prob_final"],
    }
    door_mode, door_thr, door_row, door_table = select_best_doorway_mode(
        candidates=door_candidates_val,
        door_t=v4_val["door_t"],
        map_idx=v4_val["map_idx"],
        diff_idx=v4_val["diff_idx"],
        map_names=map_names,
        diff_names=diff_names,
        thresholds=thresholds,
        fp_lambda=args.doorway_fp_lambda,
        recall_floor=args.doorway_recall_floor,
    )

    selected = {
        "occupancy_mode": occ_mode,
        "occupancy_threshold": float(occ_thr),
        "wall_mode": wall_mode,
        "wall_threshold": float(wall_thr),
        "doorway_mode": door_mode,
        "doorway_threshold": float(door_thr),
        "free_mode": free_mode,
        "free_threshold": float(free_thr),
    }

    def choose_from_test(mode: str) -> np.ndarray:
        if mode.startswith("v4_"):
            src = v4_test
            key = mode.split("_", 1)[1]
            return src[f"{key}_prob"] if key in ["occ", "wall", "free"] else src[f"door_prob_{key}"]
        src = v5_test
        key = mode.split("_", 1)[1]
        return src[f"{key}_prob"] if key in ["occ", "wall", "free"] else src[f"door_prob_{key}"]

    occ_prob_test = choose_from_test(occ_mode)
    wall_prob_test = choose_from_test(wall_mode)
    door_prob_test = choose_from_test(door_mode)
    free_prob_test = choose_from_test(free_mode)

    thresholds_sel = {
        "occupancy": float(occ_thr),
        "wall": float(wall_thr),
        "doorway": float(door_thr),
        "free": float(free_thr),
    }

    v6_test_metrics = evaluate_combined(
        occ_prob=occ_prob_test,
        wall_prob=wall_prob_test,
        door_prob=door_prob_test,
        free_prob=free_prob_test,
        occ_t=v4_test["occ_t"],
        wall_t=v4_test["wall_t"],
        door_t=v4_test["door_t"],
        free_t=v4_test["free_t"],
        map_idx=v4_test["map_idx"],
        diff_idx=v4_test["diff_idx"],
        map_names=map_names,
        diff_names=diff_names,
        thresholds=thresholds_sel,
    )
    flags = acceptance_flags(v6_test_metrics)
    v6_test_metrics["acceptance_flags"] = flags
    v6_test_metrics["selected_modes_and_thresholds"] = selected
    v6_test_metrics["doorway_validation_selection_row"] = door_row

    v4_json = safe_read_json(v4_dir / "test_metrics.json")
    v5_json = safe_read_json(v5_dir / "test_metrics.json")
    v4_ref = v4_json.get("test_metrics_at_best_thresholds", {})
    v5_ref = v5_json.get("test_metrics_at_selected_mode", {})

    def flatten_for_compare(block: Dict[str, object]) -> Dict[str, float]:
        ov = block.get("overall", {})
        by_map = block.get("by_map", {})
        by_diff = block.get("by_difficulty", {})
        out = {
            "occupancy_iou": float(ov.get("occupancy_iou", 0.0)),
            "wall_f1": float(ov.get("wall_f1", 0.0)),
            "doorway_f1": float(ov.get("doorway_f1", 0.0)),
            "doorway_precision": float(ov.get("doorway_precision", 0.0)),
            "doorway_recall": float(ov.get("doorway_recall", 0.0)),
            "doorway_map_f1": float(by_map.get("doorway", {}).get("doorway_f1", 0.0)),
            "hard_noise_doorway_f1": float(by_diff.get("hard_noise", {}).get("doorway_f1", 0.0)),
            "doorway_false_positive_rate_empty_room": float(block.get("doorway_false_positive_rate_empty_room", 0.0)),
            "doorway_false_positive_rate_corridor": float(block.get("doorway_false_positive_rate_corridor", 0.0)),
            "doorway_false_positive_rate_single_block": float(block.get("doorway_false_positive_rate_single_block", 0.0)),
            "doorway_false_positive_rate_cluttered_room": float(block.get("doorway_false_positive_rate_cluttered_room", 0.0)),
        }
        out["promotion_score"] = (
            out["occupancy_iou"]
            + out["wall_f1"]
            + 1.4 * out["doorway_f1"]
            + 0.7 * out["doorway_map_f1"]
            + 0.7 * out["hard_noise_doorway_f1"]
            - 0.9
            * np.mean(
                [
                    out["doorway_false_positive_rate_empty_room"],
                    out["doorway_false_positive_rate_corridor"],
                    out["doorway_false_positive_rate_single_block"],
                    out["doorway_false_positive_rate_cluttered_room"],
                ]
            )
        )
        return out

    v4_comp = flatten_for_compare(v4_ref if isinstance(v4_ref, dict) else {})
    v5_comp = flatten_for_compare(v5_ref if isinstance(v5_ref, dict) else {})
    v6_comp = flatten_for_compare(v6_test_metrics if isinstance(v6_test_metrics, dict) else {})

    comparison = {
        "selected_modes_and_thresholds": selected,
        "v4": v4_comp,
        "v5": v5_comp,
        "v6_hybrid_calibrated": v6_comp,
        "acceptance_flags_v6": flags,
        "accepted_v6": bool(all(flags.values())),
    }

    # README recommendation logic.
    if comparison["accepted_v6"]:
        recommendation = (
            "v6 passes all targets and can replace v4/v5 as default. Keep v4 and v5 as fallbacks."
        )
    else:
        recommendation = (
            "v6 should act as hybrid selector: use v4-like structure heads and v5-like doorway suppression mode, "
            "while retaining v4 and v5 checkpoints as task-specific fallbacks."
        )
    comparison["recommendation"] = recommendation

    (out_dir / "selected_modes_thresholds.json").write_text(
        json.dumps(
            {
                "selected": selected,
                "occupancy_validation_stats": occ_stats,
                "wall_validation_stats": wall_stats,
                "free_validation_stats": free_stats,
                "doorway_validation_best": door_row,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "doorway_mode_threshold_table_val.json").write_text(json.dumps(door_table, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(v6_test_metrics, indent=2), encoding="utf-8")
    (out_dir / "comparison_vs_v4_v5.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    if args.save_plots:
        make_plots(out_dir=out_dir, v4=v4_comp, v5=v5_comp, v6=v6_comp)

    readme = (
        "# Phase 2C.5 Hybrid Calibrated Mapper (v6)\n\n"
        "This run calibrates a hybrid selector over v4/v5 checkpoints.\n\n"
        f"- Selected occupancy mode: `{selected['occupancy_mode']}` @ {selected['occupancy_threshold']:.2f}\n"
        f"- Selected wall mode: `{selected['wall_mode']}` @ {selected['wall_threshold']:.2f}\n"
        f"- Selected doorway mode: `{selected['doorway_mode']}` @ {selected['doorway_threshold']:.2f}\n"
        f"- Selected free mode: `{selected['free_mode']}` @ {selected['free_threshold']:.2f}\n\n"
        "Acceptance targets:\n"
        f"- doorway_f1 >= 0.50: {flags['doorway_f1_ge_0_50']}\n"
        f"- hard_noise_doorway_f1 >= 0.42: {flags['hard_noise_doorway_f1_ge_0_42']}\n"
        f"- occupancy_iou >= 0.68: {flags['occupancy_iou_ge_0_68']}\n"
        f"- wall_f1 >= 0.50: {flags['wall_f1_ge_0_50']}\n"
        f"- empty_room_fp <= 0.15: {flags['empty_room_fp_le_0_15']}\n"
        f"- corridor_fp <= 0.05: {flags['corridor_fp_le_0_05']}\n\n"
        f"Recommendation: {recommendation}\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("Phase 2C.5 hybrid calibration complete.")
    print(f"Output dir: {out_dir}")
    print(f"Selected modes: {selected}")
    print(
        "Test v6 metrics: "
        f"occ_iou={v6_comp['occupancy_iou']:.4f}, wall_f1={v6_comp['wall_f1']:.4f}, "
        f"door_p/r/f1={v6_comp['doorway_precision']:.4f}/{v6_comp['doorway_recall']:.4f}/{v6_comp['doorway_f1']:.4f}"
    )
    print(
        "Doorway FP empty/corr/single/clutter: "
        f"{v6_comp['doorway_false_positive_rate_empty_room']:.4f}/"
        f"{v6_comp['doorway_false_positive_rate_corridor']:.4f}/"
        f"{v6_comp['doorway_false_positive_rate_single_block']:.4f}/"
        f"{v6_comp['doorway_false_positive_rate_cluttered_room']:.4f}"
    )
    print(f"Accepted v6: {comparison['accepted_v6']}")


if __name__ == "__main__":
    main()
