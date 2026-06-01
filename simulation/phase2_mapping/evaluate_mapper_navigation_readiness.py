from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
DOORWAY_CROSSING_TOLERANCE = 0.03
DOORWAY_FALLBACK_MODEL = "v5_context"
DOORWAY_FALLBACK_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2C.6 mapper navigation-readiness evaluation (v4/v5/v6).")
    parser.add_argument("--data-dir", type=str, default="datasets/phase2_echo_mapping")
    parser.add_argument("--v4-run-dir", type=str, default="runs/phase2_echo_mapper_v4")
    parser.add_argument("--v5-run-dir", type=str, default="runs/phase2_echo_mapper_v5")
    parser.add_argument("--v6-run-dir", type=str, default="runs/phase2_echo_mapper_v6_hybrid_calibrated")
    parser.add_argument("--output-dir", type=str, default="runs/phase2_mapper_navigation_readiness")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "iou": iou,
    }


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_outputs(
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
    context_prob = []
    occ_t, wall_t, door_t, free_t = [], [], [], []
    map_idx, diff_idx = [], []

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

    arr = {
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
        arr["door_prob_soft"] = apply_doorway_structural_gating(
            door_prob_2d=arr["door_prob_raw"],
            wall_prob_2d=arr["wall_prob"],
            free_prob_2d=arr["free_prob"],
            occ_prob_2d=arr["occ_prob"],
            patch_size=patch_size,
            gate_strength=gate_strength,
            min_gate=min_gate,
            structure_weight=structure_weight,
        )
    else:
        arr["door_prob_soft"] = arr["door_prob_raw"].copy()
    arr["door_prob_context"] = arr["door_prob_raw"] * arr["context_prob"][:, None]
    arr["door_prob_final"] = arr["door_prob_soft"] * arr["context_prob"][:, None]
    return arr


def make_binary_predictions(
    arr: Dict[str, np.ndarray],
    occ_thr: float,
    wall_thr: float,
    door_thr: float,
    free_thr: float,
    door_mode: str,
) -> Dict[str, np.ndarray]:
    door_prob = arr[f"door_prob_{door_mode}"] if door_mode in {"raw", "soft", "context", "final"} else arr["door_prob_raw"]
    return {
        "occ": arr["occ_prob"] >= occ_thr,
        "wall": arr["wall_prob"] >= wall_thr,
        "door": door_prob >= door_thr,
        "free": arr["free_prob"] >= free_thr,
    }


def frontal_mask(p: int) -> np.ndarray:
    m = np.zeros((p, p), dtype=bool)
    m[: p // 2, p // 4 : (3 * p) // 4] = True
    return m


def center_mask(p: int) -> np.ndarray:
    m = np.zeros((p, p), dtype=bool)
    c = p // 2
    r = max(2, p // 6)
    m[c - r : c + r + 1, c - r : c + r + 1] = True
    return m


def boundary_mask(batch_mask: np.ndarray) -> np.ndarray:
    up = _shift_no_wrap(batch_mask.astype(np.uint8), -1, 0) > 0
    dn = _shift_no_wrap(batch_mask.astype(np.uint8), 1, 0) > 0
    lf = _shift_no_wrap(batch_mask.astype(np.uint8), 0, -1) > 0
    rg = _shift_no_wrap(batch_mask.astype(np.uint8), 0, 1) > 0
    eroded = up & dn & lf & rg
    return batch_mask & (~eroded)


def evaluate_navigation_readiness(
    pred: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    map_idx: np.ndarray,
    diff_idx: np.ndarray,
    map_names: List[str],
    diff_names: List[str],
) -> Dict[str, object]:
    patch_area = pred["occ"].shape[1]
    p = int(np.sqrt(patch_area))
    fmask = frontal_mask(p).reshape(-1)
    cmask = center_mask(p).reshape(-1)

    non_door_maps = [m for m in map_names if m != "doorway"]
    non_door_ids = {map_names.index(m) for m in non_door_maps}
    mask_non_door = np.isin(map_idx, list(non_door_ids))
    mask_non_door_gt0 = mask_non_door & (gt["door"].sum(axis=1) == 0)
    fake_doorway = pred["door"][mask_non_door_gt0][:, fmask].any(axis=1)
    fake_doorway_approach_rate = float(fake_doorway.mean()) if fake_doorway.size else 0.0

    if "doorway" in map_names:
        didx = map_names.index("doorway")
        dm = map_idx == didx
        door_present = gt["door"][dm].sum(axis=1) > 0
        dm2 = np.where(dm)[0][door_present]
    else:
        dm2 = np.array([], dtype=np.int64)
    successes = []
    for idx in dm2:
        g = gt["door"][idx]
        pr = pred["door"][idx]
        inter = np.logical_and(g, pr).sum()
        union = np.logical_or(g, pr).sum()
        iou = float(inter / max(1, union))
        free_ok = float(pred["free"][idx][g].mean()) if g.any() else 0.0
        successes.append(iou >= 0.10 and free_ok >= 0.50)
    doorway_crossing_success_rate = float(np.mean(successes)) if successes else 0.0

    free_near = pred["free"][:, cmask]
    occ_gt_near = gt["occ"][:, cmask]
    risky = np.logical_and(free_near, occ_gt_near).sum(axis=1)
    denom = np.maximum(1, free_near.sum(axis=1))
    collision_risk_near_predicted_free_space = float(np.mean(risky / denom))

    if "corridor" in map_names:
        cidx = map_names.index("corridor")
        cm = map_idx == cidx
        gfree = gt["free"][cm][:, fmask]
        pfree = pred["free"][cm][:, fmask]
        pocc = pred["occ"][cm][:, fmask]
        free_recall = np.logical_and(pfree, gfree).sum(axis=1) / np.maximum(1, gfree.sum(axis=1))
        occ_fp = np.logical_and(pocc, gfree).sum(axis=1) / np.maximum(1, gfree.sum(axis=1))
        corridor_progress_score = float(np.mean(0.7 * free_recall + 0.3 * (1.0 - occ_fp)))
    else:
        corridor_progress_score = 0.0

    occ_iou = binary_stats(pred["occ"], gt["occ"])["iou"]
    free_iou = binary_stats(pred["free"], gt["free"])["iou"]
    map_coverage_score = float(0.5 * occ_iou + 0.5 * free_iou)

    pb = boundary_mask(pred["occ"].reshape(-1, p, p)).reshape(pred["occ"].shape[0], -1)
    gb = boundary_mask(gt["occ"].reshape(-1, p, p)).reshape(gt["occ"].shape[0], -1)
    obstacle_boundary_consistency = float(binary_stats(pb, gb)["f1"])

    door_all = binary_stats(pred["door"], gt["door"])
    doorway_map_f1 = 0.0
    if "doorway" in map_names:
        didx = map_names.index("doorway")
        m = map_idx == didx
        if np.any(m):
            doorway_map_f1 = float(binary_stats(pred["door"][m], gt["door"][m])["f1"])

    hard_noise_doorway_f1 = 0.0
    if "hard_noise" in diff_names:
        hidx = diff_names.index("hard_noise")
        m = diff_idx == hidx
        if np.any(m):
            hard_noise_doorway_f1 = float(binary_stats(pred["door"][m], gt["door"][m])["f1"])

    fpr_by_map = {}
    for mname in NON_DOORWAY_MAPS:
        if mname not in map_names:
            continue
        midx = map_names.index(mname)
        m = map_idx == midx
        neg = m & (gt["door"].sum(axis=1) == 0)
        if np.any(neg):
            fpr_by_map[mname] = float(pred["door"][neg].any(axis=1).mean())
        else:
            fpr_by_map[mname] = 0.0
    avg_fp = float(np.mean([fpr_by_map.get(n, 0.0) for n in NON_DOORWAY_MAPS]))

    navigation_readiness_score = float(
        100.0
        * (
            0.24 * map_coverage_score
            + 0.22 * obstacle_boundary_consistency
            + 0.20 * corridor_progress_score
            + 0.20 * doorway_crossing_success_rate
            + 0.14 * (1.0 - collision_risk_near_predicted_free_space)
        )
        - 25.0 * fake_doorway_approach_rate
        - 15.0 * avg_fp
    )

    return {
        "fake_doorway_approach_rate": fake_doorway_approach_rate,
        "doorway_crossing_success_rate": doorway_crossing_success_rate,
        "collision_risk_near_predicted_free_space": collision_risk_near_predicted_free_space,
        "corridor_progress_score": corridor_progress_score,
        "map_coverage_score": map_coverage_score,
        "obstacle_boundary_consistency": obstacle_boundary_consistency,
        "navigation_readiness_score": navigation_readiness_score,
        "doorway_precision": float(door_all["precision"]),
        "doorway_recall": float(door_all["recall"]),
        "doorway_f1": float(door_all["f1"]),
        "occupancy_iou": float(occ_iou),
        "wall_f1": float(binary_stats(pred["wall"], gt["wall"])["f1"]),
        "doorway_map_f1": doorway_map_f1,
        "hard_noise_doorway_f1": hard_noise_doorway_f1,
        "doorway_false_positive_rate_empty_room": fpr_by_map.get("empty_room", 0.0),
        "doorway_false_positive_rate_corridor": fpr_by_map.get("corridor", 0.0),
        "doorway_false_positive_rate_single_block": fpr_by_map.get("single_block", 0.0),
        "doorway_false_positive_rate_cluttered_room": fpr_by_map.get("cluttered_room", 0.0),
    }


def save_plots(out_dir: Path, comparison: Dict[str, Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    names = ["v4", "v5", "v6"]
    nav_scores = [comparison[n]["navigation_readiness_score"] for n in names]
    fake_rates = [comparison[n]["fake_doorway_approach_rate"] for n in names]
    cross_rates = [comparison[n]["doorway_crossing_success_rate"] for n in names]

    plt.figure(figsize=(7, 4), dpi=120)
    plt.bar(names, nav_scores, color=["#777", "#4d79ff", "#18a558"])
    plt.ylabel("Navigation Readiness Score")
    plt.title("Mapper Navigation Readiness (v4/v5/v6)")
    plt.tight_layout()
    plt.savefig(out_dir / "navigation_readiness_bar.png")
    plt.close()

    plt.figure(figsize=(7, 4), dpi=120)
    plt.bar(names, fake_rates, color=["#777", "#4d79ff", "#18a558"])
    plt.ylabel("Rate")
    plt.title("Fake Doorway Approach Rate (Lower is Better)")
    plt.tight_layout()
    plt.savefig(out_dir / "fake_doorway_rate_bar.png")
    plt.close()

    plt.figure(figsize=(7, 4), dpi=120)
    plt.bar(names, cross_rates, color=["#777", "#4d79ff", "#18a558"])
    plt.ylabel("Rate")
    plt.title("Doorway Crossing Success Rate (Higher is Better)")
    plt.tight_layout()
    plt.savefig(out_dir / "doorway_crossing_success_bar.png")
    plt.close()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v4_dir = Path(args.v4_run_dir)
    v5_dir = Path(args.v5_run_dir)
    v6_dir = Path(args.v6_run_dir)

    v4_cfg = load_json(v4_dir / "config.json")
    v5_cfg = load_json(v5_dir / "config.json")
    v4_metrics = load_json(v4_dir / "test_metrics.json")
    v5_metrics = load_json(v5_dir / "test_metrics.json")
    v6_sel = load_json(v6_dir / "selected_modes_thresholds.json")

    ds = EchoMappingNPZDataset(Path(args.data_dir) / "test.npz", patch_size=args.patch_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    map_names = ds.map_names
    diff_names = ds.difficulty_names

    v4_model = V4Model(
        in_channels=ds.in_channels,
        n_bins=ds.n_bins,
        patch_size=args.patch_size,
        hidden_dim=int(v4_cfg.get("hidden_dim", 320)),
        meta_dim=8,
        use_visibility_head=bool(v4_cfg.get("use_visibility_head", True)),
        use_pose_head=bool(v4_cfg.get("use_pose_head", True)),
    ).to(device)
    v4_model.load_state_dict(torch.load(v4_dir / "best_model.pt", map_location=device))

    v5_model = V5Model(
        in_channels=ds.in_channels,
        n_bins=ds.n_bins,
        patch_size=args.patch_size,
        hidden_dim=int(v5_cfg.get("hidden_dim", 320)),
        meta_dim=8,
        use_visibility_head=bool(v5_cfg.get("use_visibility_head", True)),
        use_pose_head=bool(v5_cfg.get("use_pose_head", True)),
        use_doorway_context_head=bool(v5_cfg.get("use_doorway_context_head", True)),
    ).to(device)
    v5_model.load_state_dict(torch.load(v5_dir / "best_model.pt", map_location=device))

    arr_v4 = collect_outputs(
        model=v4_model,
        loader=loader,
        device=device,
        patch_size=args.patch_size,
        use_soft_gating=bool(v4_cfg.get("use_soft_doorway_gating", True)),
        gate_strength=float(v4_cfg.get("doorway_gate_strength", 0.5)),
        min_gate=float(v4_cfg.get("doorway_min_gate", 0.35)),
        structure_weight=float(v4_cfg.get("doorway_structure_weight", 0.5)),
        has_context_head=False,
    )
    arr_v5 = collect_outputs(
        model=v5_model,
        loader=loader,
        device=device,
        patch_size=args.patch_size,
        use_soft_gating=bool(v5_cfg.get("use_soft_doorway_gating", True)),
        gate_strength=float(v5_cfg.get("doorway_gate_strength", 0.5)),
        min_gate=float(v5_cfg.get("doorway_min_gate", 0.35)),
        structure_weight=float(v5_cfg.get("doorway_structure_weight", 0.5)),
        has_context_head=bool(v5_cfg.get("use_doorway_context_head", True)),
    )

    gt = {
        "occ": arr_v4["occ_t"],
        "wall": arr_v4["wall_t"],
        "door": arr_v4["door_t"],
        "free": arr_v4["free_t"],
    }
    map_idx = arr_v4["map_idx"]
    diff_idx = arr_v4["diff_idx"]

    v4_thr = v4_metrics["selected_thresholds_from_validation"]
    v5_thr = v5_metrics["selected_thresholds_from_validation"]
    v5_mode = str(v5_metrics.get("doorway_prediction_mode", "final")).replace("_gated", "")
    if v5_mode not in {"raw", "soft", "context", "final"}:
        v5_mode = "final"

    v4_pred = make_binary_predictions(
        arr=arr_v4,
        occ_thr=float(v4_thr["occupancy"]),
        wall_thr=float(v4_thr["wall"]),
        door_thr=float(v4_thr["doorway"]),
        free_thr=float(v4_thr["free"]),
        door_mode="soft",
    )

    v5_pred = make_binary_predictions(
        arr=arr_v5,
        occ_thr=float(v5_thr["occupancy"]),
        wall_thr=float(v5_thr["wall"]),
        door_thr=float(v5_thr["doorway"]),
        free_thr=float(v5_thr["free"]),
        door_mode=v5_mode,
    )

    selected = v6_sel["selected"] if "selected" in v6_sel else v6_sel["selected_modes_and_thresholds"]
    occ_mode = str(selected["occupancy_mode"])
    wall_mode = str(selected["wall_mode"])
    door_mode = str(selected["doorway_mode"])
    free_mode = str(selected["free_mode"])

    def pick_source(mode: str, key_base: str) -> Dict[str, np.ndarray]:
        if mode.startswith("v4_"):
            return arr_v4
        return arr_v5

    v6_occ_src = pick_source(occ_mode, "occ")
    v6_wall_src = pick_source(wall_mode, "wall")
    v6_door_src = pick_source(door_mode, "door")
    v6_free_src = pick_source(free_mode, "free")

    door_suffix = door_mode.split("_", 1)[1]
    if door_suffix == "soft":
        door_key = "door_prob_soft"
    elif door_suffix == "context":
        door_key = "door_prob_context"
    elif door_suffix == "final":
        door_key = "door_prob_final"
    else:
        door_key = "door_prob_raw"

    v6_pred = {
        "occ": v6_occ_src["occ_prob"] >= float(selected["occupancy_threshold"]),
        "wall": v6_wall_src["wall_prob"] >= float(selected["wall_threshold"]),
        "door": v6_door_src[door_key] >= float(selected["doorway_threshold"]),
        "free": v6_free_src["free_prob"] >= float(selected["free_threshold"]),
    }

    m_v4 = evaluate_navigation_readiness(v4_pred, gt, map_idx, diff_idx, map_names, diff_names)
    m_v5 = evaluate_navigation_readiness(v5_pred, gt, map_idx, diff_idx, map_names, diff_names)
    m_v6 = evaluate_navigation_readiness(v6_pred, gt, map_idx, diff_idx, map_names, diff_names)

    doorway_crossing_gap_vs_v5 = float(m_v5["doorway_crossing_success_rate"] - m_v6["doorway_crossing_success_rate"])
    acceptance = {
        "v6_navigation_readiness_higher_than_v4": (
            m_v6["navigation_readiness_score"] > m_v4["navigation_readiness_score"]
        ),
        "v6_navigation_readiness_higher_than_v5": (
            m_v6["navigation_readiness_score"] > m_v5["navigation_readiness_score"]
        ),
        "v6_fake_doorway_approach_rate_lower_than_v4": (
            m_v6["fake_doorway_approach_rate"] < m_v4["fake_doorway_approach_rate"]
        ),
        "v6_collision_risk_not_worse_than_v4": (
            m_v6["collision_risk_near_predicted_free_space"] <= m_v4["collision_risk_near_predicted_free_space"]
        ),
        "v6_doorway_crossing_within_tolerance_of_v5": (
            m_v6["doorway_crossing_success_rate"]
            >= (m_v5["doorway_crossing_success_rate"] - DOORWAY_CROSSING_TOLERANCE)
        ),
    }

    readiness_safety_core_pass = (
        acceptance["v6_navigation_readiness_higher_than_v4"]
        and acceptance["v6_navigation_readiness_higher_than_v5"]
        and acceptance["v6_fake_doorway_approach_rate_lower_than_v4"]
        and acceptance["v6_collision_risk_not_worse_than_v4"]
    )
    doorway_tol_pass = acceptance["v6_doorway_crossing_within_tolerance_of_v5"]

    if readiness_safety_core_pass and doorway_tol_pass:
        v6_navigation_status = "recommended_default"
        v6_recommended_default_for_navigation = True
    elif readiness_safety_core_pass and (not doorway_tol_pass):
        v6_navigation_status = "recommended_with_doorway_fallback"
        v6_recommended_default_for_navigation = False
    else:
        v6_navigation_status = "not_recommended"
        v6_recommended_default_for_navigation = False

    fallback_recommendation = {
        "doorway_fallback_model": DOORWAY_FALLBACK_MODEL,
        "doorway_fallback_threshold": DOORWAY_FALLBACK_THRESHOLD,
    }

    comparison = {
        "v4": m_v4,
        "v5": m_v5,
        "v6": m_v6,
        "selected_modes_v6": selected,
        "maps": map_names,
        "difficulties": diff_names,
        "acceptance": acceptance,
        "doorway_crossing_gap_vs_v5": doorway_crossing_gap_vs_v5,
        "doorway_crossing_tolerance": DOORWAY_CROSSING_TOLERANCE,
        "v6_recommended_default_for_navigation": v6_recommended_default_for_navigation,
        "v6_navigation_status": v6_navigation_status,
        "fallback_recommendation": fallback_recommendation,
    }

    (out_dir / "comparison_navigation_readiness_v4_v5_v6.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    save_plots(out_dir=out_dir, comparison={"v4": m_v4, "v5": m_v5, "v6": m_v6})

    if comparison["v6_navigation_status"] == "recommended_default":
        recommendation = "Keep v6 as default mapper for navigation."
    elif comparison["v6_navigation_status"] == "recommended_with_doorway_fallback":
        recommendation = (
            "Use v6 as primary mapper with doorway-heavy fallback to "
            f"{DOORWAY_FALLBACK_MODEL} @ {DOORWAY_FALLBACK_THRESHOLD:.2f}."
        )
    else:
        recommendation = "Do not use v6 as default; keep v4/v5 primary with selective hybrid use."
    readme = (
        "# Mapper Navigation Readiness (Phase 2C.6)\n\n"
        "This evaluation compares v4, v5, and v6 for downstream planning readiness.\n\n"
        f"- v4 navigation_readiness_score: {m_v4['navigation_readiness_score']:.4f}\n"
        f"- v5 navigation_readiness_score: {m_v5['navigation_readiness_score']:.4f}\n"
        f"- v6 navigation_readiness_score: {m_v6['navigation_readiness_score']:.4f}\n\n"
        "Acceptance checks:\n"
        f"- v6 > v4 readiness: {acceptance['v6_navigation_readiness_higher_than_v4']}\n"
        f"- v6 > v5 readiness: {acceptance['v6_navigation_readiness_higher_than_v5']}\n"
        f"- v6 fake-doorway rate < v4: {acceptance['v6_fake_doorway_approach_rate_lower_than_v4']}\n"
        f"- v6 doorway crossing >= v5 - tolerance({DOORWAY_CROSSING_TOLERANCE:.2f}): "
        f"{acceptance['v6_doorway_crossing_within_tolerance_of_v5']}\n"
        f"- v6 collision risk <= v4: {acceptance['v6_collision_risk_not_worse_than_v4']}\n\n"
        f"- doorway_crossing_gap_vs_v5: {doorway_crossing_gap_vs_v5:.4f}\n"
        f"- doorway_crossing_tolerance: {DOORWAY_CROSSING_TOLERANCE:.4f}\n"
        f"- v6_navigation_status: {v6_navigation_status}\n"
        f"- fallback: {DOORWAY_FALLBACK_MODEL} @ {DOORWAY_FALLBACK_THRESHOLD:.2f}\n\n"
        f"Recommendation: {recommendation}\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("Phase 2C.6 navigation-readiness evaluation complete.")
    print(f"Saved: {out_dir / 'comparison_navigation_readiness_v4_v5_v6.json'}")
    print(
        "Readiness scores "
        f"v4={m_v4['navigation_readiness_score']:.3f}, "
        f"v5={m_v5['navigation_readiness_score']:.3f}, "
        f"v6={m_v6['navigation_readiness_score']:.3f}"
    )
    print(f"v6 recommended default: {comparison['v6_recommended_default_for_navigation']}")
    print(f"v6 navigation status: {comparison['v6_navigation_status']}")


if __name__ == "__main__":
    main()
