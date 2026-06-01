from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from .echo_dataset import (
        EchoMappingNPZDataset,
        build_weighted_sampling_weights,
        compute_pos_weight_from_ratio,
    )
    from .echo_model import AcousticEchoMapper
except ImportError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.phase2_mapping.echo_dataset import (  # type: ignore
        EchoMappingNPZDataset,
        build_weighted_sampling_weights,
        compute_pos_weight_from_ratio,
    )
    from simulation.phase2_mapping.echo_model import AcousticEchoMapper  # type: ignore


@dataclass
class LossWeights:
    occupancy: float = 1.0
    wall: float = 1.0
    doorway: float = 1.5
    free: float = 0.8
    visibility: float = 0.25
    pose: float = 0.1


class MetricAccumulator:
    def __init__(self) -> None:
        self.occ_correct = 0.0
        self.occ_total = 0.0
        self.occ_inter = 0.0
        self.occ_union = 0.0

        self.free_correct = 0.0
        self.free_total = 0.0

        self.wall_tp = 0.0
        self.wall_fp = 0.0
        self.wall_fn = 0.0

        self.door_tp = 0.0
        self.door_fp = 0.0
        self.door_fn = 0.0

        self.wall_correct = 0.0
        self.wall_total = 0.0
        self.door_correct = 0.0
        self.door_total = 0.0

        self.pose_sse = 0.0
        self.pose_count = 0.0

    @staticmethod
    def _prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
        p = tp / max(1.0, tp + fp)
        r = tp / max(1.0, tp + fn)
        f1 = (2.0 * p * r) / max(1e-8, p + r)
        return p, r, f1

    def update(
        self,
        occ_logits: torch.Tensor,
        wall_logits: torch.Tensor,
        door_logits: torch.Tensor,
        free_logits: torch.Tensor,
        occ_t: torch.Tensor,
        wall_t: torch.Tensor,
        door_t: torch.Tensor,
        free_t: torch.Tensor,
        pose_pred: torch.Tensor | None = None,
        pose_t: torch.Tensor | None = None,
    ) -> None:
        occ_p = (torch.sigmoid(occ_logits) >= 0.5)
        wall_p = (torch.sigmoid(wall_logits) >= 0.5)
        door_p = (torch.sigmoid(door_logits) >= 0.5)
        free_p = (torch.sigmoid(free_logits) >= 0.5)

        occ_b = occ_t >= 0.5
        wall_b = wall_t >= 0.5
        door_b = door_t >= 0.5
        free_b = free_t >= 0.5

        self.occ_correct += float((occ_p == occ_b).sum().item())
        self.occ_total += float(occ_b.numel())
        self.occ_inter += float((occ_p & occ_b).sum().item())
        self.occ_union += float((occ_p | occ_b).sum().item())

        self.free_correct += float((free_p == free_b).sum().item())
        self.free_total += float(free_b.numel())

        self.wall_correct += float((wall_p == wall_b).sum().item())
        self.wall_total += float(wall_b.numel())
        self.door_correct += float((door_p == door_b).sum().item())
        self.door_total += float(door_b.numel())

        self.wall_tp += float((wall_p & wall_b).sum().item())
        self.wall_fp += float((wall_p & ~wall_b).sum().item())
        self.wall_fn += float((~wall_p & wall_b).sum().item())

        self.door_tp += float((door_p & door_b).sum().item())
        self.door_fp += float((door_p & ~door_b).sum().item())
        self.door_fn += float((~door_p & door_b).sum().item())

        if pose_pred is not None and pose_t is not None:
            err = pose_pred - pose_t
            self.pose_sse += float((err * err).sum().item())
            self.pose_count += float(err.numel())

    def to_dict(self) -> Dict[str, float]:
        wall_p, wall_r, wall_f1 = self._prf(self.wall_tp, self.wall_fp, self.wall_fn)
        door_p, door_r, door_f1 = self._prf(self.door_tp, self.door_fp, self.door_fn)

        occ_acc = self.occ_correct / max(1.0, self.occ_total)
        occ_iou = self.occ_inter / max(1.0, self.occ_union)
        free_acc = self.free_correct / max(1.0, self.free_total)
        wall_acc = self.wall_correct / max(1.0, self.wall_total)
        door_acc = self.door_correct / max(1.0, self.door_total)
        mean_patch_acc = float(np.mean([occ_acc, wall_acc, door_acc, free_acc]))

        out = {
            "occupancy_accuracy": float(occ_acc),
            "occupancy_iou": float(occ_iou),
            "wall_precision": float(wall_p),
            "wall_recall": float(wall_r),
            "wall_f1": float(wall_f1),
            "doorway_precision": float(door_p),
            "doorway_recall": float(door_r),
            "doorway_f1": float(door_f1),
            "free_space_accuracy": float(free_acc),
            "mean_patch_accuracy": float(mean_patch_acc),
        }

        if self.pose_count > 0:
            out["pose_correction_rmse"] = float(math.sqrt(self.pose_sse / self.pose_count))
        else:
            out["pose_correction_rmse"] = 0.0
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2C - train neural echo mapper.")
    parser.add_argument("--data-dir", type=str, default="datasets/phase2_echo_mapping")
    parser.add_argument("--output-dir", type=str, default="runs/phase2_echo_mapper")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-plots", action="store_true")

    parser.add_argument("--use-pose-head", action="store_true", default=True)
    parser.add_argument("--use-visibility-head", action="store_true", default=True)
    parser.add_argument("--doorway-focal-gamma", type=float, default=1.5)
    parser.add_argument("--disable-focal-doorway", action="store_true")
    parser.add_argument("--oversample-doorway", action="store_true", default=True)
    parser.add_argument("--doorway-oversample-boost", type=float, default=4.0)
    parser.add_argument("--wall-oversample-boost", type=float, default=1.5)
    parser.add_argument("--prediction-plot-count", type=int, default=24)
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


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 1.5,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = torch.pow(1.0 - pt, gamma)
    return (focal * bce).mean()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: LossWeights,
    wall_pos_weight: torch.Tensor,
    door_pos_weight: torch.Tensor,
    use_focal_doorway: bool,
    doorway_focal_gamma: float,
    use_visibility_head: bool,
    use_pose_head: bool,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}

    losses["occupancy"] = F.binary_cross_entropy_with_logits(outputs["occupancy_logits"], batch["occupancy"])
    losses["wall"] = F.binary_cross_entropy_with_logits(
        outputs["wall_logits"],
        batch["wall"],
        pos_weight=wall_pos_weight,
    )

    if use_focal_doorway:
        losses["doorway"] = focal_bce_with_logits(
            outputs["doorway_logits"],
            batch["doorway"],
            pos_weight=door_pos_weight,
            gamma=doorway_focal_gamma,
        )
    else:
        losses["doorway"] = F.binary_cross_entropy_with_logits(
            outputs["doorway_logits"],
            batch["doorway"],
            pos_weight=door_pos_weight,
        )

    losses["free"] = F.binary_cross_entropy_with_logits(outputs["free_logits"], batch["free"])

    if use_visibility_head and "visibility_logits" in outputs:
        losses["visibility"] = F.binary_cross_entropy_with_logits(outputs["visibility_logits"], batch["visibility"])
    else:
        losses["visibility"] = torch.zeros((), device=outputs["occupancy_logits"].device)

    if use_pose_head and "pose_pred" in outputs:
        losses["pose"] = F.mse_loss(outputs["pose_pred"], batch["pose_correction"])
    else:
        losses["pose"] = torch.zeros((), device=outputs["occupancy_logits"].device)

    total = (
        loss_weights.occupancy * losses["occupancy"]
        + loss_weights.wall * losses["wall"]
        + loss_weights.doorway * losses["doorway"]
        + loss_weights.free * losses["free"]
        + loss_weights.visibility * losses["visibility"]
        + loss_weights.pose * losses["pose"]
    )
    losses["total"] = total
    return losses


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def run_epoch(
    model: AcousticEchoMapper,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: LossWeights,
    wall_pos_weight: torch.Tensor,
    door_pos_weight: torch.Tensor,
    use_focal_doorway: bool,
    doorway_focal_gamma: float,
    use_visibility_head: bool,
    use_pose_head: bool,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(is_train)

    meter = MetricAccumulator()
    loss_sums = {
        "total": 0.0,
        "occupancy": 0.0,
        "wall": 0.0,
        "doorway": 0.0,
        "free": 0.0,
        "visibility": 0.0,
        "pose": 0.0,
    }
    nb = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["signal"], batch["meta"])
        losses = compute_losses(
            outputs=outputs,
            batch=batch,
            loss_weights=loss_weights,
            wall_pos_weight=wall_pos_weight,
            door_pos_weight=door_pos_weight,
            use_focal_doorway=use_focal_doorway,
            doorway_focal_gamma=doorway_focal_gamma,
            use_visibility_head=use_visibility_head,
            use_pose_head=use_pose_head,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()

        meter.update(
            occ_logits=outputs["occupancy_logits"],
            wall_logits=outputs["wall_logits"],
            door_logits=outputs["doorway_logits"],
            free_logits=outputs["free_logits"],
            occ_t=batch["occupancy"],
            wall_t=batch["wall"],
            door_t=batch["doorway"],
            free_t=batch["free"],
            pose_pred=outputs.get("pose_pred", None),
            pose_t=batch["pose_correction"] if use_pose_head else None,
        )

        for k in loss_sums:
            loss_sums[k] += float(losses[k].item())
        nb += 1

    metrics = meter.to_dict()
    for k, v in loss_sums.items():
        metrics[f"{k}_loss"] = float(v / max(1, nb))
    return metrics


def evaluate_with_breakdown(
    model: AcousticEchoMapper,
    loader: DataLoader,
    device: torch.device,
    map_names: list[str],
    difficulty_names: list[str],
    use_pose_head: bool,
) -> Dict[str, object]:
    model.eval()
    overall = MetricAccumulator()
    by_map = {name: MetricAccumulator() for name in map_names}
    by_diff = {name: MetricAccumulator() for name in difficulty_names}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["signal"], batch["meta"])

            overall.update(
                occ_logits=outputs["occupancy_logits"],
                wall_logits=outputs["wall_logits"],
                door_logits=outputs["doorway_logits"],
                free_logits=outputs["free_logits"],
                occ_t=batch["occupancy"],
                wall_t=batch["wall"],
                door_t=batch["doorway"],
                free_t=batch["free"],
                pose_pred=outputs.get("pose_pred", None),
                pose_t=batch["pose_correction"] if use_pose_head else None,
            )

            for i in range(batch["map_index"].shape[0]):
                map_name = map_names[int(batch["map_index"][i].item())]
                diff_name = difficulty_names[int(batch["difficulty_index"][i].item())]
                by_map[map_name].update(
                    occ_logits=outputs["occupancy_logits"][i : i + 1],
                    wall_logits=outputs["wall_logits"][i : i + 1],
                    door_logits=outputs["doorway_logits"][i : i + 1],
                    free_logits=outputs["free_logits"][i : i + 1],
                    occ_t=batch["occupancy"][i : i + 1],
                    wall_t=batch["wall"][i : i + 1],
                    door_t=batch["doorway"][i : i + 1],
                    free_t=batch["free"][i : i + 1],
                    pose_pred=outputs.get("pose_pred", None)[i : i + 1] if "pose_pred" in outputs else None,
                    pose_t=batch["pose_correction"][i : i + 1] if use_pose_head else None,
                )
                by_diff[diff_name].update(
                    occ_logits=outputs["occupancy_logits"][i : i + 1],
                    wall_logits=outputs["wall_logits"][i : i + 1],
                    door_logits=outputs["doorway_logits"][i : i + 1],
                    free_logits=outputs["free_logits"][i : i + 1],
                    occ_t=batch["occupancy"][i : i + 1],
                    wall_t=batch["wall"][i : i + 1],
                    door_t=batch["doorway"][i : i + 1],
                    free_t=batch["free"][i : i + 1],
                    pose_pred=outputs.get("pose_pred", None)[i : i + 1] if "pose_pred" in outputs else None,
                    pose_t=batch["pose_correction"][i : i + 1] if use_pose_head else None,
                )

    out = {
        "overall": overall.to_dict(),
        "by_map": {k: v.to_dict() for k, v in by_map.items()},
        "by_difficulty": {k: v.to_dict() for k, v in by_diff.items()},
    }

    # Explicit requested slices
    if "corridor" in out["by_map"]:
        out["corridor_performance"] = out["by_map"]["corridor"]
    if "doorway" in out["by_map"]:
        out["doorway_map_performance"] = out["by_map"]["doorway"]
    if "hard_noise" in out["by_difficulty"]:
        out["hard_noise_performance"] = out["by_difficulty"]["hard_noise"]

    return out


def save_prediction_plots(
    model: AcousticEchoMapper,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    out_dir: Path,
    max_plots: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["signal"], batch["meta"])

            occ_p = torch.sigmoid(outputs["occupancy_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)
            wall_p = torch.sigmoid(outputs["wall_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)
            door_p = torch.sigmoid(outputs["doorway_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)
            free_p = torch.sigmoid(outputs["free_logits"]).cpu().numpy().reshape(-1, patch_size, patch_size)

            occ_t = batch["occupancy"].cpu().numpy().reshape(-1, patch_size, patch_size)
            wall_t = batch["wall"].cpu().numpy().reshape(-1, patch_size, patch_size)
            door_t = batch["doorway"].cpu().numpy().reshape(-1, patch_size, patch_size)
            free_t = batch["free"].cpu().numpy().reshape(-1, patch_size, patch_size)
            timing = batch["echo_timing"].cpu().numpy()
            intensity = batch["echo_intensity"].cpu().numpy()

            bsz = occ_p.shape[0]
            for i in range(bsz):
                if plotted >= max_plots:
                    return

                fig, axes = plt.subplots(3, 3, figsize=(10, 9), dpi=120)
                axes = axes.flatten()

                x = np.arange(timing.shape[1])
                axes[0].plot(x, timing[i], marker="o", label="timing(m)")
                axes[0].plot(x, intensity[i], marker="s", label="intensity")
                axes[0].set_title("Input Echo Timing/Intensity")
                axes[0].set_xticks(x)
                axes[0].legend(fontsize=7)

                axes[1].imshow(occ_t[i], cmap="gray_r", vmin=0, vmax=1)
                axes[1].set_title("GT Occupancy")
                axes[2].imshow(occ_p[i], cmap="gray_r", vmin=0, vmax=1)
                axes[2].set_title("Pred Occupancy")

                axes[3].imshow(wall_t[i], cmap="magma", vmin=0, vmax=1)
                axes[3].set_title("GT Wall")
                axes[4].imshow(wall_p[i], cmap="magma", vmin=0, vmax=1)
                axes[4].set_title("Pred Wall")

                axes[5].imshow(door_t[i], cmap="viridis", vmin=0, vmax=1)
                axes[5].set_title("GT Doorway")
                axes[6].imshow(door_p[i], cmap="viridis", vmin=0, vmax=1)
                axes[6].set_title("Pred Doorway")

                axes[7].imshow(free_t[i], cmap="Blues", vmin=0, vmax=1)
                axes[7].set_title("GT Free-Space")
                axes[8].imshow(free_p[i], cmap="Blues", vmin=0, vmax=1)
                axes[8].set_title("Pred Free-Space")

                for ax in axes:
                    ax.set_xticks([])
                    ax.set_yticks([])

                fig.tight_layout()
                fig.savefig(out_dir / f"prediction_sample_{plotted:04d}.png")
                plt.close(fig)
                plotted += 1


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    train_ds = EchoMappingNPZDataset(data_dir / "train.npz", patch_size=args.patch_size)
    val_ds = EchoMappingNPZDataset(data_dir / "val.npz", patch_size=args.patch_size)
    test_ds = EchoMappingNPZDataset(data_dir / "test.npz", patch_size=args.patch_size)

    info = train_ds.get_info()
    if info.patch_size != args.patch_size:
        raise ValueError(f"Patch size mismatch: dataset {info.patch_size}, arg {args.patch_size}")

    if args.oversample_doorway:
        w = build_weighted_sampling_weights(
            train_ds,
            doorway_boost=args.doorway_oversample_boost,
            wall_boost=args.wall_oversample_boost,
        )
        sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

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

    wall_ratio = float(train_ds.wall.mean())
    door_ratio = float(train_ds.doorway.mean())
    wall_pos_w = compute_pos_weight_from_ratio(wall_ratio, min_w=1.0, max_w=100.0)
    door_pos_w = compute_pos_weight_from_ratio(door_ratio, min_w=5.0, max_w=300.0)

    wall_pos_weight = torch.tensor([wall_pos_w], dtype=torch.float32, device=device)
    door_pos_weight = torch.tensor([door_pos_w], dtype=torch.float32, device=device)

    model = AcousticEchoMapper(
        in_channels=info.in_channels,
        n_bins=info.n_bins,
        patch_size=info.patch_size,
        hidden_dim=args.hidden_dim,
        meta_dim=8,
        use_visibility_head=args.use_visibility_head,
        use_pose_head=args.use_pose_head,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_weights = LossWeights()

    config = {
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patch_size": args.patch_size,
        "hidden_dim": args.hidden_dim,
        "device": str(device),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "use_pose_head": args.use_pose_head,
        "use_visibility_head": args.use_visibility_head,
        "doorway_focal_gamma": args.doorway_focal_gamma,
        "use_focal_doorway": (not args.disable_focal_doorway),
        "oversample_doorway": args.oversample_doorway,
        "doorway_oversample_boost": args.doorway_oversample_boost,
        "wall_oversample_boost": args.wall_oversample_boost,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "in_channels": info.in_channels,
        "n_bins": info.n_bins,
        "wall_positive_ratio": wall_ratio,
        "doorway_positive_ratio": door_ratio,
        "wall_pos_weight": wall_pos_w,
        "door_pos_weight": door_pos_w,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val = float("inf")
    best_epoch = -1
    history = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_weights=loss_weights,
            wall_pos_weight=wall_pos_weight,
            door_pos_weight=door_pos_weight,
            use_focal_doorway=(not args.disable_focal_doorway),
            doorway_focal_gamma=args.doorway_focal_gamma,
            use_visibility_head=args.use_visibility_head,
            use_pose_head=args.use_pose_head,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                loss_weights=loss_weights,
                wall_pos_weight=wall_pos_weight,
                door_pos_weight=door_pos_weight,
                use_focal_doorway=(not args.disable_focal_doorway),
                doorway_focal_gamma=args.doorway_focal_gamma,
                use_visibility_head=args.use_visibility_head,
                use_pose_head=args.use_pose_head,
            )

        rec = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_sec": float(time.time() - start),
        }
        history.append(rec)

        print(
            f"epoch={epoch:02d} "
            f"train_total={train_metrics['total_loss']:.4f} "
            f"val_total={val_metrics['total_loss']:.4f} "
            f"val_occ_iou={val_metrics['occupancy_iou']:.4f} "
            f"val_wall_f1={val_metrics['wall_f1']:.4f} "
            f"val_door_f1={val_metrics['doorway_f1']:.4f}"
        )

        if val_metrics["total_loss"] < best_val:
            best_val = float(val_metrics["total_loss"])
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Validation metrics for best checkpoint
    best_model = AcousticEchoMapper(
        in_channels=info.in_channels,
        n_bins=info.n_bins,
        patch_size=info.patch_size,
        hidden_dim=args.hidden_dim,
        meta_dim=8,
        use_visibility_head=args.use_visibility_head,
        use_pose_head=args.use_pose_head,
    ).to(device)
    best_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

    with torch.no_grad():
        val_eval = run_epoch(
            model=best_model,
            loader=val_loader,
            device=device,
            optimizer=None,
            loss_weights=loss_weights,
            wall_pos_weight=wall_pos_weight,
            door_pos_weight=door_pos_weight,
            use_focal_doorway=(not args.disable_focal_doorway),
            doorway_focal_gamma=args.doorway_focal_gamma,
            use_visibility_head=args.use_visibility_head,
            use_pose_head=args.use_pose_head,
        )

    val_payload = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "metrics": val_eval,
    }
    (out_dir / "validation_metrics.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")

    test_metrics = evaluate_with_breakdown(
        model=best_model,
        loader=test_loader,
        device=device,
        map_names=info.map_names,
        difficulty_names=info.difficulty_names,
        use_pose_head=args.use_pose_head,
    )
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_validation_loss"] = best_val
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    if args.save_plots:
        save_prediction_plots(
            model=best_model,
            loader=test_loader,
            device=device,
            patch_size=args.patch_size,
            out_dir=out_dir / "prediction_samples",
            max_plots=args.prediction_plot_count,
        )

    print("Training complete")
    print(f"Best validation loss: {best_val:.6f} at epoch {best_epoch}")
    print(
        "Test summary: "
        f"occ_iou={test_metrics['overall']['occupancy_iou']:.4f}, "
        f"wall_f1={test_metrics['overall']['wall_f1']:.4f}, "
        f"doorway_p/r/f1={test_metrics['overall']['doorway_precision']:.4f}/"
        f"{test_metrics['overall']['doorway_recall']:.4f}/"
        f"{test_metrics['overall']['doorway_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
