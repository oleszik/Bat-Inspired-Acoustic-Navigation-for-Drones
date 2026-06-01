from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .echo_dataset import EchoMappingNPZDataset
    from .echo_model import AcousticEchoMapper
    from .train_echo_mapper import evaluate_with_breakdown, save_prediction_plots
except ImportError:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from simulation.phase2_mapping.echo_dataset import EchoMappingNPZDataset  # type: ignore
    from simulation.phase2_mapping.echo_model import AcousticEchoMapper  # type: ignore
    from simulation.phase2_mapping.train_echo_mapper import evaluate_with_breakdown, save_prediction_plots  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained Phase 2C echo mapper.")
    p.add_argument("--data-dir", type=str, default="datasets/phase2_echo_mapping")
    p.add_argument("--model-path", type=str, default="runs/phase2_echo_mapper/best_model.pt")
    p.add_argument("--output-dir", type=str, default="runs/phase2_echo_mapper")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ds = EchoMappingNPZDataset(Path(args.data_dir) / "test.npz", patch_size=args.patch_size)
    info = test_ds.get_info()

    model = AcousticEchoMapper(
        in_channels=info.in_channels,
        n_bins=info.n_bins,
        patch_size=info.patch_size,
        hidden_dim=args.hidden_dim,
        meta_dim=8,
        use_visibility_head=True,
        use_pose_head=True,
    )

    device = resolve_device(args.device)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    metrics = evaluate_with_breakdown(
        model=model,
        loader=loader,
        device=device,
        map_names=info.map_names,
        difficulty_names=info.difficulty_names,
        use_pose_head=True,
    )

    (out_dir / "test_metrics_recomputed.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.save_plots:
        save_prediction_plots(
            model=model,
            loader=loader,
            device=device,
            patch_size=args.patch_size,
            out_dir=out_dir / "prediction_samples_recomputed",
            max_plots=24,
        )

    print("Evaluation complete")
    print(
        f"occ_iou={metrics['overall']['occupancy_iou']:.4f}, "
        f"wall_f1={metrics['overall']['wall_f1']:.4f}, "
        f"doorway_f1={metrics['overall']['doorway_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
