from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from .train_echo_mapper import EchoMapperNet
except ImportError:  # pragma: no cover
    from simulation.phase2_mapping.train_echo_mapper import EchoMapperNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2D evaluate baseline vs neural mapper.")
    p.add_argument("--baseline-results", type=str, default="runs/phase2_mapping/phase2_mapping_results.json")
    p.add_argument("--dataset", type=str, default="datasets/phase2_echo_mapping/phase2_echo_mapping_dataset.npz")
    p.add_argument("--checkpoint", type=str, default="runs/phase2_mapping/phase2_echo_mapper_best.pt")
    p.add_argument("--output-dir", type=str, default="runs/phase2_mapping")
    return p.parse_args()


def evaluate_neural(dataset_path: str, checkpoint_path: str) -> dict:
    if torch is None:
        return {"available": False, "reason": "PyTorch unavailable"}
    if not Path(checkpoint_path).exists():
        return {"available": False, "reason": f"Checkpoint not found: {checkpoint_path}"}

    d = np.load(dataset_path)
    x = torch.from_numpy(d["echo_input"]).float()
    occ = torch.from_numpy(d["occupancy_patch"]).float().flatten(1)
    wall = torch.from_numpy(d["wall_patch"]).float().flatten(1)
    door = torch.from_numpy(d["doorway_patch"]).float().flatten(1)
    pose = torch.from_numpy(d["pose_correction"]).float()
    conf = torch.from_numpy(d["confidence_target"]).float()

    model = EchoMapperNet(in_channels=x.shape[1], bins=x.shape[2], patch_size=int(np.sqrt(occ.shape[1])))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out = model(x)
        occ_p = (torch.sigmoid(out["occ"]) >= 0.5).float()
        wall_p = (torch.sigmoid(out["wall"]) >= 0.5).float()
        door_p = (torch.sigmoid(out["door"]) >= 0.5).float()
        conf_p = torch.sigmoid(out["conf"])

    occ_acc = float((occ_p == occ).float().mean().item())
    wall_acc = float((wall_p == wall).float().mean().item())
    door_acc = float((door_p == door).float().mean().item())
    pose_rmse = float(torch.sqrt(torch.mean((out["pose"] - pose) ** 2)).item())
    conf_corr = float(np.corrcoef(conf_p.squeeze().numpy(), conf.squeeze().numpy())[0, 1]) if len(conf) > 1 else 0.0

    return {
        "available": True,
        "occupancy_patch_accuracy": occ_acc,
        "wall_patch_accuracy": wall_acc,
        "doorway_patch_accuracy": door_acc,
        "pose_correction_rmse": pose_rmse,
        "confidence_correlation": conf_corr,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = {}
    if Path(args.baseline_results).exists():
        baseline = json.loads(Path(args.baseline_results).read_text(encoding="utf-8"))
    neural = evaluate_neural(args.dataset, args.checkpoint)

    comparison = {
        "phase": "Phase 2D baseline vs neural",
        "baseline_available": bool(baseline),
        "baseline_summary": baseline.get("difficulty_aggregate", {}) if baseline else {},
        "neural_summary": neural,
    }

    out_path = out_dir / "phase2_mapping_evaluation_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(comparison)


if __name__ == "__main__":
    main()
