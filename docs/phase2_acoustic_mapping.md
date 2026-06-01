# Phase 2 - Acoustic Mapping and Neural Echo Interpretation

Phase 2 is separate from Phase 1 navigation policy tuning.

## Scope
- Phase 1: acoustic navigation and coverage behavior.
- Phase 2: acoustic mapping quality.

## Current accepted navigation mapper
- Current accepted navigation mapper: **Phase 2C.5 hybrid calibrated mapper** (`phase2c5_hybrid_acoustic_mapper`).
- Default navigation status: `recommended_default`.
- Manifest: `runs/accepted_models/phase2c5_hybrid_acoustic_mapper/manifest.json`.

## Current accepted mapper-guided navigation
- **Phase 2D.2 mapper-guided navigation v3** (`phase2d_mapper_guided_navigation_v3`).
- Manifest: `runs/accepted_models/phase2d_mapper_guided_navigation_v3/manifest.json`.

## Module layout
- `simulation/phase2_mapping/mapping_utils.py`
  - shared map geometry, acoustic simulation, and mapping metrics utilities.
- `simulation/phase2_mapping/acoustic_mapping_baseline.py`
  - Phase 2A classical baseline (inverse sensor updates into occupancy/wall/doorway/confidence maps + pose estimate).
- `simulation/phase2_mapping/echo_dataset_generator.py`
  - Phase 2B supervised dataset generation for neural echo mapping.
- `simulation/phase2_mapping/train_echo_mapper.py`
  - Phase 2C small neural echo mapper training (1D CNN, multi-head outputs).
- `simulation/phase2_mapping/evaluate_mapping.py`
  - Phase 2D baseline vs neural comparison.
- `simulation/simple_2d_acoustic_mapping_phase2.py`
  - convenience runner for Phase 2A baseline.

## Outputs
- Runs: `runs/phase2_mapping/`
  - `phase2_mapping_results.json`
  - `per_map_metrics.json`
  - optional plot images under `runs/phase2_mapping/plots/`
- Dataset: `datasets/phase2_echo_mapping/`
  - `phase2_echo_mapping_dataset.npz`
  - metadata and summary JSON

## Baseline metrics (Phase 2A)
- map accuracy (occupancy)
- wall reconstruction error
- doorway precision/recall
- localization RMSE
- map confidence quality

Secondary behavior metrics are also logged (collision/timeout/action distribution) but are not the primary acceptance objective for Phase 2.

## Current publish pack
- Curated release visuals: `runs/publish/phase2_release_2026-06-01/`
- Selection manifest: `runs/publish/phase2_release_2026-06-01/publish_manifest.json`
