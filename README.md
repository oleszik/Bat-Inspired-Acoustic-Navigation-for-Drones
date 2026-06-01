# Acoustic Drone Navigation and Mapping

This repository contains a bat-inspired ultrasonic perception stack with two separated tracks:

- `Phase 1`: acoustic navigation policy/simulation tuning
- `Phase 2`: acoustic mapping and neural echo interpretation

## Current accepted components

- Accepted mapper:
  - `runs/accepted_models/phase2c5_hybrid_acoustic_mapper/manifest.json`
- Accepted mapper-guided navigation:
  - `runs/accepted_models/phase2d_mapper_guided_navigation_v3/manifest.json`

## Repository layout

- `signal_processing/`
  - dataset generation, matched-filter baselines, acoustic feature extraction
- `neural_network/`
  - supervised CNNs/regressors, hybrid evaluators, confidence/safety logic
- `simulation/phase2_mapping/`
  - Phase 2 mapping pipeline (baseline mapper, dataset generator, neural mapper, readiness eval)
- `sim_env/`
  - clean visual simulation sandbox for reusable demos
- `runs/`
  - experiment outputs, accepted-model manifests
- `datasets/`
  - generated synthetic datasets
- `docs/`
  - project documentation

## Phase 2 status snapshot

- Mapper-guided navigation v3 (accepted):
  - success_rate: `0.7817`
  - collision_rate: `0.0000`
  - fake_doorway_approach_rate: `0.0000`
  - doorway_crossing_success_rate: `0.9283`
  - coverage: `0.6608`
  - timeout_rate: `0.0617`

## Quick demos (`sim_env`)

```bash
python sim_env/examples/run_basic_world.py --map doorway --difficulty clean --steps 100 --save-plots
python sim_env/examples/run_mapper_demo.py --map doorway --difficulty clean --steps 100 --save-plots
python sim_env/examples/run_navigation_demo.py --map doorway --difficulty clean --steps 150 --save-plots
```

Outputs are saved under `sim_env/outputs/`.

## Recommended published figures

A curated publish set is prepared in:

- `runs/publish/phase2_release_2026-06-01/`

with selection metadata in:

- `runs/publish/phase2_release_2026-06-01/publish_manifest.json`

## Notes

- `simulation/phase2_mapping/` remains the primary training/evaluation pipeline.
- `sim_env/` is intentionally lightweight (`numpy`, `matplotlib`) and is not an RL training loop.
