# sim_env

`sim_env/` is the clean, reusable visual simulation sandbox for this project.

It is intentionally separated from `simulation/phase2_mapping/`, which remains the training/evaluation pipeline.

## Integrated accepted manifests

- Mapper manifest:
  - `runs/accepted_models/phase2c5_hybrid_acoustic_mapper/manifest.json`
- Navigation manifest:
  - `runs/accepted_models/phase2d_mapper_guided_navigation_v3/manifest.json`

## Current recommended sandbox policy

`adaptive_hybrid` is the recommended `sim_env` v1.6 policy for visual simulation and replay.

It combines `stabilized_frontier` and `coverage_sweep`, switching based on local obstacle density, repeated blockage, doorway/corridor structure, and coverage plateau diagnostics. It improved coverage on the accepted replay set while preserving `collision_count = 0`.

- Accepted policy doc: `sim_env/ACCEPTED_POLICY.md`
- Accepted policy manifest: `sim_env/outputs/accepted_policy_manifest.json`
- Accepted replay GIFs: `sim_env/outputs/accepted_replays/`

The older modes remain available as diagnostic modes:

- `simple`
- `stabilized_frontier`
- `route_committed`
- `coverage_sweep`

## Folder structure

- `acoustic_world/`
  - `environments.py`: map definitions (`empty_room`, `corridor`, `single_block`, `doorway`, `cluttered_room`)
  - `agent.py`: simple 2D agent dynamics and actions
  - `acoustic_sensor.py`: simplified bat-like ray acoustics + echo vectors
  - `mapper_bridge.py`: manifest loading + dummy map prediction placeholder
  - `navigation_bridge.py`: manifest loading + simple action policy placeholder
  - `renderer.py`: matplotlib visualization utilities
- `examples/`
  - `run_basic_world.py`
  - `run_mapper_demo.py`
  - `run_navigation_demo.py`
  - `run_navigation_replay.py`
- `outputs/`
  - generated figures from demos

## Demo commands

```bash
python sim_env/examples/run_basic_world.py --map doorway --difficulty clean --steps 100 --save-plots
python sim_env/examples/run_mapper_demo.py --map doorway --difficulty clean --steps 100 --save-plots
python sim_env/examples/run_navigation_demo.py --map doorway --difficulty clean --steps 150 --save-plots
```

## Dependency policy

- Lightweight only:
  - Python standard library
  - `numpy`
  - `matplotlib`
- No Gymnasium/Pygame/PyBullet/RL loop in this sandbox.
