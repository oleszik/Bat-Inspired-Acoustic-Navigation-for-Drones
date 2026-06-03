# Accepted sim_env Sandbox Policy

Accepted policy: `adaptive_hybrid`

sim_env version: `v1.6`

Accepted date: `2026-06-01`

`adaptive_hybrid` is the recommended sandbox policy for visual simulation and replay. It combines `stabilized_frontier` navigation with `coverage_sweep` exploration, switching between them based on local obstacle density, repeated forward blockage, doorway/corridor structure, and coverage plateau diagnostics.

## Accepted Manifests

- Mapper: `runs/accepted_models/phase2c5_hybrid_acoustic_mapper/manifest.json`
- Navigation: `runs/accepted_models/phase2d_mapper_guided_navigation_v3/manifest.json`
- Policy manifest: `sim_env/outputs/accepted_policy_manifest.json`

## Validation

- `python -m py_compile` passed.
- `pytest tests/test_navigation_safety.py` passed with `2 passed`.
- All accepted replay comparisons preserved `collision_count = 0`.

## Results

| map | difficulty | comparison policy | comparison coverage | adaptive_hybrid coverage | collision_count | SF steps | sweep steps | switches | revisit_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| doorway | clean | stabilized_frontier | 0.6753 | 0.7437 | 0 | 460 | 240 | 12 | 0.8329 |
| cluttered_room | medium_noise | coverage_sweep | 0.6330 | 0.8830 | 0 | 274 | 426 | 11 | 0.8357 |
| corridor | clean | stabilized_frontier | 0.4838 | 0.5724 | 0 | 392 | 308 | 13 | 0.8457 |
| single_block | hard_noise | coverage_sweep | 0.4322 | 1.0000 | 0 | 484 | 216 | 9 | 0.7743 |

## Notes

- `adaptive_hybrid` improved coverage on all tested maps while preserving zero collisions.
- `simple`, `stabilized_frontier`, `route_committed`, and `coverage_sweep` remain useful diagnostic modes.
- Explicit accepted doorway decisions are still not active and can be improved later.
