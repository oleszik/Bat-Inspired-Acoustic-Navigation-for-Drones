# Phase 2D.2 Accepted Navigation Model

Accepted model: `phase2d_mapper_guided_navigation_v3`

- Dependency mapper: `phase2c5_hybrid_acoustic_mapper`
- Mapper manifest: `runs/accepted_models/phase2c5_hybrid_acoustic_mapper/manifest.json`
- Navigation script: `simulation/phase2_mapping/mapper_guided_navigation_v3.py`
- Results JSON: `runs/phase2_mapper_guided_navigation_v3/mapper_guided_navigation_results.json`
- Status: `accepted`
- Acceptance date: `2026-06-01`

## Why accepted
- Meets all acceptance checks for coverage, safety, doorway behavior, timeout, and map-quality.
- Preserves zero collisions and zero fake doorway approaches.
- Improves coverage and success over v2.

## Fallback/Dependency note
This model depends on the accepted Phase 2C.5 hybrid acoustic mapper.
