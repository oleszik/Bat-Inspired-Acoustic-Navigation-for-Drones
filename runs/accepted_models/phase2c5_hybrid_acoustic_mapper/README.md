# Phase 2C.5 Hybrid Acoustic Mapper (Accepted)

This folder records the accepted mapper manifest for the promoted hybrid-calibrated Phase 2C.5 acoustic mapper.

## Accepted status
- `accepted_model_name`: `phase2c5_hybrid_acoustic_mapper`
- `navigation_status`: `recommended_default`
- `v6_recommended_default_for_navigation`: `true`

## Selected hybrid modes
- occupancy: `v4_occ @ 0.60`
- wall: `v4_wall @ 0.90`
- doorway: `v5_context @ 0.85`
- free: `v4_free @ 0.50`

## Fallback
- doorway fallback model: `v5_context`
- doorway fallback threshold: `0.85`

## Why this is accepted
- Highest navigation readiness score among compared models (`v6 > v5 > v4`).
- Lower fake-doorway approach rate than v4.
- Collision-risk metric is not worse than v4.
- Doorway crossing is slightly below v5, but within configured tolerance (`0.03`).

See [`manifest.json`](./manifest.json) for exact numeric values and source file references.
