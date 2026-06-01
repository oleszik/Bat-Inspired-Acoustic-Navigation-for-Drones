# Repository Structure Guide

This project includes both **source code** and **generated research artifacts**.

## Source code

- `signal_processing/`: acoustic signal simulation and feature extraction scripts
- `neural_network/`: training/evaluation scripts for classifiers/regressors
- `simulation/`: navigation and mapping simulation pipelines
- `sim_env/`: lightweight visual sandbox environment
- `tests/`: automated tests

## Generated outputs

- `runs/`: experiment outputs, metrics JSON, plots, accepted-model manifests
- `runs/publish/`: curated publish-ready figure packs

## Generated datasets

- `datasets/`: synthetic dataset outputs used by training/evaluation pipelines

## Recommended workflow

1. Keep source code and tests under version control.
2. Keep only important artifacts from `runs/`:
   - accepted manifests
   - curated publish packs
3. Re-generate heavy datasets/results as needed from scripts.
