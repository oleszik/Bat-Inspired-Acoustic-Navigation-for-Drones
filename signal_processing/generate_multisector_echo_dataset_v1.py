"""
Generate synthetic multi-sector ultrasonic echo dataset (v1).

Why sector-based sensing:
- A drone must decide where free space exists, not only front distance.
- Splitting perception into sectors (left/front/right directions) is a simple step
  toward directional navigation decisions.

Why each sector has its own state and distance:
- Obstacles can appear at different ranges in different directions.
- Sector-wise labels let downstream models learn direction-conditioned behavior.

Why this is still simpler than a true microphone array:
- We simulate independent directional sectors, not full spatial beamforming.
- It is a practical intermediate stage before complex array geometry modeling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, correlate


# -----------------------------
# Dataset and acoustic settings
# -----------------------------
DATASET_ROOT = Path("datasets/synthetic_echoes_multisector_v1")
WAV_ROOT = DATASET_ROOT / "wav"
CORR_ROOT = DATASET_ROOT / "correlation_features"
LABELS_CSV = DATASET_ROOT / "labels.csv"

SECTORS = ["left", "front_left", "front", "front_right", "right"]

SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0
RECORD_DURATION_S = 0.03

DIST_MIN_M = 0.10
DIST_MAX_M = 2.50

# Correlation feature settings.
CORR_FEATURE_LEN = 512
CORR_DIST_MIN_M = 0.05
CORR_DIST_MAX_M = 2.80
DIRECT_IGNORE_S = 0.00055

# Hard-v1 style stochastic behavior.
SECTOR_OBSTACLE_PROB = 0.52
HIGH_NOISE_PROB = 0.12
WEAK_ECHO_PROB = 0.18
SECONDARY_REFLECTION_PROB = 0.60
MOTOR_NOISE_PROB = 0.30


def make_transmit_chirp() -> np.ndarray:
    """Create the base transmit chirp."""
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


def place_signal(target: np.ndarray, signal: np.ndarray, start_idx: int, gain: float) -> None:
    if start_idx >= len(target):
        return
    end_idx = min(start_idx + len(signal), len(target))
    if end_idx <= start_idx:
        return
    target[start_idx:end_idx] += gain * signal[: end_idx - start_idx]


def expected_echo_delay_s(distance_m: float) -> float:
    return 2.0 * distance_m / SPEED_OF_SOUND


def add_motor_like_noise(waveform: np.ndarray, rng: np.random.Generator) -> None:
    """Optional tonal contamination to mimic motors/electronics."""
    if rng.uniform() > MOTOR_NOISE_PROB:
        return
    n = waveform.shape[0]
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tone_count = int(rng.integers(1, 4))
    for _ in range(tone_count):
        if rng.uniform() < 0.5:
            freq_hz = float(rng.uniform(120.0, 1800.0))
        else:
            freq_hz = float(rng.uniform(18000.0, 32000.0))
        amp = float(rng.uniform(0.0008, 0.006))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        waveform += amp * np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)


def simulate_sector_waveform(
    rng: np.random.Generator,
    tx_chirp: np.ndarray,
    has_obstacle: bool,
    distance_m: float | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """
    Simulate one sector waveform with hard_v1-style variability.
    """
    waveform = np.zeros(int(RECORD_DURATION_S * SAMPLE_RATE), dtype=np.float32)

    # Direct transmit leakage with amplitude variation.
    chirp_amp = float(rng.uniform(0.5, 1.35))
    place_signal(waveform, tx_chirp, start_idx=0, gain=chirp_amp)

    reflection_strength = 0.0
    surface_absorption = float(rng.uniform(0.10, 0.75))
    has_secondary = 0
    secondary_delay_ms = ""
    secondary_strength = ""
    echo_delay_s = None

    if has_obstacle and distance_m is not None:
        echo_delay_s = expected_echo_delay_s(distance_m)
        echo_idx = int(echo_delay_s * SAMPLE_RATE)

        base_reflection = float(rng.uniform(0.12, 1.20))
        reflection_strength = base_reflection * (1.0 - surface_absorption)

        if rng.uniform() < WEAK_ECHO_PROB:
            reflection_strength *= float(rng.uniform(0.05, 0.25))
        reflection_strength = float(max(0.01, reflection_strength))
        place_signal(waveform, tx_chirp, start_idx=echo_idx, gain=reflection_strength)

        if rng.uniform() < SECONDARY_REFLECTION_PROB:
            has_secondary = 1
            extra_delay_s = float(rng.uniform(0.0004, 0.0090))
            sec_idx = int((echo_delay_s + extra_delay_s) * SAMPLE_RATE)
            sec_gain = reflection_strength * float(rng.uniform(0.06, 0.55))
            place_signal(waveform, tx_chirp, start_idx=sec_idx, gain=sec_gain)
            secondary_delay_ms = f"{extra_delay_s * 1000.0:.6f}"
            secondary_strength = f"{sec_gain:.6f}"

    noise_level = float(rng.uniform(0.0010, 0.0150))
    if rng.uniform() < HIGH_NOISE_PROB:
        noise_level *= float(rng.uniform(1.8, 3.8))
    background_noise = float(rng.uniform(0.0002, 0.0040))

    waveform += rng.normal(0.0, noise_level, size=waveform.shape[0]).astype(np.float32)
    waveform += rng.normal(0.0, background_noise, size=waveform.shape[0]).astype(np.float32)
    add_motor_like_noise(waveform, rng)

    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = np.clip(waveform / peak * 0.95, -1.0, 1.0)

    meta: dict[str, float | int | str] = {
        "expected_echo_delay_ms": "" if echo_delay_s is None else f"{echo_delay_s * 1000.0:.6f}",
        "reflection_strength": reflection_strength,
        "noise_level": noise_level,
        "surface_absorption": surface_absorption,
        "has_secondary_reflection": has_secondary,
        "secondary_delay_ms": secondary_delay_ms,
        "secondary_strength": secondary_strength,
    }
    return waveform, meta


def save_wav(path: Path, waveform: np.ndarray) -> None:
    pcm = np.int16(waveform * 32767.0)
    wavfile.write(path, SAMPLE_RATE, pcm)


def extract_correlation_feature(signal: np.ndarray, tx_chirp: np.ndarray) -> np.ndarray:
    """
    Build fixed-length matched-filter correlation feature vector.
    """
    corr = correlate(signal, tx_chirp, mode="full", method="fft")
    corr_mag = np.abs(corr).astype(np.float32)

    lags = np.arange(-len(tx_chirp) + 1, len(signal), dtype=np.int64)
    delays_s = lags.astype(np.float64) / SAMPLE_RATE

    min_delay_s = 2.0 * CORR_DIST_MIN_M / SPEED_OF_SOUND
    max_delay_s = 2.0 * CORR_DIST_MAX_M / SPEED_OF_SOUND
    keep = (delays_s >= min_delay_s) & (delays_s <= max_delay_s)

    d = delays_s[keep]
    v = corr_mag[keep].copy()
    v[d < DIRECT_IGNORE_S] = 0.0

    grid = np.linspace(min_delay_s, max_delay_s, CORR_FEATURE_LEN, dtype=np.float64)
    feat = np.interp(grid, d, v).astype(np.float32)
    m = float(np.max(np.abs(feat)))
    if m > 1e-12:
        feat = feat / m
    else:
        feat = np.zeros_like(feat, dtype=np.float32)
    return feat


def choose_sector_obstacles(rng: np.random.Generator) -> list[bool]:
    """
    Randomly choose obstacle presence per sector.
    Independent Bernoulli draws naturally produce multi-obstacle scenes.
    """
    return [bool(rng.uniform() < SECTOR_OBSTACLE_PROB) for _ in SECTORS]


def generate_dataset(num_scenes: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tx_chirp = make_transmit_chirp()

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    WAV_ROOT.mkdir(parents=True, exist_ok=True)
    CORR_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for scene_idx in range(num_scenes):
        sample_id = f"scene_{scene_idx:05d}"
        sector_flags = choose_sector_obstacles(rng)

        for sector, has_obs in zip(SECTORS, sector_flags):
            if has_obs:
                distance_m = float(rng.uniform(DIST_MIN_M, DIST_MAX_M))
            else:
                distance_m = None

            waveform, meta = simulate_sector_waveform(
                rng=rng,
                tx_chirp=tx_chirp,
                has_obstacle=has_obs,
                distance_m=distance_m,
            )

            base_name = f"{sample_id}_{sector}"
            wav_rel = Path("wav") / f"{base_name}.wav"
            corr_rel = Path("correlation_features") / f"{base_name}.npy"

            wav_abs = DATASET_ROOT / wav_rel
            corr_abs = DATASET_ROOT / corr_rel

            save_wav(wav_abs, waveform)
            corr_feat = extract_correlation_feature(waveform, tx_chirp)
            np.save(corr_abs, corr_feat)

            rows.append(
                {
                    "sample_id": sample_id,
                    "sector": sector,
                    "wav_path": str(wav_rel).replace("\\", "/"),
                    "correlation_path": str(corr_rel).replace("\\", "/"),
                    "has_obstacle": 1 if has_obs else 0,
                    "distance_m": f"{distance_m:.6f}" if distance_m is not None else "",
                    "expected_echo_delay_ms": str(meta["expected_echo_delay_ms"]),
                    "reflection_strength": f"{float(meta['reflection_strength']):.6f}",
                    "noise_level": f"{float(meta['noise_level']):.6f}",
                    "surface_absorption": f"{float(meta['surface_absorption']):.6f}",
                    "has_secondary_reflection": int(meta["has_secondary_reflection"]),
                    "secondary_delay_ms": str(meta["secondary_delay_ms"]),
                    "secondary_strength": str(meta["secondary_strength"]),
                }
            )

        if (scene_idx + 1) % 500 == 0:
            print(f"Generated scenes: {scene_idx + 1}/{num_scenes}")

    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "sector",
                "wav_path",
                "correlation_path",
                "has_obstacle",
                "distance_m",
                "expected_echo_delay_ms",
                "reflection_strength",
                "noise_level",
                "surface_absorption",
                "has_secondary_reflection",
                "secondary_delay_ms",
                "secondary_strength",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Generated {num_scenes} scenes, {num_scenes * len(SECTORS)} sector samples.")
    print(f"WAV root: {WAV_ROOT}")
    print(f"Correlation feature root: {CORR_ROOT}")
    print(f"Labels CSV: {LABELS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-sector synthetic echo dataset v1.")
    parser.add_argument("--num-scenes", type=int, default=5000, help="Number of multi-sector scenes to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_dataset(num_scenes=args.num_scenes, seed=args.seed)


if __name__ == "__main__":
    main()
