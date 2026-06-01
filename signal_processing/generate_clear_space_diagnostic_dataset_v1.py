"""
Generate a clear-space diagnostic acoustic dataset (v1).

Purpose:
- Isolate why CLEAR classification is difficult by contrasting true echoes against
  clear-space noise artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, correlate


DATASET_ROOT = Path("datasets/clear_space_diagnostic_v1")
WAV_ROOT = DATASET_ROOT / "wav"
CORR_ROOT = DATASET_ROOT / "correlation_features"
LABELS_CSV = DATASET_ROOT / "labels.csv"

CLASSES = [
    "obstacle_easy",
    "obstacle_weak",
    "clear_clean",
    "clear_noisy",
    "clear_motor_noise",
    "clear_random_spikes",
]

SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0
RECORD_DURATION_S = 0.03
DIST_MIN_M = 0.10
DIST_MAX_M = 2.50

CORR_FEATURE_LEN = 512
CORR_DIST_MIN_M = 0.05
CORR_DIST_MAX_M = 2.80
DIRECT_IGNORE_S = 0.00055


def make_transmit_chirp() -> np.ndarray:
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


def expected_echo_delay_s(distance_m: float) -> float:
    return 2.0 * distance_m / SPEED_OF_SOUND


def place_signal(target: np.ndarray, signal: np.ndarray, start_idx: int, gain: float) -> None:
    if start_idx >= len(target):
        return
    end_idx = min(start_idx + len(signal), len(target))
    if end_idx <= start_idx:
        return
    target[start_idx:end_idx] += gain * signal[: end_idx - start_idx]


def add_motor_like_noise(waveform: np.ndarray, rng: np.random.Generator) -> None:
    n = waveform.shape[0]
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tone_count = int(rng.integers(2, 5))
    for _ in range(tone_count):
        freq_hz = float(rng.uniform(120.0, 2200.0) if rng.uniform() < 0.6 else rng.uniform(18000.0, 32000.0))
        amp = float(rng.uniform(0.001, 0.008))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        waveform += amp * np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)


def add_random_spikes(waveform: np.ndarray, rng: np.random.Generator) -> None:
    num_spikes = int(rng.integers(1, 5))
    n = waveform.shape[0]
    for _ in range(num_spikes):
        idx = int(rng.integers(0, n))
        width = int(rng.integers(1, 7))
        amp = float(rng.uniform(0.05, 0.30)) * (1.0 if rng.uniform() < 0.5 else -1.0)
        end_idx = min(n, idx + width)
        waveform[idx:end_idx] += amp


def simulate_waveform(
    class_name: str,
    tx_chirp: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    waveform = np.zeros(int(RECORD_DURATION_S * SAMPLE_RATE), dtype=np.float32)

    # Direct leakage is always present in this setup.
    tx_gain = float(rng.uniform(0.6, 1.2))
    place_signal(waveform, tx_chirp, start_idx=0, gain=tx_gain)

    has_obstacle = int(class_name.startswith("obstacle_"))
    distance_m = None
    reflection_strength = 0.0
    noise_level = 0.0
    has_secondary = 0
    motor_noise_enabled = 0
    random_spike_enabled = 0

    if class_name == "obstacle_easy":
        distance_m = float(rng.uniform(DIST_MIN_M, DIST_MAX_M))
        delay_s = expected_echo_delay_s(distance_m)
        reflection_strength = float(rng.uniform(0.55, 1.05))
        place_signal(waveform, tx_chirp, int(delay_s * SAMPLE_RATE), reflection_strength)
        noise_level = float(rng.uniform(0.0010, 0.0040))

    elif class_name == "obstacle_weak":
        distance_m = float(rng.uniform(DIST_MIN_M, DIST_MAX_M))
        delay_s = expected_echo_delay_s(distance_m)
        reflection_strength = float(rng.uniform(0.01, 0.12))
        place_signal(waveform, tx_chirp, int(delay_s * SAMPLE_RATE), reflection_strength)
        noise_level = float(rng.uniform(0.0080, 0.0250))
        if rng.uniform() < 0.65:
            has_secondary = 1
            extra_delay_s = float(rng.uniform(0.0005, 0.0090))
            sec_gain = reflection_strength * float(rng.uniform(0.15, 0.90))
            place_signal(waveform, tx_chirp, int((delay_s + extra_delay_s) * SAMPLE_RATE), sec_gain)

    elif class_name == "clear_clean":
        noise_level = float(rng.uniform(0.0008, 0.0030))

    elif class_name == "clear_noisy":
        noise_level = float(rng.uniform(0.0080, 0.0280))

    elif class_name == "clear_motor_noise":
        noise_level = float(rng.uniform(0.0040, 0.0150))
        add_motor_like_noise(waveform, rng)
        motor_noise_enabled = 1

    elif class_name == "clear_random_spikes":
        noise_level = float(rng.uniform(0.0040, 0.0150))
        add_random_spikes(waveform, rng)
        random_spike_enabled = 1

    else:
        raise ValueError(f"Unknown class: {class_name}")

    # Background/random noise.
    background = float(rng.uniform(0.0001, 0.0020))
    waveform += rng.normal(0.0, noise_level, size=waveform.shape[0]).astype(np.float32)
    waveform += rng.normal(0.0, background, size=waveform.shape[0]).astype(np.float32)

    # Normalize to int16-safe range.
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = np.clip(waveform / peak * 0.95, -1.0, 1.0)

    meta = {
        "has_obstacle": has_obstacle,
        "distance_m": distance_m,
        "reflection_strength": reflection_strength,
        "noise_level": noise_level,
        "has_secondary_reflection": has_secondary,
        "motor_noise_enabled": motor_noise_enabled,
        "random_spike_enabled": random_spike_enabled,
    }
    return waveform, meta


def save_wav(path: Path, waveform: np.ndarray) -> None:
    pcm = np.int16(waveform * 32767.0)
    wavfile.write(path, SAMPLE_RATE, pcm)


def correlation_feature(signal: np.ndarray, tx_chirp: np.ndarray) -> np.ndarray:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clear-space diagnostic dataset v1.")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    WAV_ROOT.mkdir(parents=True, exist_ok=True)
    CORR_ROOT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tx_chirp = make_transmit_chirp()

    rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        for i in range(args.samples_per_class):
            fname = f"{class_name}_{i:05d}"
            wav_rel = Path("wav") / f"{fname}.wav"
            corr_rel = Path("correlation_features") / f"{fname}.npy"
            wav_abs = DATASET_ROOT / wav_rel
            corr_abs = DATASET_ROOT / corr_rel

            waveform, meta = simulate_waveform(class_name, tx_chirp, rng)
            save_wav(wav_abs, waveform)
            feat = correlation_feature(waveform, tx_chirp)
            np.save(corr_abs, feat)

            if int(meta["has_obstacle"]) == 1:
                distance_m = float(meta["distance_m"])
            else:
                distance_m = None

            rows.append(
                {
                    "filename": fname,
                    "class_name": class_name,
                    "has_obstacle": int(meta["has_obstacle"]),
                    "distance_m": "" if distance_m is None else f"{distance_m:.6f}",
                    "wav_path": str(wav_rel).replace("\\", "/"),
                    "correlation_path": str(corr_rel).replace("\\", "/"),
                    "reflection_strength": f"{float(meta['reflection_strength']):.6f}",
                    "noise_level": f"{float(meta['noise_level']):.6f}",
                    "has_secondary_reflection": int(meta["has_secondary_reflection"]),
                    "motor_noise_enabled": int(meta["motor_noise_enabled"]),
                    "random_spike_enabled": int(meta["random_spike_enabled"]),
                }
            )

        print(f"Generated class: {class_name} ({args.samples_per_class} samples)")

    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "class_name",
                "has_obstacle",
                "distance_m",
                "wav_path",
                "correlation_path",
                "reflection_strength",
                "noise_level",
                "has_secondary_reflection",
                "motor_noise_enabled",
                "random_spike_enabled",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Saved labels: {LABELS_CSV}")
    print(f"WAV root: {WAV_ROOT}")
    print(f"Correlation root: {CORR_ROOT}")


if __name__ == "__main__":
    main()
