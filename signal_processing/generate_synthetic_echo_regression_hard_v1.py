"""
Generate a harder synthetic continuous-distance echo dataset (hard_v1).

This dataset is intentionally more challenging than the clean regression dataset.

Why keep hard_v1 separate:
- A separate dataset makes benchmarking fair and repeatable.
- You can compare clean vs hard_v1 performance directly without mixing distributions.

Why these harder effects matter:
- Secondary reflections matter because indoor echoes often include multiple paths,
  not just one clean wall return.
- Surface absorption reduces reflected energy, so some obstacles produce weaker echoes.
- Motor-like/high-noise examples are important for future drone use because onboard
  electronics and propellers can raise the acoustic noise floor.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import zoom
from scipy.signal import chirp, spectrogram


# -----------------------------
# Acoustic and dataset settings
# -----------------------------
SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0
RECORD_DURATION_S = 0.03
ECHO_FOCUS_START_S = 0.0008

WALL_SAMPLES_DEFAULT = 5000
NO_OBSTACLE_SAMPLES_DEFAULT = 1000
DISTANCE_MIN_M = 0.10
DISTANCE_MAX_M = 2.50

SPEC_VMIN_DB = -130.0
SPEC_VMAX_DB = -50.0
SPEC_IMAGE_SIZE = 128

DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")
WAV_DIR = DATASET_ROOT / "wav"
SPEC_DIR = DATASET_ROOT / "spectrograms"
LABELS_CSV = DATASET_ROOT / "labels.csv"


def make_transmit_chirp() -> np.ndarray:
    """Create base transmit chirp."""
    t = np.arange(int(CHIRP_DURATION_S * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


def expected_echo_delay_s(distance_m: float) -> float:
    """Round-trip echo time-of-flight."""
    return 2.0 * distance_m / SPEED_OF_SOUND


def place_signal(target: np.ndarray, signal: np.ndarray, start_idx: int, gain: float) -> None:
    """Add scaled signal into target at a given start index."""
    if start_idx >= len(target):
        return
    end_idx = min(start_idx + len(signal), len(target))
    if end_idx <= start_idx:
        return
    target[start_idx:end_idx] += gain * signal[: end_idx - start_idx]


def add_motor_like_noise(waveform: np.ndarray, rng: np.random.Generator) -> None:
    """
    Add optional tonal interference to mimic motor/electronics contamination.
    """
    if rng.uniform() > 0.30:
        return

    n = waveform.shape[0]
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tone_count = int(rng.integers(1, 4))

    for _ in range(tone_count):
        # Mix low-frequency and ultrasonic interference components.
        if rng.uniform() < 0.5:
            freq_hz = float(rng.uniform(120.0, 1800.0))
        else:
            freq_hz = float(rng.uniform(18000.0, 32000.0))
        amp = float(rng.uniform(0.0008, 0.006))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        waveform += amp * np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)


def simulate_sample(
    rng: np.random.Generator,
    tx_chirp: np.ndarray,
    has_obstacle: bool,
    distance_m: float | None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """
    Simulate one hard_v1 ultrasonic recording.

    Harder elements:
    - wider chirp amplitude and reflection strength variation
    - random absorption factor
    - occasional very weak echoes
    - random secondary reflection with random delay/strength
    - wider noise and occasional high-noise outliers
    - optional motor-like tonal interference
    """
    waveform = np.zeros(int(RECORD_DURATION_S * SAMPLE_RATE), dtype=np.float32)

    # Random direct chirp amplitude.
    chirp_amplitude = float(rng.uniform(0.5, 1.35))
    place_signal(waveform, tx_chirp, start_idx=0, gain=chirp_amplitude)

    reflection_strength = 0.0
    surface_absorption = float(rng.uniform(0.10, 0.75))
    has_secondary_reflection = 0
    secondary_delay_ms = ""
    secondary_strength = ""
    echo_delay_s = None

    if has_obstacle and distance_m is not None:
        echo_delay_s = expected_echo_delay_s(distance_m)
        echo_idx = int(echo_delay_s * SAMPLE_RATE)

        # Main reflection with wide variation and absorption.
        base_reflection = float(rng.uniform(0.12, 1.20))
        reflection_strength = base_reflection * (1.0 - surface_absorption)

        # Occasional very weak echoes.
        if rng.uniform() < 0.18:
            reflection_strength *= float(rng.uniform(0.05, 0.25))

        reflection_strength = float(max(0.01, reflection_strength))
        place_signal(waveform, tx_chirp, start_idx=echo_idx, gain=reflection_strength)

        # Secondary reflection from multipath.
        if rng.uniform() < 0.60:
            has_secondary_reflection = 1
            extra_delay_s = float(rng.uniform(0.0004, 0.0090))
            sec_idx = int((echo_delay_s + extra_delay_s) * SAMPLE_RATE)
            sec_gain = reflection_strength * float(rng.uniform(0.06, 0.55))
            place_signal(waveform, tx_chirp, start_idx=sec_idx, gain=sec_gain)

            secondary_delay_ms = f"{extra_delay_s * 1000.0:.6f}"
            secondary_strength = f"{sec_gain:.6f}"

    # Wider baseline noise distribution.
    noise_level = float(rng.uniform(0.0010, 0.0150))

    # Occasional high-noise (hard outlier) samples.
    if rng.uniform() < 0.12:
        noise_level *= float(rng.uniform(1.8, 3.8))

    background_noise = float(rng.uniform(0.0002, 0.0040))
    waveform += rng.normal(0.0, noise_level, size=waveform.shape[0]).astype(np.float32)
    waveform += rng.normal(0.0, background_noise, size=waveform.shape[0]).astype(np.float32)
    add_motor_like_noise(waveform, rng)

    # Normalize to int16-safe range.
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = np.clip(waveform / peak * 0.95, -1.0, 1.0)

    meta: dict[str, float | int | str] = {
        "expected_echo_delay_ms": "" if echo_delay_s is None else f"{echo_delay_s * 1000.0:.6f}",
        "reflection_strength": reflection_strength,
        "noise_level": noise_level,
        "surface_absorption": surface_absorption,
        "has_secondary_reflection": has_secondary_reflection,
        "secondary_delay_ms": secondary_delay_ms,
        "secondary_strength": secondary_strength,
    }
    return waveform, meta


def save_wav(path: Path, waveform: np.ndarray) -> None:
    """Save waveform as 16-bit PCM WAV."""
    pcm = np.int16(waveform * 32767.0)
    wavfile.write(path, SAMPLE_RATE, pcm)


def make_echo_focused_spectrogram_image(waveform: np.ndarray) -> np.ndarray:
    """Create fixed-size 128x128 echo-focused spectrogram image."""
    freqs, times, psd = spectrogram(
        waveform,
        fs=SAMPLE_RATE,
        nperseg=256,
        noverlap=192,
        mode="psd",
        scaling="density",
    )
    psd_db = 10.0 * np.log10(psd + 1e-20)

    freq_mask = (freqs >= 20000.0) & (freqs <= 80000.0)
    time_mask = times >= ECHO_FOCUS_START_S
    band = psd_db[freq_mask][:, time_mask]

    band = np.clip(band, SPEC_VMIN_DB, SPEC_VMAX_DB)
    band_norm = (band - SPEC_VMIN_DB) / (SPEC_VMAX_DB - SPEC_VMIN_DB)

    zoom_y = SPEC_IMAGE_SIZE / band_norm.shape[0]
    zoom_x = SPEC_IMAGE_SIZE / band_norm.shape[1]
    resized = zoom(band_norm, (zoom_y, zoom_x), order=1)
    resized = np.clip(resized, 0.0, 1.0)

    cmap = matplotlib.colormaps["magma"]
    rgb = (cmap(resized)[..., :3] * 255.0).astype(np.uint8)
    return rgb


def save_spectrogram(path: Path, waveform: np.ndarray) -> None:
    """Save echo-focused spectrogram PNG."""
    img = make_echo_focused_spectrogram_image(waveform)
    plt.imsave(path, img)


def generate_dataset(wall_samples: int, no_obstacle_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    tx_chirp = make_transmit_chirp()

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []

    for i in range(wall_samples):
        distance_m = float(rng.uniform(DISTANCE_MIN_M, DISTANCE_MAX_M))
        waveform, meta = simulate_sample(rng, tx_chirp, has_obstacle=True, distance_m=distance_m)

        file_id = f"wall_{i:05d}"
        wav_rel = Path("wav") / f"{file_id}.wav"
        spec_rel = Path("spectrograms") / f"{file_id}.png"

        save_wav(DATASET_ROOT / wav_rel, waveform)
        save_spectrogram(DATASET_ROOT / spec_rel, waveform)

        rows.append(
            {
                "filename": file_id,
                "spectrogram_path": str(spec_rel).replace("\\", "/"),
                "wav_path": str(wav_rel).replace("\\", "/"),
                "has_obstacle": 1,
                "distance_m": f"{distance_m:.6f}",
                "expected_echo_delay_ms": str(meta["expected_echo_delay_ms"]),
                "reflection_strength": f"{float(meta['reflection_strength']):.6f}",
                "noise_level": f"{float(meta['noise_level']):.6f}",
                "surface_absorption": f"{float(meta['surface_absorption']):.6f}",
                "has_secondary_reflection": int(meta["has_secondary_reflection"]),
                "secondary_delay_ms": str(meta["secondary_delay_ms"]),
                "secondary_strength": str(meta["secondary_strength"]),
            }
        )

        if (i + 1) % 500 == 0:
            print(f"Generated wall samples: {i + 1}/{wall_samples}")

    for i in range(no_obstacle_samples):
        waveform, meta = simulate_sample(rng, tx_chirp, has_obstacle=False, distance_m=None)

        file_id = f"no_obstacle_{i:05d}"
        wav_rel = Path("wav") / f"{file_id}.wav"
        spec_rel = Path("spectrograms") / f"{file_id}.png"

        save_wav(DATASET_ROOT / wav_rel, waveform)
        save_spectrogram(DATASET_ROOT / spec_rel, waveform)

        rows.append(
            {
                "filename": file_id,
                "spectrogram_path": str(spec_rel).replace("\\", "/"),
                "wav_path": str(wav_rel).replace("\\", "/"),
                "has_obstacle": 0,
                "distance_m": "",
                "expected_echo_delay_ms": "",
                "reflection_strength": f"{float(meta['reflection_strength']):.6f}",
                "noise_level": f"{float(meta['noise_level']):.6f}",
                "surface_absorption": f"{float(meta['surface_absorption']):.6f}",
                "has_secondary_reflection": int(meta["has_secondary_reflection"]),
                "secondary_delay_ms": "",
                "secondary_strength": "",
            }
        )

        if (i + 1) % 250 == 0:
            print(f"Generated no_obstacle samples: {i + 1}/{no_obstacle_samples}")

    with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "spectrogram_path",
                "wav_path",
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

    print(f"Done. Created {wall_samples} wall samples and {no_obstacle_samples} no_obstacle samples.")
    print(f"WAV directory: {WAV_DIR}")
    print(f"Spectrogram directory: {SPEC_DIR}")
    print(f"Labels CSV: {LABELS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate harder synthetic regression dataset (hard_v1).")
    parser.add_argument("--wall-samples", type=int, default=WALL_SAMPLES_DEFAULT, help="Number of wall samples.")
    parser.add_argument(
        "--no-obstacle-samples",
        type=int,
        default=NO_OBSTACLE_SAMPLES_DEFAULT,
        help="Number of no_obstacle samples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_dataset(
        wall_samples=args.wall_samples,
        no_obstacle_samples=args.no_obstacle_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
