"""
Generate a synthetic continuous-distance echo dataset for regression.

This script creates:
- wall samples with random continuous distances (0.10 m to 2.50 m)
- no_obstacle samples
- WAV recordings
- echo-focused spectrogram images
- a CSV file with labels/metadata

Why regression:
- Distance is naturally continuous, so predicting meters directly is often better than
  forcing many discrete distance classes.
- Random continuous distances improve generalization because the model sees many
  intermediate values, not only a few fixed points.
- Random noise/reflection/amplitude variation helps the model become robust to realistic
  sensing changes instead of overfitting to one clean synthetic pattern.
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
# Configuration
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

DATASET_ROOT = Path("datasets/synthetic_echoes_regression")
WAV_DIR = DATASET_ROOT / "wav"
SPEC_DIR = DATASET_ROOT / "spectrograms"
LABELS_CSV = DATASET_ROOT / "labels.csv"


def make_transmit_chirp() -> np.ndarray:
    """Create the base ultrasonic transmit chirp."""
    t = np.arange(int(CHIRP_DURATION_S * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


def place_signal(target: np.ndarray, signal: np.ndarray, start_idx: int, gain: float) -> None:
    """Add `signal * gain` into `target` starting at sample index `start_idx`."""
    if start_idx >= len(target):
        return
    end_idx = min(start_idx + len(signal), len(target))
    if end_idx <= start_idx:
        return
    target[start_idx:end_idx] += gain * signal[: end_idx - start_idx]


def expected_echo_delay_s(distance_m: float) -> float:
    """Round-trip time-of-flight delay."""
    return 2.0 * distance_m / SPEED_OF_SOUND


def simulate_sample(
    rng: np.random.Generator,
    tx_chirp: np.ndarray,
    has_obstacle: bool,
    distance_m: float | None,
) -> tuple[np.ndarray, float, float, float | None]:
    """
    Create one synthetic recording with randomized acoustic conditions.

    Randomized effects:
    - transmit chirp amplitude
    - reflection strength
    - additive measurement noise
    - small background noise
    - optional weak secondary reflection
    """
    waveform = np.zeros(int(RECORD_DURATION_S * SAMPLE_RATE), dtype=np.float32)

    # Random chirp amplitude (direct leakage).
    chirp_amplitude = rng.uniform(0.7, 1.1)
    place_signal(waveform, tx_chirp, start_idx=0, gain=chirp_amplitude)

    reflection_strength = 0.0
    echo_delay_s = None

    if has_obstacle and distance_m is not None:
        echo_delay_s = expected_echo_delay_s(distance_m)
        echo_idx = int(echo_delay_s * SAMPLE_RATE)

        # Primary reflection varies per sample.
        reflection_strength = rng.uniform(0.30, 0.95)
        place_signal(waveform, tx_chirp, start_idx=echo_idx, gain=reflection_strength)

        # Optional weak secondary reflection in some samples.
        if rng.uniform() < 0.35:
            extra_delay_s = rng.uniform(0.0007, 0.0060)
            sec_idx = int((echo_delay_s + extra_delay_s) * SAMPLE_RATE)
            sec_gain = reflection_strength * rng.uniform(0.12, 0.35)
            place_signal(waveform, tx_chirp, start_idx=sec_idx, gain=sec_gain)

    # Random noise terms.
    noise_level = rng.uniform(0.0015, 0.0075)
    background_noise = rng.uniform(0.0002, 0.0010)
    waveform += rng.normal(0.0, noise_level, size=waveform.shape[0]).astype(np.float32)
    waveform += rng.normal(0.0, background_noise, size=waveform.shape[0]).astype(np.float32)

    # Keep values in [-1, 1] for stable int16 WAV conversion.
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = np.clip(waveform / peak * 0.95, -1.0, 1.0)

    return waveform, reflection_strength, noise_level, echo_delay_s


def save_wav(path: Path, waveform: np.ndarray) -> None:
    """Write waveform as 16-bit PCM WAV."""
    pcm = np.int16(waveform * 32767.0)
    wavfile.write(path, SAMPLE_RATE, pcm)


def make_echo_focused_spectrogram_image(waveform: np.ndarray) -> np.ndarray:
    """
    Convert waveform into a fixed-size echo-focused spectrogram image.

    Echo focus means early direct chirp region (<0.8 ms) is removed to emphasize
    delayed echoes used for distance estimation.
    """
    freqs, times, psd = spectrogram(
        waveform,
        fs=SAMPLE_RATE,
        nperseg=256,
        noverlap=192,
        mode="psd",
        scaling="density",
    )
    psd_db = 10.0 * np.log10(psd + 1e-20)

    # Keep ultrasonic band and echo-focused time region.
    freq_mask = (freqs >= 20000.0) & (freqs <= 80000.0)
    time_mask = times >= ECHO_FOCUS_START_S
    band = psd_db[freq_mask][:, time_mask]

    # Normalize with fixed dB range for consistent supervision.
    band = np.clip(band, SPEC_VMIN_DB, SPEC_VMAX_DB)
    band_norm = (band - SPEC_VMIN_DB) / (SPEC_VMAX_DB - SPEC_VMIN_DB)

    # Resize to fixed CNN-friendly image size.
    zoom_y = SPEC_IMAGE_SIZE / band_norm.shape[0]
    zoom_x = SPEC_IMAGE_SIZE / band_norm.shape[1]
    resized = zoom(band_norm, (zoom_y, zoom_x), order=1)
    resized = np.clip(resized, 0.0, 1.0)

    # Convert to RGB image array with a stable colormap.
    cmap = matplotlib.colormaps["magma"]
    rgb = (cmap(resized)[..., :3] * 255.0).astype(np.uint8)
    return rgb


def save_spectrogram(path: Path, waveform: np.ndarray) -> None:
    """Create and save fixed-size echo-focused spectrogram PNG."""
    img = make_echo_focused_spectrogram_image(waveform)
    plt.imsave(path, img)


def generate_dataset(
    wall_samples: int,
    no_obstacle_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    tx_chirp = make_transmit_chirp()

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []

    # Wall samples with random continuous distances.
    for i in range(wall_samples):
        distance_m = float(rng.uniform(DISTANCE_MIN_M, DISTANCE_MAX_M))
        waveform, reflection_strength, noise_level, echo_delay_s = simulate_sample(
            rng=rng,
            tx_chirp=tx_chirp,
            has_obstacle=True,
            distance_m=distance_m,
        )

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
                "expected_echo_delay_ms": f"{(echo_delay_s * 1000.0):.6f}",
                "reflection_strength": f"{reflection_strength:.6f}",
                "noise_level": f"{noise_level:.6f}",
            }
        )

        if (i + 1) % 500 == 0:
            print(f"Generated wall samples: {i + 1}/{wall_samples}")

    # No-obstacle samples.
    for i in range(no_obstacle_samples):
        waveform, reflection_strength, noise_level, _ = simulate_sample(
            rng=rng,
            tx_chirp=tx_chirp,
            has_obstacle=False,
            distance_m=None,
        )

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
                "reflection_strength": f"{reflection_strength:.6f}",
                "noise_level": f"{noise_level:.6f}",
            }
        )

        if (i + 1) % 250 == 0:
            print(f"Generated no_obstacle samples: {i + 1}/{no_obstacle_samples}")

    # Save labels CSV.
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Created {wall_samples} wall samples and {no_obstacle_samples} no_obstacle samples.")
    print(f"WAV directory: {WAV_DIR}")
    print(f"Spectrogram directory: {SPEC_DIR}")
    print(f"Labels CSV: {LABELS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic continuous-distance echo regression dataset.")
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
