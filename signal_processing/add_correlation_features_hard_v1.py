"""
Add matched-filter correlation features to the hard_v1 regression dataset.

This script does not retrain any model. It only adds physics-informed features.

Why matched filtering helps:
- Correlation with the known transmit chirp boosts chirp-like echoes and suppresses
  unrelated noise, which helps estimate echo delay more reliably.

Why the direct pulse is ignored:
- The strongest correlation peak often appears near time 0 due to transmit leakage.
  If we keep that region, the model can over-focus on leakage instead of the wall echo.

Why fixed-length vectors are useful:
- Neural networks expect consistent input dimensions. Resampling each correlation
  curve to a fixed length (512) provides a stable feature shape for training.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, correlate


# Acoustic settings (must match dataset generation assumptions).
SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0

# Correlation processing settings.
DIRECT_IGNORE_S = 0.00055  # ignore/zero very early direct-transmit region (0.55 ms)
DIST_MIN_M = 0.05
DIST_MAX_M = 2.80
FEATURE_LENGTH = 512

DEFAULT_DATASET_ROOT = Path("datasets/synthetic_echoes_regression_hard_v1")


def make_transmit_chirp() -> np.ndarray:
    """Recreate the transmit chirp used in synthetic generation."""
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


def load_wav_float(wav_path: Path) -> np.ndarray:
    """Load WAV and convert to mono float32."""
    sr, data = wavfile.read(wav_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate in {wav_path}: {sr} (expected {SAMPLE_RATE})")

    if data.ndim > 1:
        data = data[:, 0]

    if np.issubdtype(data.dtype, np.integer):
        scale = float(np.iinfo(data.dtype).max)
        signal = data.astype(np.float32) / scale
    else:
        signal = data.astype(np.float32)
    return signal


def extract_correlation_feature(signal: np.ndarray, tx_chirp: np.ndarray, feature_length: int) -> np.ndarray:
    """
    Compute fixed-length correlation feature vector.

    Steps:
    1) matched-filter correlation with transmit chirp
    2) keep positive-lag window corresponding to [0.05 m, 2.80 m]
    3) zero direct region before 0.55 ms
    4) resample to fixed length (512)
    5) normalize safely
    """
    corr = correlate(signal, tx_chirp, mode="full", method="fft")
    corr_mag = np.abs(corr).astype(np.float32)

    lags = np.arange(-len(tx_chirp) + 1, len(signal), dtype=np.int64)
    lag_times_s = lags.astype(np.float64) / SAMPLE_RATE

    delay_min_s = 2.0 * DIST_MIN_M / SPEED_OF_SOUND
    delay_max_s = 2.0 * DIST_MAX_M / SPEED_OF_SOUND

    keep = (lag_times_s >= delay_min_s) & (lag_times_s <= delay_max_s)
    if not np.any(keep):
        raise ValueError("No lags inside requested distance-delay range.")

    kept_times = lag_times_s[keep]
    kept_corr = corr_mag[keep].copy()

    # Ignore direct-transmit leakage region.
    kept_corr[kept_times < DIRECT_IGNORE_S] = 0.0

    # Resample onto a fixed-length grid.
    uniform_times = np.linspace(delay_min_s, delay_max_s, feature_length, dtype=np.float64)
    feature = np.interp(uniform_times, kept_times, kept_corr).astype(np.float32)

    # Safe normalization (max-abs) to keep values bounded and comparable.
    max_abs = float(np.max(np.abs(feature)))
    if max_abs > 1e-12:
        feature = feature / max_abs
    else:
        feature = np.zeros_like(feature, dtype=np.float32)

    return feature


def main() -> None:
    parser = argparse.ArgumentParser(description="Add correlation features to hard_v1 dataset.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset root containing wav/ and labels.csv.",
    )
    parser.add_argument(
        "--feature-length",
        type=int,
        default=FEATURE_LENGTH,
        help="Length of output correlation feature vector.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    labels_csv = dataset_root / "labels.csv"
    wav_root = dataset_root / "wav"
    corr_root = dataset_root / "correlation_features"
    output_csv = dataset_root / "labels_with_correlation.csv"

    if not labels_csv.exists():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    if not wav_root.exists():
        raise FileNotFoundError(f"Missing WAV folder: {wav_root}")

    corr_root.mkdir(parents=True, exist_ok=True)

    tx_chirp = make_transmit_chirp()
    feature_length = int(args.feature_length)

    updated_rows: list[dict[str, str]] = []

    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_columns = reader.fieldnames or []
        if "filename" not in original_columns or "wav_path" not in original_columns:
            raise ValueError("labels.csv must contain at least 'filename' and 'wav_path' columns.")

        for i, row in enumerate(reader, start=1):
            filename = row["filename"]
            wav_path = dataset_root / row["wav_path"]
            if not wav_path.exists():
                raise FileNotFoundError(f"WAV file not found: {wav_path}")

            signal = load_wav_float(wav_path)
            feature = extract_correlation_feature(signal, tx_chirp, feature_length=feature_length)

            corr_rel = Path("correlation_features") / f"{filename}.npy"
            corr_abs = dataset_root / corr_rel
            np.save(corr_abs, feature)

            row["correlation_path"] = str(corr_rel).replace("\\", "/")
            updated_rows.append(row)

            if i % 500 == 0:
                print(f"Processed {i} samples...")

    output_columns = list(original_columns)
    if "correlation_path" not in output_columns:
        output_columns.append("correlation_path")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Done. Saved correlation features to: {corr_root}")
    print(f"Updated labels CSV: {output_csv}")


if __name__ == "__main__":
    main()
