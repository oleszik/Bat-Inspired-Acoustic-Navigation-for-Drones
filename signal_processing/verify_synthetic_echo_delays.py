"""
Numerically verify synthetic ultrasonic echo delays.

This script does NOT train a neural network. It only validates synthetic echo physics.

Why matched filtering:
- Matched filtering (correlation with the known transmit chirp) provides a robust and
  quantitative echo-arrival estimate, usually better than visually inspecting only a
  spectrogram.
- The direct transmit pulse near 0 ms must be ignored, otherwise the detector locks onto
  the transmitter leakage instead of the delayed wall echo.
- Delay should increase linearly with distance because time-of-flight is:
  delay = 2 * distance / speed_of_sound.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, correlate


SAMPLE_RATE = 192000
CHIRP_DURATION_S = 0.0005
CHIRP_F0_HZ = 35000.0
CHIRP_F1_HZ = 60000.0
SPEED_OF_SOUND = 343.0
SEARCH_START_MS = 0.8
PASS_THRESHOLD_MS = 0.3

WAV_ROOT = Path("datasets/synthetic_echoes/wav")
CSV_PATH = Path("datasets/synthetic_echoes/debug_comparison/echo_delay_verification.csv")

CLASS_TO_DISTANCE_M = {
    "no_obstacle": None,
    "wall_025cm": 0.25,
    "wall_050cm": 0.50,
    "wall_100cm": 1.00,
    "wall_150cm": 1.50,
    "wall_200cm": 2.00,
}


def make_transmit_chirp() -> np.ndarray:
    """Recreate the same transmit chirp used in synthetic generation."""
    n = int(CHIRP_DURATION_S * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    tx = chirp(t, f0=CHIRP_F0_HZ, t1=CHIRP_DURATION_S, f1=CHIRP_F1_HZ, method="linear")
    return tx.astype(np.float32)


def load_wav_float(path: Path) -> np.ndarray:
    """Load WAV and convert to mono float32 in [-1, 1] when needed."""
    sr, data = wavfile.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate in {path}: {sr} (expected {SAMPLE_RATE})")

    if data.ndim > 1:
        data = data[:, 0]

    if np.issubdtype(data.dtype, np.integer):
        max_abs = float(np.iinfo(data.dtype).max)
        signal = data.astype(np.float32) / max_abs
    else:
        signal = data.astype(np.float32)

    return signal


def detect_echo_delay_ms(signal: np.ndarray, tx_chirp: np.ndarray, search_start_ms: float) -> float:
    """
    Detect delayed echo with matched filtering.

    Returns the lag (ms) of the strongest correlation peak after search_start_ms.
    """
    corr = correlate(signal, tx_chirp, mode="full", method="fft")
    corr_abs = np.abs(corr)

    lags = np.arange(-len(tx_chirp) + 1, len(signal), dtype=np.int64)
    start_lag = int((search_start_ms / 1000.0) * SAMPLE_RATE)

    valid = lags >= start_lag
    if not np.any(valid):
        raise ValueError("No valid lag region after search start.")

    valid_corr = corr_abs[valid]
    valid_lags = lags[valid]

    peak_idx = int(np.argmax(valid_corr))
    peak_lag_samples = int(valid_lags[peak_idx])
    return (peak_lag_samples / SAMPLE_RATE) * 1000.0


def expected_delay_ms(distance_m: float | None) -> float | None:
    """Compute expected round-trip delay in milliseconds."""
    if distance_m is None:
        return None
    return (2.0 * distance_m / SPEED_OF_SOUND) * 1000.0


def fmt(x: float | None) -> str:
    if x is None or np.isnan(x):
        return "-"
    return f"{x:.4f}"


def main() -> None:
    tx_chirp = make_transmit_chirp()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float | int]] = []

    header = (
        f"{'class_name':<14} {'expected_delay_ms':>18} {'mean_detected_delay_ms':>24} "
        f"{'mean_error_ms':>14} {'max_error_ms':>12} {'number_of_files':>16} {'status':>10}"
    )
    print(header)
    print("-" * len(header))

    for class_name, distance_m in CLASS_TO_DISTANCE_M.items():
        class_dir = WAV_ROOT / class_name
        wav_files = sorted(class_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in {class_dir}")

        exp_ms = expected_delay_ms(distance_m)
        detected_ms_list: list[float] = []
        error_ms_list: list[float] = []

        for wav_path in wav_files:
            signal = load_wav_float(wav_path)
            det_ms = detect_echo_delay_ms(signal, tx_chirp, SEARCH_START_MS)
            detected_ms_list.append(det_ms)
            if exp_ms is not None:
                error_ms_list.append(abs(det_ms - exp_ms))

        mean_detected_ms = float(np.mean(detected_ms_list))
        if exp_ms is None:
            mean_error_ms = None
            max_error_ms = None
            status = "INFO"
            note = "No expected wall echo; strongest delayed peak reported for reference only."
        else:
            mean_error_ms = float(np.mean(error_ms_list))
            max_error_ms = float(np.max(error_ms_list))
            status = "PASS" if mean_error_ms < PASS_THRESHOLD_MS else "FAIL"
            note = ""

        print(
            f"{class_name:<14} {fmt(exp_ms):>18} {fmt(mean_detected_ms):>24} "
            f"{fmt(mean_error_ms):>14} {fmt(max_error_ms):>12} {len(wav_files):>16} {status:>10}"
        )

        rows.append(
            {
                "class_name": class_name,
                "expected_delay_ms": "" if exp_ms is None else f"{exp_ms:.6f}",
                "mean_detected_delay_ms": f"{mean_detected_ms:.6f}",
                "mean_error_ms": "" if mean_error_ms is None else f"{mean_error_ms:.6f}",
                "max_error_ms": "" if max_error_ms is None else f"{max_error_ms:.6f}",
                "number_of_files": len(wav_files),
                "status": status,
                "note": note,
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_name",
                "expected_delay_ms",
                "mean_detected_delay_ms",
                "mean_error_ms",
                "max_error_ms",
                "number_of_files",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV summary to: {CSV_PATH}")


if __name__ == "__main__":
    main()
