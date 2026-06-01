"""
Generate synthetic ultrasonic echo data for bat-inspired drone navigation.

This script is only for signal generation and debugging plots.
It does NOT include neural-network training.

Why this debug version helps:
- Long chirps can overlap with close echoes. If the transmit pulse is long, the echo
  from a nearby wall arrives while the direct pulse is still active, so both signatures
  blur together in the spectrogram.
- A shorter chirp separates events in time. Here we use a 0.5 ms pulse so early echoes
  are easier to distinguish from the transmit pulse.
- Echo delay follows round-trip distance: echo_delay = 2 * distance / 343.0.
  For example, 0.25 m -> about 1.46 ms, while 2.00 m -> about 11.66 ms.
- Fixed spectrogram color scaling (vmin=-130, vmax=-50 dB) keeps visual comparison fair.
  Without fixed limits, auto-scaling can make weak and strong echoes look too similar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp, correlate, spectrogram


# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 192000  # Hz
CHIRP_F0 = 35000.0  # Hz
CHIRP_F1 = 60000.0  # Hz
CHIRP_DURATION = 0.0005  # seconds (short pulse for clearer echo separation)
RECORD_DURATION = 0.03  # seconds (long enough for up to 2 m echoes)
SPEED_OF_SOUND = 343.0  # m/s
SAMPLES_PER_CLASS = 500
REFLECTION_STRENGTH = 0.8  # fixed temporary value for easier distance debugging
ECHO_FOCUS_START_S = 0.0008  # remove first 0.8 ms in echo-focus plots
SPEC_VMIN_DB = -130
SPEC_VMAX_DB = -50

RNG = np.random.default_rng(seed=42)

CLASS_TO_DISTANCE_M = {
    "no_obstacle": None,
    "wall_025cm": 0.25,
    "wall_050cm": 0.50,
    "wall_100cm": 1.00,
    "wall_150cm": 1.50,
    "wall_200cm": 2.00,
}

WAV_ROOT = Path("datasets/synthetic_echoes/wav")
SPEC_FULL_ROOT = Path("datasets/synthetic_echoes/spectrograms_full")
SPEC_ECHO_FOCUS_ROOT = Path("datasets/synthetic_echoes/spectrograms_echo_focus")
CORR_ROOT = Path("datasets/synthetic_echoes/correlation_debug")


def make_chirp(duration_s: float, f0_hz: float, f1_hz: float) -> np.ndarray:
    """Create a linear ultrasonic chirp."""
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    return chirp(t, f0=f0_hz, t1=duration_s, f1=f1_hz, method="linear").astype(np.float32)


def place_signal(target: np.ndarray, signal: np.ndarray, start_idx: int, gain: float) -> None:
    """Add `signal` into `target` starting at `start_idx` with scaling `gain`."""
    if start_idx >= len(target):
        return
    end_idx = min(start_idx + len(signal), len(target))
    chunk_len = end_idx - start_idx
    if chunk_len > 0:
        target[start_idx:end_idx] += gain * signal[:chunk_len]


def expected_echo_delay(distance_m: float | None) -> float | None:
    """Compute expected echo delay from round-trip time-of-flight."""
    if distance_m is None:
        return None
    return 2.0 * distance_m / SPEED_OF_SOUND


def simulate_echo_waveform(distance_m: float | None, tx_chirp: np.ndarray) -> tuple[np.ndarray, float | None]:
    """
    Build one synthetic recording.

    Physics:
    - Echo delay is round-trip time-of-flight:
      echo_delay = 2 * distance_m / speed_of_sound
    - Close wall (0.25 m) echo appears early (~1.46 ms).
    - Far wall (2.00 m) echo appears much later (~11.66 ms).
    """
    n = int(RECORD_DURATION * SAMPLE_RATE)
    waveform = np.zeros(n, dtype=np.float32)

    # Direct transmit leakage into receiver at time 0.
    tx_gain = 1.0 + RNG.uniform(-0.03, 0.03)
    place_signal(waveform, tx_chirp, start_idx=0, gain=tx_gain)

    # Echo from wall, if present.
    echo_delay = expected_echo_delay(distance_m)
    if distance_m is not None:
        # Keep reflection strength fixed temporarily for clearer class comparisons.
        echo_start_idx = int(echo_delay * SAMPLE_RATE)
        echo_gain = REFLECTION_STRENGTH * (1.0 + RNG.uniform(-0.03, 0.03))
        place_signal(waveform, tx_chirp, start_idx=echo_start_idx, gain=echo_gain)

    # Add low noise so echoes stay visible in debug plots.
    noise_std = RNG.uniform(0.0015, 0.0035)
    waveform += RNG.normal(0.0, noise_std, size=n).astype(np.float32)

    waveform *= 1.0 + RNG.uniform(-0.03, 0.03)

    # Keep waveform in [-1, 1] for stable int16 conversion.
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = np.clip(waveform / peak * 0.95, -1.0, 1.0)

    return waveform, echo_delay


def save_wav(path: Path, waveform: np.ndarray) -> None:
    """Write normalized float waveform as 16-bit PCM WAV."""
    pcm = np.int16(waveform * 32767)
    wavfile.write(path, SAMPLE_RATE, pcm)


def save_spectrogram(
    path: Path,
    waveform: np.ndarray,
    expected_delay_s: float | None,
    title: str,
    echo_focus: bool,
) -> None:
    """
    Compute and save a spectrogram image.

    Signal-processing:
    - STFT-style spectrogram (windowed FFT over time)
    - PSD converted to dB
    - Fixed dB range (vmin/vmax) so comparisons between samples are honest
    """
    freqs, times, psd = spectrogram(
        waveform,
        fs=SAMPLE_RATE,
        nperseg=256,
        noverlap=192,
        scaling="density",
        mode="psd",
    )
    psd_db = 10.0 * np.log10(psd + 1e-20)

    if echo_focus:
        # Remove the first 0.8 ms to de-emphasize direct transmit leakage and
        # make delayed echoes easier to inspect.
        keep = times >= ECHO_FOCUS_START_S
        times_plot = times[keep]
        spec_plot = psd_db[:, keep]
    else:
        times_plot = times
        spec_plot = psd_db

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    mesh = ax.pcolormesh(
        times_plot,
        freqs,
        spec_plot,
        shading="gouraud",
        cmap="magma",
        vmin=SPEC_VMIN_DB,
        vmax=SPEC_VMAX_DB,
    )
    ax.set_ylim(20000, 80000)  # focus on ultrasonic band
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title(title)
    if expected_delay_s is not None:
        ax.axvline(expected_delay_s, linestyle="--", linewidth=1.2, color="cyan", label="Expected echo")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(mesh, ax=ax, label="PSD [dB]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_correlation_debug_plot(
    path: Path, waveform: np.ndarray, tx_chirp: np.ndarray, expected_delay_s: float | None
) -> None:
    """
    Save matched-filter style debug plot.

    Correlation with the transmit chirp is a simple way to detect pulse arrivals.
    Peaks typically appear near:
    - 0 ms (direct chirp leakage)
    - expected echo delay for wall classes
    """
    corr = correlate(waveform, tx_chirp, mode="full", method="fft")
    corr_mag = np.abs(corr)
    if np.max(corr_mag) > 0:
        corr_mag = corr_mag / np.max(corr_mag)

    lags = np.arange(-len(tx_chirp) + 1, len(waveform), dtype=np.float32) / SAMPLE_RATE
    keep = lags >= 0.0
    lags_s = lags[keep]
    corr_mag = corr_mag[keep]

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    ax.plot(lags_s * 1000.0, corr_mag, color="tab:blue", linewidth=1.0)
    if expected_delay_s is not None:
        ax.axvline(
            expected_delay_s * 1000.0,
            linestyle="--",
            linewidth=1.2,
            color="tab:red",
            label=f"Expected echo {expected_delay_s * 1000.0:.2f} ms",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0.0, RECORD_DURATION * 1000.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Normalized correlation")
    ax.set_title("Matched-Filter / Correlation Debug")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def clear_directory_files(path: Path, pattern: str) -> None:
    """Remove old generated files so debug sample counts stay exact."""
    if not path.exists():
        return
    for file_path in path.glob(pattern):
        file_path.unlink()


def main() -> None:
    tx_chirp = make_chirp(CHIRP_DURATION, CHIRP_F0, CHIRP_F1)

    for class_name, distance_m in CLASS_TO_DISTANCE_M.items():
        wav_dir = WAV_ROOT / class_name
        full_dir = SPEC_FULL_ROOT / class_name
        echo_focus_dir = SPEC_ECHO_FOCUS_ROOT / class_name
        corr_dir = CORR_ROOT / class_name

        wav_dir.mkdir(parents=True, exist_ok=True)
        full_dir.mkdir(parents=True, exist_ok=True)
        echo_focus_dir.mkdir(parents=True, exist_ok=True)
        corr_dir.mkdir(parents=True, exist_ok=True)

        clear_directory_files(wav_dir, "*.wav")
        clear_directory_files(full_dir, "*.png")
        clear_directory_files(echo_focus_dir, "*.png")
        clear_directory_files(corr_dir, "*.png")

        for i in range(SAMPLES_PER_CLASS):
            waveform, expected_delay_s = simulate_echo_waveform(distance_m, tx_chirp)
            file_id = f"{class_name}_{i:04d}"

            save_wav(wav_dir / f"{file_id}.wav", waveform)
            save_spectrogram(
                path=full_dir / f"{file_id}.png",
                waveform=waveform,
                expected_delay_s=expected_delay_s,
                title="Full Spectrogram (Direct + Echo)",
                echo_focus=False,
            )
            save_spectrogram(
                path=echo_focus_dir / f"{file_id}.png",
                waveform=waveform,
                expected_delay_s=expected_delay_s,
                title="Echo-Focused Spectrogram (First 0.8 ms Removed)",
                echo_focus=True,
            )
            save_correlation_debug_plot(
                path=corr_dir / f"{file_id}.png",
                waveform=waveform,
                tx_chirp=tx_chirp,
                expected_delay_s=expected_delay_s,
            )

        print(f"Generated {SAMPLES_PER_CLASS} samples for class: {class_name}")

    print("Done. Debug synthetic echoes and plots are saved under datasets/synthetic_echoes/")


if __name__ == "__main__":
    main()
