"""
Build a side-by-side visual comparison of synthetic echo debug outputs.

This script does NOT train a neural network. It is only for visual verification.

Interpretation notes:
- In the correlation row, the delayed echo peak should move later in time as wall
  distance increases.
- Expected delays from physics (round-trip): 25 cm ~= 1.46 ms, 200 cm ~= 11.66 ms.
- The no_obstacle class should not show a strong delayed echo peak.
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


CLASS_ORDER = [
    "no_obstacle",
    "wall_025cm",
    "wall_050cm",
    "wall_100cm",
    "wall_150cm",
    "wall_200cm",
]

COLUMN_TITLES = [
    "No obstacle",
    "25 cm",
    "50 cm",
    "100 cm",
    "150 cm",
    "200 cm",
]

SPEC_ROOT = Path("datasets/synthetic_echoes/spectrograms_echo_focus")
CORR_ROOT = Path("datasets/synthetic_echoes/correlation_debug")
OUT_PATH = Path("datasets/synthetic_echoes/debug_comparison/synthetic_echo_distance_comparison.png")


def pick_one_png(folder: Path) -> Path:
    """Pick one sample image from a class folder."""
    png_files = sorted(folder.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {folder}")
    return png_files[0]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 6, figsize=(24, 8), dpi=140)

    for col, class_name in enumerate(CLASS_ORDER):
        spec_img_path = pick_one_png(SPEC_ROOT / class_name)
        corr_img_path = pick_one_png(CORR_ROOT / class_name)

        spec_img = mpimg.imread(spec_img_path)
        corr_img = mpimg.imread(corr_img_path)

        axes[0, col].imshow(spec_img)
        axes[1, col].imshow(corr_img)

        axes[0, col].set_title(COLUMN_TITLES[col], fontsize=12, pad=10)
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Echo-focused spectrogram", fontsize=12)
    axes[1, 0].set_ylabel("Correlation debug", fontsize=12)

    fig.suptitle("Synthetic Echo Distance Comparison", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PATH)
    plt.close(fig)

    print(f"Saved comparison figure: {OUT_PATH}")


if __name__ == "__main__":
    main()
