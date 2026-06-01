"""
Utility helpers for loading spectrogram images without torchvision.

This module provides an ImageFolder-like dataset and simple preprocessing:
- grayscale conversion
- resize to 128x128
- tensor conversion
- normalization to roughly [-1, 1] using mean=0.5, std=0.5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def preprocess_pil_image(image: Image.Image, image_size: int = 128) -> torch.Tensor:
    """
    Convert PIL image to normalized grayscale tensor [1, H, W].
    """
    image = image.convert("L").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
    tensor = (tensor - 0.5) / 0.5
    return tensor


class SimpleImageFolder(Dataset):
    """
    Lightweight ImageFolder equivalent.
    Folder structure:
      root/
        class_a/*.png
        class_b/*.png
        ...
    """

    def __init__(self, root: str | Path, image_size: int = 128) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        class_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not class_dirs:
            raise FileNotFoundError(f"No class folders found under: {self.root}")

        self.classes = [p.name for p in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.image_size = image_size

        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
            for file_path in files:
                self.samples.append((file_path, label))

        if not self.samples:
            raise FileNotFoundError(f"No supported image files found under: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path)
        tensor = preprocess_pil_image(image, image_size=self.image_size)
        return tensor, label


def make_splits(dataset: Dataset, seed: int = 42):
    """
    Split dataset into 70% train, 15% validation, 15% test.
    """
    total = len(dataset)
    train_len = int(0.70 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)
