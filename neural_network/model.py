"""
Simple CNN model for classifying echo-focused spectrograms.
"""

import torch
import torch.nn as nn


class AcousticCNN(nn.Module):
    """
    Input: grayscale spectrogram image, shape [B, 1, 128, 128]
    Output: logits for 6 classes
    """

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()

        # Conv2D -> ReLU -> MaxPool
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # For 128x128 input: after 3 pool layers -> 16x16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
