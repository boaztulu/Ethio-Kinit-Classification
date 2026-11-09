"""Reproduction of the EKM architecture from Retta et al. 2023, Section 5.2 / Fig 1.

Description verbatim:
  4 conv layers with output channels 32, 64, 128, 256.
  Kernels 3x3 for the first three conv layers, 2x2 for the last.
  4 MaxPool layers with pool size 3x3 for the first and 1x1 for the remaining.
  1 fully connected layer + softmax.

Inputs: MFCC features, 40 mel-bands -> 13 DCT coefs, 3 s clips at 16 kHz, hop 10 ms.
Shape (B, 1, n_mfcc=13, T) when input is MFCC; or (B, 1, n_mels, T) when input
is mel. The architecture is feature-agnostic; only the input channel is fixed at 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EKM(nn.Module):
    def __init__(self, n_classes: int = 4, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3, ceil_mode=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),

            nn.Conv2d(128, 256, kernel_size=2, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (B, 1, H, W)
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        return self.fc(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = EKM()
    x = torch.randn(2, 1, 13, 301)  # MFCC-like input
    print(m(x).shape, n_params(m))
