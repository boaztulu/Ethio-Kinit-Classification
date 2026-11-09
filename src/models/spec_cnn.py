"""Mel-spectrogram CNN baselines built on top of timm.

Wraps timm models (EfficientNet, ResNet, etc.) to accept (B, 1, n_mels, T)
log-mel-spectrogram inputs by repeating the single channel into the model's
expected number of input channels (typically 3) and rescaling.

These are the "CNN on spectrogram" family that compete with EKM.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class TimmSpecCNN(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", n_classes: int = 4,
                 pretrained: bool = True, drop_path_rate: float = 0.1):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for TimmSpecCNN")
        self.net = timm.create_model(
            backbone, pretrained=pretrained, num_classes=n_classes,
            in_chans=1, drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.net(x)
