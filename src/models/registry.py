"""Model registry - string name -> (Module, input_mode)."""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn


def build_model(name: str, n_classes: int = 4, **kw) -> Tuple[nn.Module, str]:
    """Returns (model, expected_input_mode).

    expected_input_mode is one of: "raw", "mel", "mfcc".
    """
    name = name.lower()
    if name == "ekm_mfcc":
        from src.models.ekm import EKM
        return EKM(n_classes=n_classes), "mfcc"
    if name == "ekm_mel":
        from src.models.ekm import EKM
        return EKM(n_classes=n_classes), "mel"
    if name in {"efficientnet_b0", "effnet_b0", "effnet"}:
        from src.models.spec_cnn import TimmSpecCNN
        return TimmSpecCNN("efficientnet_b0", n_classes=n_classes,
                           pretrained=kw.get("pretrained", True)), "mel"
    if name in {"efficientnet_b2", "effnet_b2"}:
        from src.models.spec_cnn import TimmSpecCNN
        return TimmSpecCNN("efficientnet_b2", n_classes=n_classes,
                           pretrained=kw.get("pretrained", True)), "mel"
    if name == "resnet50_mel":
        from src.models.spec_cnn import TimmSpecCNN
        return TimmSpecCNN("resnet50", n_classes=n_classes,
                           pretrained=kw.get("pretrained", True)), "mel"
    if name == "ast":
        from src.models.hf_audio import ASTClassifier
        return ASTClassifier(n_classes=n_classes,
                             dropout=kw.get("dropout", 0.1),
                             freeze_backbone=kw.get("freeze_backbone", False)), "raw"
    if name == "hubert":
        from src.models.hf_audio import HubertClassifier
        return HubertClassifier(n_classes=n_classes,
                                dropout=kw.get("dropout", 0.1),
                                freeze_feature_extractor=kw.get("freeze_feature_extractor", True)), "raw"
    if name in {"wav2vec2", "wav2vec"}:
        from src.models.hf_audio import Wav2Vec2Classifier
        return Wav2Vec2Classifier(n_classes=n_classes,
                                  dropout=kw.get("dropout", 0.1),
                                  freeze_feature_extractor=kw.get("freeze_feature_extractor", True)), "raw"
    raise ValueError(f"unknown model: {name}")
