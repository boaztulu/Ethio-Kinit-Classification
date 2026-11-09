"""Wrappers around Hugging Face audio classification models.

Supports:
  - AST  (MIT/ast-finetuned-audioset-10-10-0.4593)
  - HuBERT  (facebook/hubert-base-ls960)
  - wav2vec2  (facebook/wav2vec2-base)

All take raw waveform float32 at 16 kHz, shape (B, T) or (B, 1, T).

We add a fresh classification head over the pooled features so we can replace
the original (often 527-class AudioSet) head with our 4-class head.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ASTClassifier(nn.Module):
    """Audio Spectrogram Transformer fine-tuned for 4-class classification."""
    def __init__(self,
                 model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 n_classes: int = 4,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        from transformers import ASTConfig, ASTModel, ASTFeatureExtractor

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.config = ASTConfig.from_pretrained(model_name)
        self.backbone = ASTModel.from_pretrained(model_name)
        hid = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hid, n_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _prep(self, wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        # wav: (B, T) or (B, 1, T)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        feats = self.feature_extractor(
            [w.cpu().numpy() for w in wav], sampling_rate=sr, return_tensors="pt"
        )["input_values"].to(wav.device)
        return feats

    def forward(self, wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        feats = self._prep(wav, sr)
        out = self.backbone(feats)
        # pooled output from CLS token
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]
        return self.head(self.dropout(pooled))


class HubertClassifier(nn.Module):
    def __init__(self,
                 model_name: str = "facebook/hubert-base-ls960",
                 n_classes: int = 4,
                 dropout: float = 0.1,
                 freeze_feature_extractor: bool = True):
        super().__init__()
        from transformers import HubertModel, HubertConfig
        self.config = HubertConfig.from_pretrained(model_name)
        self.backbone = HubertModel.from_pretrained(model_name)
        if freeze_feature_extractor:
            self.backbone.feature_extractor._freeze_parameters()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.config.hidden_size, n_classes)

    def forward(self, wav: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        out = self.backbone(wav, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, T', H)
        pooled = h.mean(dim=1)
        return self.head(self.dropout(pooled))


class Wav2Vec2Classifier(nn.Module):
    def __init__(self,
                 model_name: str = "facebook/wav2vec2-base",
                 n_classes: int = 4,
                 dropout: float = 0.1,
                 freeze_feature_extractor: bool = True):
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_feature_extractor:
            self.backbone.feature_extractor._freeze_parameters()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.config.hidden_size, n_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        out = self.backbone(wav)
        h = out.last_hidden_state
        pooled = h.mean(dim=1)
        return self.head(self.dropout(pooled))
