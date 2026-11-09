"""PyTorch dataset for EMIR cached chunks.

Two modes:
  mode="raw" -> returns waveform tensor (1, n_samples) for HuBERT/AST/BEATs
  mode="mel" -> returns log-mel spec (1, n_mels, T) for EfficientNet/EKM-mel
  mode="mfcc" -> on-the-fly MFCC from cached raw (for EKM reproduction)

Splits are loaded from a JSON produced by `src.data.splits.paper_split`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class EmirChunks(Dataset):
    def __init__(self,
                 index_path: str,
                 source_paths: List[str],
                 mode: str = "raw",
                 sr: int = 16000,
                 augment: Optional[Callable] = None,
                 mfcc_kwargs: Optional[Dict] = None,
                 raw_perturb: Optional[Callable] = None):
        """raw_perturb (y, sr) -> y: when mode=="mel" or "mfcc" and a raw-waveform
        perturbation should be applied BEFORE feature extraction (for robustness
        probing). Distinct from `augment` which acts on the extracted feature."""
        self.mode = mode
        self.sr = sr
        self.augment = augment
        self.raw_perturb = raw_perturb
        self.mfcc_kwargs = mfcc_kwargs or {"n_mfcc": 13, "n_mels": 40,
                                           "n_fft": 400, "hop_length": 160}

        with open(index_path) as f:
            full = json.load(f)
        # `source_paths` may be either clip-level WAV paths (paper_split) or
        # chunk-level NPY paths (chunk_level_split). Match against both.
        s = set(source_paths)
        self.items = [x for x in full
                      if x["source_path"] in s or x["wav_path"] in s]
        if not self.items:
            raise RuntimeError("EmirChunks: empty after filtering by source_paths")

    def __len__(self):
        return len(self.items)

    def _load_raw(self, p: str) -> np.ndarray:
        return np.load(p).astype(np.float32)

    def _load_mel(self, p: str) -> np.ndarray:
        return np.load(p).astype(np.float32)

    def _mfcc(self, y: np.ndarray) -> np.ndarray:
        k = self.mfcc_kwargs
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr,
            n_mfcc=k["n_mfcc"], n_mels=k["n_mels"],
            n_fft=k["n_fft"], hop_length=k["hop_length"],
        )
        return mfcc.astype(np.float32)

    def __getitem__(self, idx):
        rec = self.items[idx]
        y_idx = int(rec["class_idx"])
        if self.mode == "raw":
            y = self._load_raw(rec["wav_path"])
            if self.raw_perturb is not None:
                y = self.raw_perturb(y, self.sr)
            if self.augment is not None:
                y = self.augment(y, self.sr)
            return torch.from_numpy(y).unsqueeze(0), y_idx, rec["source_path"]
        elif self.mode == "mel":
            if self.raw_perturb is not None:
                # robustness: perturb raw, then recompute mel
                y = self._load_raw(rec["wav_path"])
                y = self.raw_perturb(y, self.sr)
                S = self._logmel(y)
            else:
                S = self._load_mel(rec["mel_path"])
            if self.augment is not None:
                S = self.augment(S)
            S = (S - S.min()) / (S.max() - S.min() + 1e-6)
            return torch.from_numpy(S).unsqueeze(0), y_idx, rec["source_path"]
        elif self.mode == "mfcc":
            y = self._load_raw(rec["wav_path"])
            if self.raw_perturb is not None:
                y = self.raw_perturb(y, self.sr)
            if self.augment is not None:
                y = self.augment(y, self.sr)
            M = self._mfcc(y)
            M = (M - M.mean()) / (M.std() + 1e-6)
            return torch.from_numpy(M).unsqueeze(0), y_idx, rec["source_path"]
        else:
            raise ValueError(f"unknown mode: {self.mode}")

    def _logmel(self, y: np.ndarray, n_mels: int = 128,
                win_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
        n_fft = int(round(self.sr * win_ms / 1000.0))
        hop = int(round(self.sr * hop_ms / 1000.0))
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=n_fft, hop_length=hop,
                                           n_mels=n_mels, power=2.0, fmin=20,
                                           fmax=self.sr // 2)
        return librosa.power_to_db(S, ref=np.max, top_db=80.0).astype(np.float32)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool,
                num_workers: int = 4, drop_last: bool = False,
                weighted_sampler: Optional[torch.utils.data.Sampler] = None
                ) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle and weighted_sampler is None,
        sampler=weighted_sampler, num_workers=num_workers, drop_last=drop_last,
        pin_memory=True,
    )


def class_balanced_sampler(ds: EmirChunks, class_idx: List[int]
                           ) -> torch.utils.data.WeightedRandomSampler:
    n_per_class = np.bincount([int(it["class_idx"]) for it in ds.items], minlength=max(class_idx) + 1)
    inv = 1.0 / np.maximum(n_per_class, 1)
    weights = np.array([inv[int(it["class_idx"])] for it in ds.items], dtype=np.float64)
    return torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True)
