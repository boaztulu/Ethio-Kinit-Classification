"""Audio + spectrogram augmentation for EMIR.

Two callable factories that the dataset can use:
  build_raw_augment(cfg)  -> fn(np.ndarray, sr) -> np.ndarray
  build_mel_augment(cfg)  -> fn(np.ndarray)     -> np.ndarray  (n_mels, T)

Mixup is applied at the batch level inside the training loop, not here.

All augmentations are deterministically toggleable from the YAML config so
that the ablation table in Phase 5 can be generated automatically.
"""
from __future__ import annotations

import numpy as np

try:
    import librosa
except ImportError:  # for type-check only
    librosa = None  # type: ignore


def gain_perturb(y: np.ndarray, db_range: float = 6.0) -> np.ndarray:
    g = 10 ** (np.random.uniform(-db_range, db_range) / 20.0)
    return (y * g).astype(np.float32)


def add_gaussian(y: np.ndarray, snr_db_range=(20, 35)) -> np.ndarray:
    snr_db = np.random.uniform(*snr_db_range)
    sig_p = float(np.mean(y ** 2) + 1e-12)
    noise_p = sig_p / (10 ** (snr_db / 10))
    n = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(noise_p).astype(np.float32)
    return (y + n).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, semitones_range=(-2.0, 2.0)) -> np.ndarray:
    n_steps = float(np.random.uniform(*semitones_range))
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, rate_range=(0.9, 1.1), target_len=None) -> np.ndarray:
    rate = float(np.random.uniform(*rate_range))
    out = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    if target_len is None:
        target_len = len(y)
    if len(out) < target_len:
        pad = np.zeros(target_len, dtype=np.float32)
        pad[: len(out)] = out
        out = pad
    else:
        out = out[:target_len]
    return out


def spec_augment(S: np.ndarray, n_time_masks: int = 2, time_mask_width: int = 20,
                 n_freq_masks: int = 2, freq_mask_width: int = 16) -> np.ndarray:
    S = S.copy()
    n_mels, T = S.shape
    fill = float(S.min())
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask_width + 1)
        if f == 0 or f >= n_mels:
            continue
        f0 = np.random.randint(0, n_mels - f)
        S[f0: f0 + f, :] = fill
    for _ in range(n_time_masks):
        t = np.random.randint(0, time_mask_width + 1)
        if t == 0 or t >= T:
            continue
        t0 = np.random.randint(0, T - t)
        S[:, t0: t0 + t] = fill
    return S


def build_raw_augment(cfg: dict):
    """cfg keys: gain, gaussian, pitch (bool), time_stretch (bool)."""
    def fn(y: np.ndarray, sr: int) -> np.ndarray:
        if cfg.get("gain", True) and np.random.rand() < 0.5:
            y = gain_perturb(y, db_range=cfg.get("gain_db", 6.0))
        if cfg.get("gaussian", True) and np.random.rand() < 0.5:
            y = add_gaussian(y, snr_db_range=cfg.get("snr_db_range", (20, 35)))
        if cfg.get("pitch", False) and np.random.rand() < 0.3:
            y = pitch_shift(y, sr, semitones_range=cfg.get("semitones_range", (-2.0, 2.0)))
        if cfg.get("time_stretch", True) and np.random.rand() < 0.3:
            y = time_stretch(y, rate_range=cfg.get("rate_range", (0.9, 1.1)),
                             target_len=len(y))
        return y
    return fn


def build_mel_augment(cfg: dict):
    def fn(S: np.ndarray) -> np.ndarray:
        if cfg.get("spec_aug", True):
            S = spec_augment(
                S,
                n_time_masks=cfg.get("n_time_masks", 2),
                time_mask_width=cfg.get("time_mask_width", 20),
                n_freq_masks=cfg.get("n_freq_masks", 2),
                freq_mask_width=cfg.get("freq_mask_width", 16),
            )
        return S
    return fn


def mixup_batch(x, y, alpha: float = 0.2):
    """Returns mixed_x, y_a, y_b, lam."""
    import torch
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam
