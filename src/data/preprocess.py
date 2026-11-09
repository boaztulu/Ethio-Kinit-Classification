"""Phase 2 - preprocessing & cache.

Per clip:
  1. load mono float32 @ 16 kHz (resample if needed)
  2. loudness-normalize to -23 LUFS (skip if clip too short for measurement)
  3. trim leading / trailing silence (energy threshold)
  4. chunk into `win_sec` windows with `overlap` fraction stride
For each chunk we store BOTH:
  - the raw waveform tensor (used by HuBERT / wav2vec / BEATs / AST that take raw audio)
  - a log-mel spectrogram (used by EfficientNet / EKM)

The cached layout:
  cache/raw/{class}/{stem}__{idx}.wav     (or .npy)
  cache/mel/{class}/{stem}__{idx}.npy     log-mel power-dB, shape (n_mels, T)
  cache/index.json                         list of {wav, mel, class, class_idx, source_path}

We don't extract MFCCs at cache time - those are computed on the fly only for
the EKM-reproduction baseline (cheap).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

from src.data.manifest import CLASS_NAMES, ClipMeta, load_manifest, save_manifest

TARGET_SR = 16_000


@dataclass
class ChunkRec:
    source_path: str
    chunk_idx: int
    class_name: str
    class_idx: int
    wav_path: str
    mel_path: str
    start_sec: float
    end_sec: float


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Best-effort loudness normalization. Falls back to peak-normalization if too short."""
    if len(y) < int(sr * 0.5):
        return y / (np.max(np.abs(y)) + 1e-9)
    try:
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(y)
        if np.isfinite(loud):
            return pyln.normalize.loudness(y, loud, target_lufs).astype(np.float32)
    except Exception:
        pass
    return y / (np.max(np.abs(y)) + 1e-9)


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    y_t, _ = librosa.effects.trim(y, top_db=top_db)
    return y_t


def chunk(y: np.ndarray, sr: int, win_sec: float = 3.0, overlap: float = 0.5
          ) -> List[np.ndarray]:
    win = int(round(win_sec * sr))
    hop = max(1, int(round(win * (1.0 - overlap))))
    out = []
    if len(y) < win:
        # pad to one window
        pad = np.zeros(win, dtype=y.dtype)
        pad[: len(y)] = y
        out.append(pad)
        return out
    for start in range(0, len(y) - win + 1, hop):
        out.append(y[start: start + win])
    # if there's a trailing tail with > 50% of a window, include a final chunk
    if (len(y) - (out[-1].shape[0]) ) % hop != 0:
        tail = y[-win:]
        if not np.array_equal(tail, out[-1]):
            out.append(tail.copy())
    return out


def logmel(y: np.ndarray, sr: int, n_mels: int = 128,
           win_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    n_fft = int(round(sr * win_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, power=2.0, fmin=20, fmax=sr // 2)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0).astype(np.float32)
    return S_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--win_sec", type=float, default=3.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--target_sr", type=int, default=TARGET_SR)
    ap.add_argument("--top_db", type=float, default=30.0)
    ap.add_argument("--n_mels", type=int, default=128)
    args = ap.parse_args()

    meta = load_manifest(Path(args.manifest))
    cache_dir = Path(args.cache_dir)
    raw_root = cache_dir / "raw"
    mel_root = cache_dir / "mel"
    for c in CLASS_NAMES:
        (raw_root / c).mkdir(parents=True, exist_ok=True)
        (mel_root / c).mkdir(parents=True, exist_ok=True)

    index: List[ChunkRec] = []
    skipped = 0

    for m in tqdm(meta, desc="preprocess"):
        try:
            y = load_audio(m.path, args.target_sr)
            y = loudness_normalize(y, args.target_sr)
            y = trim_silence(y, top_db=args.top_db)
            if len(y) < args.target_sr * 0.5:
                # less than 0.5 s of usable audio after silence-trim
                skipped += 1
                continue
            chunks = chunk(y, args.target_sr, win_sec=args.win_sec, overlap=args.overlap)
            stem = Path(m.path).stem
            for i, ch in enumerate(chunks):
                wav_p = raw_root / m.class_name / f"{stem}__{i:03d}.npy"
                mel_p = mel_root / m.class_name / f"{stem}__{i:03d}.npy"
                np.save(wav_p, ch.astype(np.float32))
                S_db = logmel(ch, args.target_sr, n_mels=args.n_mels)
                np.save(mel_p, S_db)
                index.append(ChunkRec(
                    source_path=m.path,
                    chunk_idx=i,
                    class_name=m.class_name,
                    class_idx=m.class_idx,
                    wav_path=str(wav_p),
                    mel_path=str(mel_p),
                    start_sec=i * args.win_sec * (1 - args.overlap),
                    end_sec=i * args.win_sec * (1 - args.overlap) + args.win_sec,
                ))
        except Exception as e:
            print(f"[WARN] skipping {m.path}: {e}")
            skipped += 1

    index_p = cache_dir / "index.json"
    with index_p.open("w") as f:
        json.dump([c.__dict__ for c in index], f, indent=2)
    print(f"[preprocess] wrote {len(index)} chunks from {len(meta) - skipped}/{len(meta)} clips")
    print(f"[preprocess] index: {index_p}")
    if skipped:
        print(f"[preprocess] skipped {skipped} clips")


if __name__ == "__main__":
    main()
