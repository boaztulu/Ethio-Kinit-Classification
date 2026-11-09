"""Rebuild the segment cache on a local machine from the EMIR WAVs.

Uses exactly the Phase-2 functions in src/data/preprocess.py (resample to
16 kHz, -23 LUFS, 30 dB silence trim, 3-s windows with 50 % overlap, 128-band
log-mel). The index keeps the `source_path` strings of the manifest so the
split files (results/split_*.json) select the same recordings, while
`wav_path` / `mel_path` point at the local cache.

Usage:
    python -m scripts.revision.rebuild_cache --out_dir <cache dir> [--workers 20]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.data.preprocess import (TARGET_SR, chunk, load_audio, logmel,
                                 loudness_normalize, trim_silence)

HPG_ROOT = ""
LOCAL_ROOT = Path(__file__).resolve().parents[2]


def local_wav(hpg_path: str) -> Path:
    return LOCAL_ROOT / hpg_path[len(HPG_ROOT):]


def process(args):
    m, out_dir = args
    y = load_audio(str(local_wav(m["path"])), TARGET_SR)
    y = loudness_normalize(y, TARGET_SR)
    y = trim_silence(y, top_db=30.0)
    stem = Path(m["path"]).stem
    recs = []
    for i, ch in enumerate(chunk(y, TARGET_SR, win_sec=3.0, overlap=0.5)):
        wav_p = out_dir / "raw" / m["class_name"] / f"{stem}__{i:03d}.npy"
        mel_p = out_dir / "mel" / m["class_name"] / f"{stem}__{i:03d}.npy"
        np.save(wav_p, ch.astype(np.float32))
        np.save(mel_p, logmel(ch, TARGET_SR, n_mels=128))
        recs.append({
            "source_path": m["path"], "chunk_idx": i,
            "class_name": m["class_name"], "class_idx": m["class_idx"],
            "wav_path": str(wav_p), "mel_path": str(mel_p),
            "start_sec": i * 1.5, "end_sec": i * 1.5 + 3.0,
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", default="results/manifest.json")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meta = json.load(open(args.manifest))
    for c in ("Tizita", "Bati", "Ambassel", "Anchihoye"):
        (out_dir / "raw" / c).mkdir(parents=True, exist_ok=True)
        (out_dir / "mel" / c).mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        parts = pool.map(process, [(m, out_dir) for m in meta], chunksize=4)
    index = [r for part in parts for r in part]
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)
    print(f"[rebuild_cache] {len(index)} segments from {len(meta)} recordings -> {out_dir}")


if __name__ == "__main__":
    main()
