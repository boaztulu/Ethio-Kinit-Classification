"""Per-segment perturbation caches for the 120 test recordings.

Reproduces the robustness protocol of src/eval.py (white Gaussian noise at a
segment-level SNR; librosa phase-vocoder pitch shift of each 3-s segment), but
with a fixed random seed per segment and with the perturbed segments stored,
so that several models and both log-mel scalings see identical inputs.

Usage:
    python -m scripts.revision.perturb_cache --out_dir <dir>
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from scripts.revision.common import RESULTS
from src.data.preprocess import TARGET_SR, logmel

CONDITIONS = {"noise20": ("noise", 20), "noise10": ("noise", 10), "noise5": ("noise", 5),
              "pitch-2": ("pitch", -2), "pitch+2": ("pitch", 2)}


def perturb(args):
    k, rec, cond, out_dir = args
    kind, val = CONDITIONS[cond]
    y = np.load(rec["wav_path"]).astype(np.float32)
    if kind == "noise":
        rng = np.random.default_rng(20_000 + k)
        p = float(np.mean(y ** 2) + 1e-12) / (10 ** (val / 10))
        z = y + rng.standard_normal(len(y)).astype(np.float32) * np.float32(np.sqrt(p))
    else:
        import librosa
        z = librosa.effects.pitch_shift(y, sr=TARGET_SR, n_steps=val)
    z = z.astype(np.float32)
    stem = Path(rec["wav_path"]).stem
    wav_p = out_dir / cond / "raw" / f"{stem}.npy"
    mel_p = out_dir / cond / "mel" / f"{stem}.npy"
    np.save(wav_p, z)
    np.save(mel_p, logmel(z, TARGET_SR, n_mels=128))
    return dict(rec, wav_path=str(wav_p), mel_path=str(mel_p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="clean local cache index.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    test = set(json.load(open(RESULTS / "split_paper.json"))["test"])
    recs = [r for r in json.load(open(args.index)) if r["source_path"] in test]
    with Pool(args.workers) as pool:
        for cond in CONDITIONS:
            (out_dir / cond / "raw").mkdir(parents=True, exist_ok=True)
            (out_dir / cond / "mel").mkdir(parents=True, exist_ok=True)
            index = pool.map(perturb, [(k, r, cond, out_dir) for k, r in enumerate(recs)], chunksize=16)
            json.dump(index, open(out_dir / cond / "index.json", "w"))
            print(cond, len(index), flush=True)


if __name__ == "__main__":
    main()
