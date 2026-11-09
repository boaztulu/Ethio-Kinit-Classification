"""Build degraded copies of the 120 test recordings of the recording-level split.

Each degradation is applied to the whole preprocessed recording (16 kHz,
-23 LUFS, silence-trimmed), the result is re-normalised to -23 LUFS, and it is
then cut into the same 3-s / 50 % windows as the clean cache, so segment k of a
degraded recording is aligned with segment k of the clean one.

Conditions
  env10   environmental noise from ESC-50 (all 50 classes) at 10 dB SNR
  reverb  synthetic room response, RT60 = 1.0 s, direct-to-reverberant 0 dB
  mp3     MPEG-1/2 Layer III round trip at 32 kbit/s CBR (libsndfile/LAME)
  tel     telephone band: 8 kHz round trip plus a 300 Hz high-pass

Usage:
    python -m scripts.revision.degrade --esc50 <ESC-50-master dir> --out_dir <dir>
"""
from __future__ import annotations

import argparse
import io
import json
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as sps
import soundfile as sf

from scripts.revision.rebuild_cache import local_wav
from src.data.preprocess import (TARGET_SR, chunk, load_audio, logmel,
                                 loudness_normalize, trim_silence)

SR = TARGET_SR
CONDITIONS = ("env10", "reverb", "mp3", "tel")
_NOISE = None


def _load_noise_bank(esc50_dir: str):
    clips = []
    for p in sorted(Path(esc50_dir, "audio").glob("*.wav")):
        y, sr = sf.read(str(p), dtype="float32")
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        y, _ = librosa.effects.trim(y, top_db=40)
        if len(y) > SR // 2 and np.mean(y ** 2) > 1e-7:
            clips.append(y.astype(np.float32))
    return clips


def _init(esc50_dir):
    global _NOISE
    _NOISE = _load_noise_bank(esc50_dir)


def env_noise(y, rng, snr_db=10.0):
    parts, n = [], 0
    while n < len(y):
        c = _NOISE[rng.integers(len(_NOISE))]
        parts.append(c); n += len(c)
    noise = np.concatenate(parts)[: len(y)]
    p_sig, p_noise = np.mean(y ** 2), np.mean(noise ** 2)
    return y + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))


def reverb(y, rng, rt60=1.0, drr_db=0.0):
    n = int(rt60 * SR)
    t = np.arange(1, n) / SR
    tail = rng.standard_normal(n - 1) * np.exp(-6.9078 * t / rt60)
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail ** 2))
    h = np.concatenate([[1.0], tail])
    return sps.fftconvolve(y, h)[: len(y)]


def mp3(y, rng, level=0.85):
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), SR, format="MP3", subtype="MPEG_LAYER_III",
             compression_level=level, bitrate_mode="CONSTANT")
    kbps = buf.tell() * 8 / (len(y) / SR) / 1000
    buf.seek(0)
    z, _ = sf.read(buf, dtype="float32")
    # align the decoded signal to the input (encoder delay / padding)
    lag = int(np.argmax(sps.correlate(z[: SR * 2], y[: SR * 2], mode="full",
                                      method="fft")) - (SR * 2 - 1))
    z = z[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, np.float32), z])
    z = np.pad(z, (0, max(0, len(y) - len(z))))[: len(y)]
    return z, kbps, lag


def telephone(y, rng):
    z = librosa.resample(librosa.resample(y, orig_sr=SR, target_sr=8000),
                         orig_sr=8000, target_sr=SR)[: len(y)]
    sos = sps.butter(4, 300, btype="highpass", fs=SR, output="sos")
    return sps.sosfiltfilt(sos, z)


def process(args):
    k, m, cond, out_dir = args
    rng = np.random.default_rng(10_000 + k)
    y = load_audio(str(local_wav(m["path"])), SR)
    y = loudness_normalize(y, SR)
    y = trim_silence(y, top_db=30.0)
    info = {}
    if cond == "env10":
        z = env_noise(y, rng)
    elif cond == "reverb":
        z = reverb(y, rng)
    elif cond == "mp3":
        z, kbps, lag = mp3(y, rng)
        info = {"kbps": kbps, "lag": lag}
    elif cond == "tel":
        z = telephone(y, rng)
    else:
        raise ValueError(cond)
    z = loudness_normalize(z.astype(np.float32), SR)
    stem = Path(m["path"]).stem
    recs = []
    for i, ch in enumerate(chunk(z.astype(np.float32), SR, win_sec=3.0, overlap=0.5)):
        wav_p = out_dir / cond / "raw" / f"{stem}__{i:03d}.npy"
        mel_p = out_dir / cond / "mel" / f"{stem}__{i:03d}.npy"
        np.save(wav_p, ch.astype(np.float32))
        np.save(mel_p, logmel(ch, SR, n_mels=128))
        recs.append({"source_path": m["path"], "chunk_idx": i,
                     "class_name": m["class_name"], "class_idx": m["class_idx"],
                     "wav_path": str(wav_p), "mel_path": str(mel_p)})
    return recs, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    meta = {m["path"]: m for m in json.load(open("results/manifest.json"))}
    test = json.load(open("results/split_paper.json"))["test"]
    jobs_meta = [meta[p] for p in test]
    summary = {}
    for cond in args.conditions:
        (out_dir / cond / "raw").mkdir(parents=True, exist_ok=True)
        (out_dir / cond / "mel").mkdir(parents=True, exist_ok=True)
        init = dict(initializer=_init, initargs=(args.esc50,)) if cond == "env10" else {}
        with Pool(args.workers, **init) as pool:
            res = pool.map(process, [(k, m, cond, out_dir) for k, m in enumerate(jobs_meta)])
        index = [r for recs, _ in res for r in recs]
        json.dump(index, open(out_dir / cond / "index.json", "w"))
        infos = [i for _, i in res if i]
        summary[cond] = {"n_segments": len(index), "n_recordings": len(res)}
        if infos:
            summary[cond]["kbps_mean"] = float(np.mean([i["kbps"] for i in infos]))
            summary[cond]["lag_set"] = sorted({i["lag"] for i in infos})
        print(cond, summary[cond], flush=True)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
