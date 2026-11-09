"""Search EMIR for recordings that share audio, by landmark fingerprinting.

Implements the spectral-peak-pair ("landmark") fingerprint of Wang (2003):
spectrogram peaks are paired within a target zone, each pair is hashed as
(f1, f2, dt), and two recordings share audio when many hashes agree at one
consistent time offset. The detector is validated on positive controls, in
which 40 recordings are cropped to 12 s at a random offset, MP3-compressed at
32 kbit/s and mixed with white noise at 20 dB SNR, and on the score
distribution of all unrelated pairs.

Writes results/revision/fingerprint.json.
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter

from scripts.revision.common import RESULTS
from scripts.revision.rebuild_cache import local_wav
from src.data.preprocess import TARGET_SR, load_audio, loudness_normalize, trim_silence

SR = 8000
N_FFT, HOP = 512, 128            # 64 ms window, 16 ms hop
FAN_OUT, DT_MAX, DF_MAX = 5, 63, 63
PEAKS_PER_S = 20


def to8k(y):
    import librosa
    return librosa.resample(y, orig_sr=TARGET_SR, target_sr=SR)


def landmarks(y):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S = librosa.amplitude_to_db(S, ref=np.max)
    loc = (S == maximum_filter(S, size=(15, 15))) & (S > -60)
    f, t = np.nonzero(loc)
    keep = int(PEAKS_PER_S * len(y) / SR)
    order = np.argsort(-S[f, t])[:keep]
    f, t = f[order], t[order]
    o = np.argsort(t); f, t = f[o], t[o]
    hashes, times = [], []
    for i in range(len(t)):
        n = 0
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            if dt > DT_MAX:
                break
            if dt < 1 or abs(int(f[j]) - int(f[i])) > DF_MAX:
                continue
            hashes.append((int(f[i]) << 16) | (int(f[j]) << 6) | int(dt))
            times.append(int(t[i]))
            n += 1
            if n == FAN_OUT:
                break
    return np.array(hashes, np.int64), np.array(times, np.int32)


def prep(path):
    y = trim_silence(loudness_normalize(load_audio(str(local_wav(path)), TARGET_SR), TARGET_SR), 30.0)
    return to8k(y)


def fp_recording(path):
    return path, landmarks(prep(path))


def fp_control(args):
    k, path = args
    rng = np.random.default_rng(500 + k)
    y = prep(path)
    start = int(rng.integers(0, len(y) - 12 * SR))
    z = y[start: start + 12 * SR].astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.clip(z, -1, 1), SR, format="MP3", subtype="MPEG_LAYER_III",
             compression_level=0.85, bitrate_mode="CONSTANT")
    buf.seek(0)
    z, _ = sf.read(buf, dtype="float32")
    z = z + rng.standard_normal(len(z)).astype(np.float32) * np.sqrt(np.mean(z ** 2) / 100)
    return path, landmarks(z)


def match_scores(db, queries=None, max_bucket=50):
    """Best consistent-offset hash count for every (query, reference) pair."""
    index = defaultdict(list)
    for r, (h, t) in enumerate(db):
        for hh, tt in zip(h.tolist(), t.tolist()):
            index[hh].append((r, tt))
    scores = defaultdict(int)
    qs = db if queries is None else queries
    for q, (h, t) in enumerate(qs):
        counts = defaultdict(int)
        for hh, tt in zip(h.tolist(), t.tolist()):
            bucket = index.get(hh)
            if not bucket or len(bucket) > max_bucket:
                continue
            for r, tr in bucket:
                if queries is None and r <= q:
                    continue
                counts[(r, tr - tt)] += 1
        for (r, off), c in counts.items():
            if c > scores[(q, r)]:
                scores[(q, r)] = c
    return scores


def main():
    man = json.load(open(RESULTS / "manifest.json"))
    paths = [m["path"] for m in man]
    cls = {m["path"]: m["class_name"] for m in man}
    sp = json.load(open(RESULTS / "split_paper.json"))
    part = {p: n for n in ("train", "val", "test") for p in sp[n]}

    with Pool(12) as pool:
        db = dict(pool.map(fp_recording, paths, chunksize=5))
        rng = np.random.default_rng(1)
        ctrl_paths = [paths[i] for i in rng.choice(len(paths), 40, replace=False)]
        ctrl = dict(pool.map(fp_control, list(enumerate(ctrl_paths))))
    dbl = [db[p] for p in paths]

    # all unrelated / related pairs among the 600 recordings
    pair = match_scores(dbl)
    vals = np.array(list(pair.values()) or [0])
    # positive controls: each degraded excerpt queried against the full corpus
    cs = match_scores(dbl, [ctrl[p] for p in ctrl_paths])
    pos, neg_best = [], []
    for q, p in enumerate(ctrl_paths):
        own = paths.index(p)
        pos.append(cs.get((q, own), 0))
        neg_best.append(max([v for (qq, r), v in cs.items() if qq == q and r != own] or [0]))

    # byte-identical files
    import hashlib
    md5 = {p: hashlib.md5(open(local_wav(p), "rb").read()).hexdigest() for p in paths}
    md5_groups = defaultdict(list)
    for p, h in md5.items():
        md5_groups[h].append(p.split("/")[-1])

    edges = [0, 5, 10, 20, 50, 100, 500, 1000, 10 ** 6]
    hist = np.histogram(vals, bins=edges)[0].tolist()
    res = {
        "n_landmarks_mean": float(np.mean([len(h) for h, _ in dbl])),
        "control_positive_scores": pos,
        "control_best_other": neg_best,
        "pair_score_hist": {"edges": edges, "counts": hist},
        "md5_duplicate_groups": [g for g in md5_groups.values() if len(g) > 1],
        "pairs_ge5": [
            {"score": int(v), "a": paths[a].split("/")[-1], "b": paths[b].split("/")[-1],
             "class_a": cls[paths[a]], "class_b": cls[paths[b]],
             "split_a": part[paths[a]], "split_b": part[paths[b]]}
            for (a, b), v in sorted(pair.items(), key=lambda kv: -kv[1]) if v >= 5],
    }
    (RESULTS / "revision").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(RESULTS / "revision" / "fingerprint.json", "w"), indent=1)
    print("control positive scores: min", min(pos), "median", int(np.median(pos)))
    print("pair score histogram", list(zip(edges[:-1], hist)))
    print("md5 duplicate groups:", len(res["md5_duplicate_groups"]),
          "files in them:", sum(len(g) for g in res["md5_duplicate_groups"]))


if __name__ == "__main__":
    main()
