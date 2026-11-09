"""Absolute-pitch analysis with pitch-class profiles (no deep model involved).

For every recording a 36-bin (one-third-semitone) pitch-class profile (PCP)
is computed from a constant-Q chromagram of the preprocessed 16-kHz signal,
with tuning fixed at A4 = 440 Hz so that absolute pitch is preserved. Three
logistic-regression classifiers are compared on the same partitions as the
deep models:

  absolute     the PCP as measured
  tonic-norm   every PCP rotated so that its dominant semitone class is first
  dft-mag      magnitude of the discrete Fourier transform of the PCP, which
               is exactly invariant to circular shifts (transposition)

They are evaluated on the clean test recordings and on the same recordings
pitch-shifted by -2, -1, +1 and +2 semitones (librosa phase vocoder, applied
to the whole recording). The dominant semitone class of each recording is also
reported per Kinit as a crude tonic proxy.

Writes results/revision/pitch_profiles.json.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.revision.common import CLASSES, RESULTS
from scripts.revision.rebuild_cache import local_wav
from src.data.preprocess import TARGET_SR, load_audio, loudness_normalize, trim_silence

BPO = 36                      # bins per octave
SHIFTS = (-2, -1, 1, 2)
OUT = RESULTS / "revision" / "pitch_profiles.json"


def pcp(y):
    C = librosa.feature.chroma_cqt(y=y, sr=TARGET_SR, hop_length=512, n_chroma=BPO,
                                   bins_per_octave=BPO, n_octaves=6, tuning=0.0,
                                   fmin=librosa.note_to_hz("C2"))
    v = C.mean(1)
    return v / (v.sum() + 1e-12)


def features(args):
    path, shifts = args
    y = trim_silence(loudness_normalize(load_audio(str(local_wav(path)), TARGET_SR), TARGET_SR), 30.0)
    out = {0: pcp(y)}
    for s in shifts:
        out[s] = pcp(librosa.effects.pitch_shift(y, sr=TARGET_SR, n_steps=s))
    return path, out


def semitone_class(v):
    """Fold the 36 bins onto 12 semitone classes (bin 3k is centred on k)."""
    return np.array([v[(3 * k - 1) % BPO] + v[3 * k] + v[3 * k + 1] for k in range(12)])


def rotate(v, semitones):
    return np.roll(v, 3 * semitones)


def tonic_norm(v):
    return rotate(v, -int(np.argmax(semitone_class(v))))


def dft_mag(v):
    return np.abs(np.fft.rfft(v))[1:]


def fit(X, y):
    """Logistic regression with C chosen by a repeated (4 x 5-fold) inner CV
    on the training recordings only."""
    clf = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                       {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
                       cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0))
    return clf.fit(X, y)


def main():
    man = {m["path"]: m for m in json.load(open(RESULTS / "manifest.json"))}
    sp = json.load(open(RESULTS / "split_paper.json"))
    test = set(sp["test"])
    jobs = [(p, SHIFTS if p in test else ()) for p in man]
    with Pool(12) as pool:
        feats = dict(pool.map(features, jobs, chunksize=4))

    lab = {p: man[p]["class_idx"] for p in man}
    res = {}

    # tonic proxy: dominant semitone class per Kinit (0 = C, ..., 11 = B)
    dom = {c: np.zeros(12, int) for c in CLASSES}
    for p, f in feats.items():
        dom[man[p]["class_name"]][int(np.argmax(semitone_class(f[0])))] += 1
    res["dominant_class_hist"] = {c: v.tolist() for c, v in dom.items()}
    res["dominant_top2_share"] = {c: float(np.sort(v)[-2:].sum() / v.sum()) for c, v in dom.items()}

    variants = {"absolute": lambda v: v, "tonic-norm": tonic_norm, "dft-mag": dft_mag}

    # duplicate-free subsets defined in results/revision/{duplicates,stats}.json
    clean_test = set(json.load(open(RESULTS / "revision" / "duplicates.json"))["thr10"]["_clean_test"])
    clean_cv = set(json.load(open(RESULTS / "revision" / "stats.json"))["cv_clean_keys"])

    # recording-level split: fit on train+val, test on the 120 test recordings
    tr = sp["train"] + sp["val"]
    split_res = {}
    for name, fx in variants.items():
        clf = fit(np.stack([fx(feats[p][0]) for p in tr]), np.array([lab[p] for p in tr]))
        r = {}
        for s in (0,) + SHIFTS:
            Xt = np.stack([fx(feats[p][s]) for p in sp["test"]])
            pred = clf.predict(Xt)
            y = np.array([lab[p] for p in sp["test"]])
            ok = pred == y
            r[str(s)] = float(np.mean(ok) * 100)
            m = np.array([p in clean_test for p in sp["test"]])
            r[f"{s}_clean"] = float(np.mean(ok[m]) * 100)
            r[f"{s}_clean_bal"] = float(np.mean([np.mean(pred[m & (y == c)] == c)
                                                 for c in range(4) if np.any(m & (y == c))]) * 100)
        r["C"] = clf.best_params_["logisticregression__C"]
        split_res[name] = r
    res["split"] = split_res

    # five-fold CV on the folds used for the deep models
    cv = {}
    for name, fx in variants.items():
        accs, hits = [], {}
        for k in range(5):
            f = json.load(open(RESULTS / f"split_kfold_effnet_b0_mel_f{k}.json"))
            trk = f["train"] + f["val"]
            clf = fit(np.stack([fx(feats[p][0]) for p in trk]), np.array([lab[p] for p in trk]))
            Xt = np.stack([fx(feats[p][0]) for p in f["test"]])
            ok = clf.predict(Xt) == np.array([lab[p] for p in f["test"]])
            hits.update(dict(zip(f["test"], ok.tolist())))
            accs.append(float(np.mean(ok) * 100))
        cv[name] = {"folds": accs, "mean": float(np.mean(accs)), "sd_sample": float(np.std(accs, ddof=1)),
                    "clean_acc": float(np.mean([v for p, v in hits.items() if p in clean_cv]) * 100),
                    "n_clean": sum(p in clean_cv for p in hits),
                    "correct": {p: bool(v) for p, v in hits.items()}}
    res["cv"] = cv

    # paired McNemar of each profile classifier against the deep models on the
    # pooled out-of-fold predictions (all 600 and duplicate-free 217)
    from scripts.revision.stats import mcnemar
    deep = {}
    for m in ("effnet_b0_mel", "ekm_mfcc"):
        deep[m] = {}
        for k in range(5):
            for r in json.load(open(RESULTS / f"{m}_f{k}_s1234.json"))["test"]["rows"]:
                deep[m][r["source"]] = r
    for name in variants:
        prof = {p: {"y": 0, "pred": 0 if v else 1} for p, v in cv[name]["correct"].items()}
        for m in deep:
            for part, keys in (("all", sorted(prof)), ("clean", sorted(clean_cv))):
                cv[name][f"mcnemar_vs_{m}_{part}"] = mcnemar(prof, deep[m], keys)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    np.savez_compressed(RESULTS / "revision" / "pitch_profiles_features.npz",
                        paths=np.array(list(feats)),
                        pcp=np.stack([feats[p][0] for p in feats]))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
