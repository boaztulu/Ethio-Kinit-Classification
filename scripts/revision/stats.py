"""Statistics for the revision, recomputed from the stored per-recording rows.

All inputs are the run JSONs in results/ (nothing is re-trained). Every
benchmark statistic is computed twice: on the full 120-recording test set and
on its duplicate-free subset (test recordings with no copy in training or
validation, from results/revision/duplicates.json). Outputs
results/revision/stats.json.

  * per-configuration mean and SAMPLE standard deviation over three seeds,
    macro-F1 and balanced accuracy, bootstrap CI of the three-seed mean
  * paired bootstrap of the three-seed-mean accuracy difference to the
    EKM-MFCC reproduction (Holm-adjusted over the six comparisons) and
    seed-matched McNemar tests (Holm-adjusted within each seed)
  * per-class F1 over all seeds, pooled and median-seed confusion matrices
  * five-fold cross-validation on all 600 and on the duplicate-free
    out-of-fold recordings, with McNemar tests on the pooled predictions
  * accuracy by native sampling rate of the source file
  * k-nearest-neighbour label purity of the EfficientNet-B0 segment
    embeddings, excluding the same recording and its duplicate group
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import binomtest, chi2

from scripts.revision.common import CLASSES, MODELS, RESULTS, SEEDS
from scripts.revision.duplicates import components

OUT = RESULTS / "revision"
REF = "ekm_mfcc"
OTHERS = [m for m in MODELS if m != REF]
N_BOOT = 10_000
THR = 10


def rows(tag):
    return {r["source"]: r for r in json.load(open(RESULTS / f"{tag}.json"))["test"]["rows"]}


def correct(r):
    return int(r["y"] == r["pred"])


def per_class_f1(rs):
    y = np.array([r["y"] for r in rs]); p = np.array([r["pred"] for r in rs])
    f = []
    for c in range(4):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(f)


def balanced_acc(rs):
    y = np.array([r["y"] for r in rs]); p = np.array([r["pred"] for r in rs])
    return float(np.mean([np.mean(p[y == c] == c) for c in range(4) if np.any(y == c)]))


def confusion(rs):
    m = np.zeros((4, 4), int)
    for r in rs:
        m[r["y"], r["pred"]] += 1
    return m


def mcnemar(a, b, keys=None):
    keys = sorted(set(a) & set(b)) if keys is None else keys
    n10 = sum(correct(a[k]) and not correct(b[k]) for k in keys)
    n01 = sum(correct(b[k]) and not correct(a[k]) for k in keys)
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if n10 + n01 else 0.0
    p = float(chi2.sf(stat, 1)) if n10 + n01 else 1.0
    p_exact = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"n": len(keys), "n10": n10, "n01": n01, "chi2": stat, "p": p, "p_exact": p_exact}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def benchmark(R, keys, rng):
    """All seed-level statistics restricted to the recordings in `keys`."""
    out = {"n": len(keys), "summary": {}, "mcnemar": {}, "paired_bootstrap": {}}
    idx = rng.integers(0, len(keys), (N_BOOT, len(keys)))
    C = {m: np.array([[correct(R[m][s][k]) for k in keys] for s in SEEDS]) for m in MODELS}
    for m in MODELS:
        accs = C[m].mean(1) * 100
        sub = {s: [R[m][s][k] for k in keys] for s in SEEDS}
        f1s = np.array([per_class_f1(sub[s]).mean() for s in SEEDS]) * 100
        bal = np.array([balanced_acc(sub[s]) for s in SEEDS]) * 100
        boot = C[m][:, idx].mean(axis=(0, 2)) * 100
        out["summary"][m] = {
            "acc_seeds": accs.tolist(), "acc_mean": accs.mean(), "acc_sd": accs.std(ddof=1),
            "f1_mean": f1s.mean(), "f1_sd": f1s.std(ddof=1),
            "bal_mean": bal.mean(), "bal_sd": bal.std(ddof=1),
            "median_seed": int(np.array(SEEDS)[np.argsort(accs, kind="stable")[1]]),
            "ci_seedmean": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    for m in OTHERS:
        out["mcnemar"][m] = {s: mcnemar(R[m][s], R[REF][s], keys) for s in SEEDS}
    for s in SEEDS:
        adj = holm([out["mcnemar"][m][s]["p"] for m in OTHERS])
        for m, h in zip(OTHERS, adj):
            out["mcnemar"][m][s]["p_holm"] = h
    pairs = [(m, REF) for m in OTHERS] + [("effnet_b0_mel", b) for b in
                                          ("ast", "wav2vec2", "hubert", "effnet_b2_mel")]
    for a, b in pairs:
        d = C[a].mean(0) - C[b].mean(0)
        bd = d[idx].mean(1) * 100
        p = 2 * min(np.mean(bd <= 0), np.mean(bd >= 0))
        out["paired_bootstrap"][f"{a}-vs-{b}"] = {
            "diff": d.mean() * 100, "p_boot": float(min(1.0, max(p, 1 / N_BOOT))),
            "ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
    adj = holm([out["paired_bootstrap"][f"{m}-vs-{REF}"]["p_boot"] for m in OTHERS])
    for m, h in zip(OTHERS, adj):
        out["paired_bootstrap"][f"{m}-vs-{REF}"]["p_holm"] = h
    out["per_class"] = {}
    for m in ["effnet_b0_mel", REF]:
        F = np.array([per_class_f1([R[m][s][k] for k in keys]) for s in SEEDS]) * 100
        med = out["summary"][m]["median_seed"]
        out["per_class"][m] = {
            "f1_mean": F.mean(0).tolist(), "f1_sd": F.std(0, ddof=1).tolist(),
            "confusion_pooled": sum(confusion([R[m][s][k] for k in keys]) for s in SEEDS).tolist(),
            "confusion_median": confusion([R[m][med][k] for k in keys]).tolist()}
    return out


def main():
    rng = np.random.default_rng(0)
    dup = json.load(open(OUT / "duplicates.json"))[f"thr{THR}"]
    R = {m: {s: rows(f"{m}_s{s}") for s in SEEDS} for m in MODELS}
    test = json.load(open(RESULTS / "split_paper.json"))["test"]
    clean = dup["_clean_test"]
    S = {"threshold": THR, "full": benchmark(R, test, rng), "clean": benchmark(R, clean, rng)}

    # ensembles (posterior mean of the three EfficientNet-B0 seeds; heterogeneous)
    ens = {n: {r["source"]: r for r in json.load(open(RESULTS / f"ensemble_{n}.json"))["rows"]}
           for n in ["effnet3", "top3", "all"]}
    S["ensembles"] = {}
    for name, keys in (("full", test), ("clean", clean)):
        S["ensembles"][name] = {
            "effnet3_acc": float(np.mean([correct(ens["effnet3"][k]) for k in keys]) * 100),
            "top3_acc": float(np.mean([correct(ens["top3"][k]) for k in keys]) * 100),
            "all_acc": float(np.mean([correct(ens["all"][k]) for k in keys]) * 100),
            "effnet3_vs_top3": mcnemar(ens["effnet3"], ens["top3"], keys),
            "effnet3_vs_all": mcnemar(ens["effnet3"], ens["all"], keys),
            "effnet3_vs_seeds": {s: mcnemar(ens["effnet3"], R["effnet_b0_mel"][s], keys) for s in SEEDS}}

    # cross-validation: all 600 and duplicate-free out-of-fold recordings
    man = {m["path"]: m for m in json.load(open(RESULTS / "manifest.json"))}
    n2p = {p.split("/")[-1]: p for p in man}
    fp = json.load(open(OUT / "fingerprint.json"))
    edges = [(n2p[p["a"]], n2p[p["b"]]) for p in fp["pairs_ge5"] if p["score"] >= THR]
    groups = components(list(man), edges)
    gid = {p: i for i, g in enumerate(groups) for p in g}
    pooled, clean_cv = {}, set()
    for m in ["effnet_b0_mel", REF]:
        pooled[m] = {}
        for k in range(5):
            f = json.load(open(RESULTS / f"split_kfold_{m}_f{k}.json"))
            rr = json.load(open(RESULTS / f"{m}_f{k}_s1234.json"))["test"]["rows"]
            pooled[m].update({r["source"]: r for r in rr})
            trg = {gid[p] for p in f["train"] + f["val"]}
            clean_cv |= {p for p in f["test"] if gid[p] not in trg}
    clean_cv = sorted(clean_cv)
    cv = {}
    for name, keys in (("all", sorted(pooled[REF])), ("clean", clean_cv)):
        d = {"n": len(keys), "n_by_class": {c: sum(man[k]["class_name"] == c for k in keys) for c in CLASSES}}
        for m in ["effnet_b0_mel", REF]:
            rs = [pooled[m][k] for k in keys]
            C = np.array([correct(r) for r in rs])
            boot = C[rng.integers(0, len(keys), (N_BOOT, len(keys)))].mean(1) * 100
            d[m] = {"acc": C.mean() * 100, "f1": per_class_f1(rs).mean() * 100,
                    "bal": balanced_acc(rs) * 100,
                    "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "per_class_f1": (per_class_f1(rs) * 100).tolist(),
                    "confusion": confusion(rs).tolist()}
        d["mcnemar"] = mcnemar(pooled["effnet_b0_mel"], pooled[REF], keys)
        cv[name] = d
    folds = {m: [json.load(open(RESULTS / f"{m}_f{k}_s1234.json"))["test"]["clip_acc"] * 100
                 for k in range(5)] for m in ["effnet_b0_mel", REF]}
    cv["fold_acc"] = {m: {"folds": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                      for m, v in folds.items()}
    S["cv"] = cv
    S["cv_clean_keys"] = clean_cv

    # native sampling rate
    sr_tab = {c: {"16000": 0, "44100": 0} for c in CLASSES}
    for m in man.values():
        sr_tab[m["class_name"]][str(m["samplerate"])] += 1
    by_sr = {}
    for m in MODELS:
        by_sr[m] = {}
        for sr in (16000, 44100):
            ks = [k for k in test if man[k]["samplerate"] == sr]
            by_sr[m][str(sr)] = {"n": len(ks), "acc": float(np.mean(
                [np.mean([correct(R[m][s][k]) for k in ks]) for s in SEEDS]) * 100)}
    S["sampling_rate"] = {"corpus_by_class": sr_tab, "acc_by_sr": by_sr}

    # embedding neighbourhoods (EfficientNet-B0 test segments)
    knn = {}
    for s in SEEDS:
        d = np.load(RESULTS / f"effnet_b0_mel_s{s}" / "eval" / "embeddings.npz", allow_pickle=True)
        X = d["X"].astype(np.float64); Y = d["Y"]; src = d["srcs"]
        g = np.array([gid[p] for p in src])
        sq = (X ** 2).sum(1)
        D = sq[:, None] + sq[None, :] - 2 * X @ X.T
        np.fill_diagonal(D, np.inf)
        same_rec = src[:, None] == src[None, :]
        same_grp = g[:, None] == g[None, :]
        o = {}
        for name, mask in (("all", None), ("excl_recording", same_rec), ("excl_group", same_grp)):
            Dm = D if mask is None else np.where(mask, np.inf, D)
            nn = np.argsort(Dm, axis=1)[:, :10]
            o[name] = float((Y[nn] == Y[:, None]).mean() * 100)
            o[name + "_share_same_rec"] = float(same_rec[np.arange(len(Y))[:, None], nn].mean() * 100)
        knn[s] = o
    S["knn"] = knn

    json.dump(S, open(OUT / "stats.json", "w"), indent=1, default=float)

    # printout
    for part in ("full", "clean"):
        B = S[part]
        print(f"===== {part} test set (n={B['n']})")
        for m in MODELS:
            v = B["summary"][m]
            print(f"{m:15s} acc {v['acc_mean']:.2f} ± {v['acc_sd']:.2f}  F1 {v['f1_mean']:.2f} ± {v['f1_sd']:.2f}"
                  f"  bal {v['bal_mean']:.2f}  seeds {np.round(v['acc_seeds'], 1)}  CI [{v['ci_seedmean'][0]:.1f}, {v['ci_seedmean'][1]:.1f}]")
        for k, v in B["paired_bootstrap"].items():
            print(f"   {k:30s} {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] p={v['p_boot']:.4f}"
                  + (f" holm={v['p_holm']:.4f}" if "p_holm" in v else ""))
        for m in OTHERS:
            q = B["mcnemar"][m]
            print(f"   McNemar {m:15s} " + " | ".join(
                f"s{s} {q[s]['n10']}/{q[s]['n01']} p={q[s]['p']:.3f} holm={q[s]['p_holm']:.3f}" for s in SEEDS))
        for m, v in B["per_class"].items():
            print(f"   per-class {m}: " + ", ".join(f"{a:.1f}±{b:.1f}" for a, b in zip(v["f1_mean"], v["f1_sd"]))
                  + f"  pooled CM {v['confusion_pooled']}")
    print("===== ensembles", json.dumps(S["ensembles"], default=float)[:1500])
    print("===== CV")
    for name in ("all", "clean"):
        d = cv[name]
        print(f"{name}: n={d['n']} {d['n_by_class']}  EffNet {d['effnet_b0_mel']['acc']:.2f} {np.round(d['effnet_b0_mel']['ci'], 1)} "
              f"F1 {d['effnet_b0_mel']['f1']:.2f} bal {d['effnet_b0_mel']['bal']:.2f} | EKM {d[REF]['acc']:.2f} {np.round(d[REF]['ci'], 1)} "
              f"F1 {d[REF]['f1']:.2f} bal {d[REF]['bal']:.2f} | McNemar {d['mcnemar']}")
    print("fold acc", cv["fold_acc"])
    print("===== kNN", json.dumps(knn))


if __name__ == "__main__":
    main()
