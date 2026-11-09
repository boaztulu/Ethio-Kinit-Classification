"""Collect every number quoted in the revised manuscript into one JSON.

Reads results/ and results/revision/ (all produced by the other scripts in
this folder) and writes results/revision/paper_numbers.json. Sections whose
inputs are not yet available are skipped with a note, so the script can be
re-run as evaluations finish.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.revision.common import CLASSES, MODELS, RESULTS, SEEDS
from scripts.revision.duplicates import components

REV = RESULTS / "revision"
N_BOOT = 10_000


def load(p):
    return json.load(open(p))


def correct(r):
    return int(r["y"] == r["pred"])


def confusion(rows):
    m = np.zeros((4, 4), int)
    for r in rows:
        m[r["y"], r["pred"]] += 1
    return m


def bal(rows):
    """Balanced accuracy (mean per-class recall), in %."""
    y = np.array([r["y"] for r in rows]); p = np.array([r["pred"] for r in rows])
    return float(np.mean([np.mean(p[y == c] == c) for c in range(4) if np.any(y == c)]) * 100)


def tizita_share(rows):
    return float(np.mean([r["pred"] == 0 for r in rows]) * 100)


def main():
    rng = np.random.default_rng(0)
    out = {"notes": []}
    man = {m["path"]: m for m in load(RESULTS / "manifest.json")}
    n2p = {p.split("/")[-1]: p for p in man}
    sp = load(RESULTS / "split_paper.json")
    test = sp["test"]
    dup = load(REV / "duplicates.json")
    clean = dup["thr10"]["_clean_test"]
    cont = [p for p in test if p not in set(clean)]
    stats = load(REV / "stats.json")
    fp = load(REV / "fingerprint.json")

    # duplicate structure
    edges = [(n2p[p["a"]], n2p[p["b"]]) for p in fp["pairs_ge5"] if p["score"] >= 10]
    groups = components(list(man), edges)
    gid = {p: i for i, g in enumerate(groups) for p in g}
    per_class = {}
    for c in CLASSES:
        files = [p for p in man if man[p]["class_name"] == c]
        per_class[c] = {"files": len(files), "distinct": len({gid[p] for p in files}),
                        "in_groups": sum(len(groups[gid[p]]) > 1 for p in files)}
    out["duplicates"] = {
        "md5_groups": dup["md5_groups"], "md5_files": dup["md5_files"],
        "thr10": {k: dup["thr10"][k] for k in ("n_related_pairs", "n_groups", "n_groups_size_gt1",
                                               "n_recordings_in_groups", "group_size_hist",
                                               "n_label_conflict_groups", "label_conflict_groups")},
        "thr20": {k: dup["thr20"][k] for k in ("n_groups", "n_groups_size_gt1", "n_recordings_in_groups",
                                               "n_label_conflict_groups")},
        "per_class": per_class,
        "test_contaminated": len(cont), "test_clean": len(clean),
        "test_clean_by_class": dup["thr10"]["split"]["clean_by_class"],
        "cv_contaminated": dup["thr10"]["cv"]["effnet_b0_mel"]["n_contaminated"],
        "cv_clean": dup["thr10"]["cv"]["effnet_b0_mel"]["n_clean"],
        "controls_min": int(min(fp["control_positive_scores"])),
        "controls_median": int(np.median(fp["control_positive_scores"])),
        "score_hist": fp["pair_score_hist"],
        "cross_class_by_band": {},
    }
    for lo, hi in [(5, 10), (10, 20), (20, 1000), (1000, 10 ** 9)]:
        sel = [p for p in fp["pairs_ge5"] if lo <= p["score"] < hi]
        out["duplicates"]["cross_class_by_band"][f"{lo}-{hi}"] = [len(sel), sum(p["class_a"] != p["class_b"] for p in sel)]

    # benchmark (from stats.json)
    bench = {}
    for m in MODELS:
        f, c = stats["full"]["summary"][m], stats["clean"]["summary"][m]
        a_cont = [np.mean([correct(r) for r in [ {x["source"]: x for x in load(RESULTS / f"{m}_s{s}.json")["test"]["rows"]}[k] for k in cont]]) * 100 for s in SEEDS]
        bench[m] = {"full_acc": f["acc_mean"], "full_sd": f["acc_sd"], "full_f1": f["f1_mean"],
                    "full_ci": f["ci_seedmean"], "cont_acc": float(np.mean(a_cont)),
                    "clean_acc": c["acc_mean"], "clean_sd": c["acc_sd"], "clean_f1": c["f1_mean"],
                    "clean_f1_sd": c["f1_sd"], "clean_bal": c["bal_mean"], "clean_ci": c["ci_seedmean"]}
        if m != "ekm_mfcc":
            bench[m]["full_p_holm"] = stats["full"]["paired_bootstrap"][f"{m}-vs-ekm_mfcc"]["p_holm"]
            bench[m]["clean_p_holm"] = stats["clean"]["paired_bootstrap"][f"{m}-vs-ekm_mfcc"]["p_holm"]
            bench[m]["clean_diff"] = stats["clean"]["paired_bootstrap"][f"{m}-vs-ekm_mfcc"]["diff"]
            bench[m]["clean_diff_ci"] = stats["clean"]["paired_bootstrap"][f"{m}-vs-ekm_mfcc"]["ci"]
    out["benchmark"] = bench
    out["ensemble"] = stats["ensembles"]

    # cross-validation per fold, all vs duplicate-free
    cvf = {}
    for m in ["effnet_b0_mel", "ekm_mfcc"]:
        a, cl = [], []
        for k in range(5):
            f = load(RESULTS / f"split_kfold_{m}_f{k}.json")
            rows = {r["source"]: r for r in load(RESULTS / f"{m}_f{k}_s1234.json")["test"]["rows"]}
            trg = {gid[p] for p in f["train"] + f["val"]}
            ck = [p for p in f["test"] if gid[p] not in trg]
            a.append(np.mean([correct(rows[p]) for p in f["test"]]) * 100)
            cl.append(np.mean([correct(rows[p]) for p in ck]) * 100)
        cvf[m] = {"all": a, "clean": cl, "clean_n": None}
    out["cv"] = {"per_fold": cvf, "pooled": {k: stats["cv"][k] for k in ("all", "clean")},
                 "fold_acc": stats["cv"]["fold_acc"]}

    # readings of the published 95 %
    rd = {}
    wc = REV / "whole_corpus"
    if all((wc / f"ekm_mfcc_s{s}__clean.json").exists() for s in SEEDS):
        for m in ["ekm_mfcc", "effnet_b0_mel"]:
            accs, cms = [], []
            for s in SEEDS:
                d = load(wc / f"{m}_s{s}__clean.json")
                accs.append(d["clip_acc"] * 100)
                cms.append(confusion(d["rows"]))
                part = {}
                for name in ("train", "val", "test"):
                    ks = set(sp[name]); rr = [r for r in d["rows"] if r["source"] in ks]
                    part[name] = float(np.mean([correct(r) for r in rr]) * 100)
                rd.setdefault(m, {}).setdefault("by_partition", {})[s] = part
            rd[m]["whole_corpus_seeds"] = accs
            rd[m]["whole_corpus_mean"] = float(np.mean(accs))
            rd[m]["whole_corpus_confusion_seed_median"] = cms[int(np.argsort(accs)[1])].tolist()
            rd[m]["whole_corpus_confusion_sums"] = cms[0].sum(1).tolist()
        rd["ekm_mfcc"]["contaminated_test_mean"] = bench["ekm_mfcc"]["cont_acc"]
        rd["ekm_mfcc"]["recording_split_test_mean"] = bench["ekm_mfcc"]["full_acc"]
        rd["ekm_mfcc"]["cv_pooled_all"] = stats["cv"]["all"]["ekm_mfcc"]["acc"]
        rd["ekm_mfcc"]["cv_pooled_clean"] = stats["cv"]["clean"]["ekm_mfcc"]["acc"]
        rd["ekm_mfcc"]["clean_test_mean"] = bench["ekm_mfcc"]["clean_acc"]
    else:
        out["notes"].append("whole-corpus evaluation incomplete")
    out["readings_95"] = rd

    # segment accuracy ladder and inflation CIs
    cp = REV / "chunk_protocol"
    if all((cp / f"{m}_chunk_s{s}__clean.npz").exists() for m in ("effnet_b0_mel", "ekm_mfcc") for s in SEEDS):
        lad = {}
        for m in ("effnet_b0_mel", "ekm_mfcc"):
            leak = [np.load(cp / f"{m}_chunk_s{s}__clean.npz") for s in SEEDS]
            free = [np.load(wc / f"{m}_s{s}__clean.npz") for s in SEEDS]
            # per-recording lists of segment correctness, averaged over seeds
            def by_rec(segs, keep):
                d = defaultdict(list)
                ok = np.mean([(z["probs"].argmax(1) == z["y"]) for z in segs], 0)
                for s, v in zip(segs[0]["src"], ok):
                    if s in keep:
                        d[s].append(v)
                return d
            L = by_rec(leak, set(leak[0]["src"].tolist()))
            Fa = by_rec(free, set(test))
            Fc = by_rec(free, set(clean))
            def seg_acc(d):
                return float(np.mean([v for vs in d.values() for v in vs]) * 100)
            def boot(d1, d2, n=N_BOOT):
                k1, k2 = list(d1), list(d2)
                s1 = np.array([np.sum(d1[k]) for k in k1]); c1 = np.array([len(d1[k]) for k in k1])
                s2 = np.array([np.sum(d2[k]) for k in k2]); c2 = np.array([len(d2[k]) for k in k2])
                i1 = rng.integers(0, len(k1), (n, len(k1))); i2 = rng.integers(0, len(k2), (n, len(k2)))
                b = s1[i1].sum(1) / c1[i1].sum(1) - s2[i2].sum(1) / c2[i2].sum(1)
                return [float(np.percentile(b, 2.5) * 100), float(np.percentile(b, 97.5) * 100)]
            lad[m] = {"segment_split": seg_acc(L), "recording_split_all": seg_acc(Fa),
                      "recording_split_clean": seg_acc(Fc),
                      "inflation_seg_vs_rec": seg_acc(L) - seg_acc(Fa),
                      "inflation_seg_vs_rec_ci": boot(L, Fa),
                      "inflation_rec_vs_clean": seg_acc(Fa) - seg_acc(Fc),
                      "inflation_rec_vs_clean_ci": boot(Fa, Fc),
                      "n_segments": {"segment_split": int(sum(len(v) for v in L.values())),
                                     "recording_split_all": int(sum(len(v) for v in Fa.values())),
                                     "recording_split_clean": int(sum(len(v) for v in Fc.values()))}}
        out["ladder"] = lad
    else:
        out["notes"].append("segment-level protocol predictions incomplete")

    # stored robustness probes (3 seeds, full test)
    acc = defaultdict(list)
    with open(Path("tables") / "robustness.csv", newline="") as f:
        for r in csv.DictReader(f):
            acc[(r["tag"].rsplit("_s", 1)[0], r["probe"])].append(float(r["acc"]) * 100)
    out["robust_stored"] = {f"{m}|{p}": float(np.mean(v)) for (m, p), v in acc.items()}
    out["robust_stored_nseeds"] = {f"{m}|{p}": len(v) for (m, p), v in acc.items()}

    # realistic degradations (rows saved; full and duplicate-free)
    # AST was scored on the duplicate-free recordings only (degradation_clean/),
    # every other model on all 120 test recordings (degradation/).
    deg = {}
    cl = set(clean)
    for m in MODELS:
        for cnd in ("env10", "reverb", "mp3", "tel"):
            fs = [REV / "degradation" / f"{m}_s{s}__{cnd}.json" for s in SEEDS]
            fc = [REV / "degradation_clean" / f"{m}_s{s}__{cnd}.json" for s in SEEDS]
            if all(f.exists() for f in fs):
                rows = [load(f)["rows"] for f in fs]
                crows = [[r for r in rr if r["source"] in cl] for rr in rows]
                deg[f"{m}|{cnd}"] = {
                    "full": float(np.mean([np.mean([correct(r) for r in rr]) for rr in rows]) * 100),
                    "full_tizita_share": float(np.mean([tizita_share(rr) for rr in rows])),
                    "clean": float(np.mean([np.mean([correct(r) for r in rr]) for rr in crows]) * 100),
                    "clean_bal": float(np.mean([bal(rr) for rr in crows]))}
            elif all(f.exists() for f in fc):
                rows = [load(f)["rows"] for f in fc]
                assert all(r["source"] in cl for rr in rows for r in rr)
                deg[f"{m}|{cnd}"] = {"full": None,
                                     "clean": float(np.mean([np.mean([correct(r) for r in rr]) for rr in rows]) * 100),
                                     "clean_bal": float(np.mean([bal(rr) for rr in rows]))}
    out["degradation"] = deg
    # clean-condition accuracy on the duplicate-free subset for reference
    out["clean_reference"] = {m: bench[m]["clean_acc"] for m in MODELS}

    # seeded probes and normalisation control
    for name, folder in (("probes", "probes"), ("probes_fixed", "probes_fixed"), ("probes_clean", "probes_clean")):
        res = {}
        d = REV / folder
        if d.exists():
            for f in d.glob("*.json"):
                tag, cnd = f.stem.split("__")
                m = tag.rsplit("_s", 1)[0]
                rr = load(f)["rows"]
                res.setdefault(f"{m}|{cnd}", {"full": [], "clean": [], "clean_bal": [], "tiz": []})
                if folder != "probes_clean":       # probes_clean holds duplicate-free rows only
                    res[f"{m}|{cnd}"]["full"].append(np.mean([correct(r) for r in rr]) * 100)
                    res[f"{m}|{cnd}"]["tiz"].append(tizita_share(rr))
                cr = [r for r in rr if r["source"] in set(clean)]
                if cr:
                    res[f"{m}|{cnd}"]["clean"].append(np.mean([correct(r) for r in cr]) * 100)
                    res[f"{m}|{cnd}"]["clean_bal"].append(bal(cr))
        out[name] = {k: {"n_seeds": max(len(v["full"]), len(v["clean"])),
                         "full": float(np.mean(v["full"])) if v["full"] else None,
                         "full_tizita_share": float(np.mean(v["tiz"])) if v["tiz"] else None,
                         "clean": float(np.mean(v["clean"])) if v["clean"] else None,
                         "clean_bal": float(np.mean(v["clean_bal"])) if v["clean_bal"] else None}
                     for k, v in res.items()}

    out["pitch_profiles"] = {k: v for k, v in load(REV / "pitch_profiles.json").items()}
    for v in out["pitch_profiles"]["cv"].values():
        v.pop("correct", None)
    out["knn"] = stats["knn"]
    out["sampling_rate"] = stats["sampling_rate"]
    json.dump(out, open(REV / "paper_numbers.json", "w"), indent=1, default=float)
    print("notes:", out["notes"])
    print("cv per fold:", json.dumps(cvf))
    print("readings:", json.dumps({m: {k: v for k, v in d.items() if k != "by_partition"} for m, d in rd.items()}, default=float))
    print("by partition:", json.dumps({m: d.get("by_partition") for m, d in rd.items()}))
    print("ladder:", json.dumps(out.get("ladder"), default=float))
    print("degradation:", json.dumps(deg))


if __name__ == "__main__":
    main()
