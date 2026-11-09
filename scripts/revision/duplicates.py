"""Duplicate structure of EMIR and its effect on the reported accuracies.

Uses results/revision/fingerprint.json. Two recordings are `related` when
their best consistent-offset landmark count reaches THR (default 10; 20 is
reported as a sensitivity check). Related recordings are merged into groups
(connected components). For the recording-level 70/10/20 split and for each
cross-validation fold, a test recording is `contaminated` when it is related
to any recording of its training or validation set. Accuracies of the stored
predictions (no re-training) are then recomputed on the contaminated and on
the duplicate-free test recordings.

Writes results/revision/duplicates.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from scripts.revision.common import CLASSES, MODELS, RESULTS, SEEDS

FP = RESULTS / "revision" / "fingerprint.json"


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return list(groups.values())


def analyse(thr, fp, man, name_to_path):
    edges = [(name_to_path[p["a"]], name_to_path[p["b"]]) for p in fp["pairs_ge5"] if p["score"] >= thr]
    nbr = defaultdict(set)
    for a, b in edges:
        nbr[a].add(b); nbr[b].add(a)
    groups = components(list(man), edges)
    multi = [g for g in groups if len(g) > 1]
    conflicts = [sorted({man[p]["class_name"] for p in g}) for g in multi
                 if len({man[p]["class_name"] for p in g}) > 1]
    conflict_groups = [sorted(p.split("/")[-1] for p in g) for g in multi
                       if len({man[p]["class_name"] for p in g}) > 1]
    out = {"threshold": thr, "n_related_pairs": len(edges), "n_groups": len(groups),
           "n_groups_size_gt1": len(multi), "n_recordings_in_groups": sum(len(g) for g in multi),
           "group_size_hist": np.bincount([len(g) for g in groups]).tolist(),
           "n_label_conflict_groups": len(conflicts), "label_conflict_groups": conflict_groups,
           "n_recordings_in_conflict_groups": sum(len(g) for g in conflict_groups)}

    def contaminated(test, train):
        tr = set(train)
        return [p for p in test if nbr[p] & tr]

    sp = json.load(open(RESULTS / "split_paper.json"))
    cont = contaminated(sp["test"], sp["train"] + sp["val"])
    cont_train_only = contaminated(sp["test"], sp["train"])
    clean = [p for p in sp["test"] if p not in set(cont)]
    out["split"] = {"n_test": len(sp["test"]), "n_contaminated": len(cont),
                    "n_contaminated_train_only": len(cont_train_only), "n_clean": len(clean),
                    "clean_by_class": {c: sum(man[p]["class_name"] == c for p in clean) for c in CLASSES}}
    acc = {}
    for m in MODELS:
        a_all, a_cont, a_clean = [], [], []
        for s in SEEDS:
            rows = {r["source"]: r for r in json.load(open(RESULTS / f"{m}_s{s}.json"))["test"]["rows"]}
            ok = {k: rows[k]["y"] == rows[k]["pred"] for k in rows}
            a_all.append(np.mean([ok[k] for k in sp["test"]]) * 100)
            a_cont.append(np.mean([ok[k] for k in cont]) * 100)
            a_clean.append(np.mean([ok[k] for k in clean]) * 100)
        acc[m] = {"all": [float(np.mean(a_all)), float(np.std(a_all, ddof=1))],
                  "contaminated": [float(np.mean(a_cont)), float(np.std(a_cont, ddof=1))],
                  "clean": [float(np.mean(a_clean)), float(np.std(a_clean, ddof=1))],
                  "clean_seeds": a_clean}
    ens = {r["source"]: r for r in json.load(open(RESULTS / "ensemble_effnet3.json"))["rows"]}
    acc["effnet_b0_ensemble"] = {
        "all": [float(np.mean([ens[k]["y"] == ens[k]["pred"] for k in sp["test"]]) * 100), 0.0],
        "contaminated": [float(np.mean([ens[k]["y"] == ens[k]["pred"] for k in cont]) * 100), 0.0],
        "clean": [float(np.mean([ens[k]["y"] == ens[k]["pred"] for k in clean]) * 100), 0.0]}
    out["split"]["accuracy"] = acc

    # cross-validation folds: pooled out-of-fold predictions
    cv = {}
    for m in ["effnet_b0_mel", "ekm_mfcc"]:
        n_cont, ok_cont, ok_clean, n_clean = 0, 0, 0, 0
        for k in range(5):
            f = json.load(open(RESULTS / f"split_kfold_{m}_f{k}.json"))
            rows = {r["source"]: r for r in json.load(open(RESULTS / f"{m}_f{k}_s1234.json"))["test"]["rows"]}
            c = set(contaminated(f["test"], f["train"] + f["val"]))
            for p in f["test"]:
                hit = rows[p]["y"] == rows[p]["pred"]
                if p in c:
                    n_cont += 1; ok_cont += hit
                else:
                    n_clean += 1; ok_clean += hit
        cv[m] = {"n_contaminated": n_cont, "n_clean": n_clean,
                 "acc_contaminated": 100 * ok_cont / n_cont, "acc_clean": 100 * ok_clean / n_clean,
                 "acc_all": 100 * (ok_cont + ok_clean) / (n_cont + n_clean)}
    out["cv"] = cv
    out["_clean_test"] = clean
    return out


def main():
    fp = json.load(open(FP))
    man = {m["path"]: m for m in json.load(open(RESULTS / "manifest.json"))}
    name_to_path = {p.split("/")[-1]: p for p in man}
    res = {"md5_groups": len(fp["md5_duplicate_groups"]),
           "md5_files": sum(len(g) for g in fp["md5_duplicate_groups"]),
           "md5_cross_class": [g for g in fp["md5_duplicate_groups"]
                               if len({man[name_to_path[n]]["class_name"] for n in g}) > 1]}
    for thr in (10, 20):
        res[f"thr{thr}"] = analyse(thr, fp, man, name_to_path)
    json.dump(res, open(RESULTS / "revision" / "duplicates.json", "w"), indent=1)
    for thr in (10, 20):
        r = res[f"thr{thr}"]
        print(f"== threshold {thr}: pairs {r['n_related_pairs']}, groups>1 {r['n_groups_size_gt1']} "
              f"covering {r['n_recordings_in_groups']} recordings, distinct recordings {r['n_groups']}, "
              f"group sizes {r['group_size_hist']}")
        print("   label-conflict groups:", r["n_label_conflict_groups"], r["label_conflict_groups"])
        s = r["split"]
        print(f"   test: contaminated {s['n_contaminated']} (train only {s['n_contaminated_train_only']}), "
              f"clean {s['n_clean']} {s['clean_by_class']}")
        for m, v in s["accuracy"].items():
            print(f"   {m:20s} all {v['all'][0]:.2f}  contaminated {v['contaminated'][0]:.2f}  "
                  f"clean {v['clean'][0]:.2f} ± {v['clean'][1]:.2f}")
        print("   CV:", r["cv"])
    print("md5:", res["md5_groups"], "groups,", res["md5_files"], "files; cross-class:", res["md5_cross_class"])


if __name__ == "__main__":
    main()
