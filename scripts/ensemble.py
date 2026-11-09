"""Logit / softmax ensemble of N model predictions on the test set.

Reads `rows` from each model's eval/eval.json (which contain per-clip softmax
posteriors), averages, and reports accuracy + McNemar's against each member.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import erfc, sqrt
from pathlib import Path

import numpy as np


def acc(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, n_classes=4):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            p = tp / (tp + fp); r = tp / (tp + fn)
            f1 = 2*p*r/(p+r) if p+r > 0 else 0.0
        f1s.append(float(f1))
    return float(np.mean(f1s))


def mcnemar(rows_a, rows_b):
    a = {r["source"]: (r["y"], r["pred"]) for r in rows_a}
    b = {r["source"]: (r["y"], r["pred"]) for r in rows_b}
    common = sorted(set(a) & set(b))
    n10 = n01 = 0
    for s in common:
        ca = (a[s][1] == a[s][0]); cb = (b[s][1] == b[s][0])
        if ca and not cb: n10 += 1
        elif cb and not ca: n01 += 1
    if n10 + n01 == 0:
        return {"n": len(common), "n10": n10, "n01": n01, "stat": 0.0, "p": 1.0}
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p = erfc(sqrt(stat / 2.0))
    return {"n": len(common), "n10": n10, "n01": n01, "stat": float(stat), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="list of model tags whose eval/eval.json to ensemble")
    ap.add_argument("--out", default="results/ensemble.json")
    args = ap.parse_args()

    by_clip = defaultdict(lambda: {"probs": [], "y": None})
    members = {}
    for tag in args.tags:
        ev = Path(f"results/{tag}/eval/eval.json")
        if not ev.exists():
            print(f"[skip] {tag}: missing {ev}")
            continue
        d = json.load(open(ev))
        members[tag] = d["rows"]
        for r in d["rows"]:
            s = r["source"]
            by_clip[s]["probs"].append(r["probs"])
            by_clip[s]["y"] = r["y"]

    rows = []
    for s, d in by_clip.items():
        if len(d["probs"]) < len(members):
            continue
        p = np.mean(np.stack(d["probs"], 0), axis=0)
        rows.append({"source": s, "y": int(d["y"]), "pred": int(p.argmax()), "probs": p.tolist()})
    if not rows:
        print("[error] no clips with predictions from all members")
        return

    a = acc([r["y"] for r in rows], [r["pred"] for r in rows])
    f = macro_f1([r["y"] for r in rows], [r["pred"] for r in rows])

    out = {
        "members": args.tags,
        "n_members": len(members),
        "n_clips": len(rows),
        "acc": a,
        "macro_f1": f,
        "rows": rows,
    }
    print(f"[ensemble] members={list(members)}  acc={a:.4f}  f1={f:.4f}")

    # McNemar vs each member
    out["mcnemar_vs_members"] = {}
    for tag, mrows in members.items():
        out["mcnemar_vs_members"][tag] = mcnemar(rows, mrows)
        print(f"  vs {tag:30s} n10={out['mcnemar_vs_members'][tag]['n10']}  n01={out['mcnemar_vs_members'][tag]['n01']}  p={out['mcnemar_vs_members'][tag]['p']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ensemble] wrote {args.out}")


if __name__ == "__main__":
    main()
