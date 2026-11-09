"""Stratified 5-fold cross-validation orchestrator.

For each fold, build a clip-level (train+val) / test split from the
stratified-kfold partitioner in src.data.splits, write a transient split
JSON, and invoke `src.train` via subprocess so each fold trains in a fresh
process (clean memory).

Outputs are written to checkpoints/<tag>_fold{k}_s{seed}/ and aggregated to
results/<base_tag>_kfold_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml

from src.data.manifest import load_manifest
from src.data.splits import Split, save_split, stratified_kfold


def split_train_val(train_val: List[str], y_lookup, seed: int, val_frac: float = 0.10):
    """Carve a stratified val split out of the train+val pool."""
    from sklearn.model_selection import train_test_split
    arr = np.array(train_val)
    y = np.array([y_lookup[p] for p in train_val])
    tr_idx, va_idx = train_test_split(
        np.arange(len(arr)),
        test_size=val_frac / (1 - 0.20),  # paper proportion
        random_state=seed, stratify=y,
    )
    return arr[tr_idx].tolist(), arr[va_idx].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default="results/manifest.json")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--inner_val_seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base_tag = cfg["log"]["tag"]

    meta = load_manifest(Path(args.manifest))
    y_lookup = {m.path: m.class_idx for m in meta}

    folds = stratified_kfold(meta, n_splits=args.n_splits, seed=args.seed)
    results = []
    for k, (trval, test) in enumerate(folds):
        tr, va = split_train_val(trval, y_lookup, seed=args.inner_val_seed + k)
        sp = Split(train=tr, val=va, test=test)
        sp_path = Path(f"results/split_kfold_{base_tag}_f{k}.json")
        save_split(sp, sp_path)

        # write a fold-specific config
        cfg_k = dict(cfg)
        cfg_k["data"] = dict(cfg["data"])
        cfg_k["data"]["split_path"] = str(sp_path)
        cfg_k["log"] = dict(cfg["log"])
        cfg_k["log"]["tag"] = f"{base_tag}_f{k}"
        cfg_path = Path(f"configs/_kfold_{base_tag}_f{k}.yaml")
        cfg_path.write_text(yaml.safe_dump(cfg_k, sort_keys=False))

        cmd = [
            sys.executable, "-m", "src.train",
            "--config", str(cfg_path),
            "--seed", str(args.seed),
        ]
        print(f"[kfold {k}] running:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[kfold {k}] FAILED with code {rc}")
            continue
        res_path = Path("results") / f"{base_tag}_f{k}_s{args.seed}.json"
        if res_path.exists():
            with res_path.open() as f:
                results.append(json.load(f))

    if results:
        accs = [r["test"]["clip_acc"] for r in results]
        f1s = [r["test"]["clip_macro_f1"] for r in results]
        summary = {
            "model": cfg["model"],
            "base_tag": base_tag,
            "n_splits": args.n_splits,
            "fold_acc": accs,
            "fold_f1": f1s,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }
        out = Path("results") / f"{base_tag}_kfold_summary.json"
        with out.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"[kfold] summary -> {out}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
