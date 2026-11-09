"""Evaluate saved checkpoints on arbitrary segment indices (CPU).

For every checkpoint tag the model is loaded once and evaluated on each
condition, given as name=<index.json>. `--sources` selects the recordings:
  test   the 120 test recordings of the recording-level split (results/split_paper.json)
  all    all 600 recordings (training, validation and test)
  chunk  the segments of the segment-level test split (split_chunk_level.json)

Writes <out_dir>/<tag>__<condition>.json with per-recording rows and
accuracies, and with --save_seg also the per-segment posteriors (.npz).
Existing outputs are skipped, so an interrupted run can be resumed.

Usage:
    python -m scripts.revision.eval_sets --tags ast_s1234 ... \
        --conditions env10=<dir>/env10/index.json ... --sources test \
        --out_dir results/revision/degradation
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scripts.revision.common import (RESULTS, FixedScaleChunks, acc, load_model,
                                     pin_cpus, pool_clips, predict_segments, split)
from src.data.dataset import EmirChunks


def chunk_sources(index_path: str):
    """Map the HiPerGator chunk paths of split_chunk_level.json to local ones."""
    idx = json.load(open(index_path))
    by_key = {(r["source_path"], r["chunk_idx"]): r["wav_path"] for r in idx}
    out = []
    for p in split("split_chunk_level.json")["test"]:
        cls, name = p.split("/")[-2:]
        stem, k = name[:-4].rsplit("__", 1)
        src = f"EMIR_Dataset_V1/{cls}/{stem}.wav"
        out.append(by_key[(src, int(k))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--sources", choices=["test", "all", "chunk", "clean"], default="test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_seg", action="store_true")
    ap.add_argument("--fixed_scale", action="store_true",
                    help="log-mel scaled by (S+80)/80 instead of per-segment min-max")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    pin_cpus()
    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = [c.split("=", 1) for c in args.conditions]

    for tag in args.tags:
        todo = [(n, i) for n, i in conds if not (out_dir / f"{tag}__{n}.json").exists()]
        if not todo:
            continue
        model, mode, _ = load_model(tag)
        for name, index in todo:
            t0 = time.time()
            if args.sources == "chunk":
                src = chunk_sources(index)
            elif args.sources == "all":
                src = json.load(open(RESULTS / "split_paper.json"))
                src = src["train"] + src["val"] + src["test"]
            elif args.sources == "clean":
                src = json.load(open(RESULTS / "revision" / "duplicates.json"))["thr10"]["_clean_test"]
            else:
                src = split()["test"]
            seg = predict_segments(model, mode, src, index=index,
                                   batch_size=args.batch_size,
                                   dataset_cls=FixedScaleChunks if args.fixed_scale else EmirChunks)
            rows = pool_clips(seg)
            seg_acc = float(np.mean(seg["probs"].argmax(1) == seg["y"]))
            out = {"tag": tag, "condition": name, "sources": args.sources,
                   "n_recordings": len(rows), "n_segments": int(len(seg["y"])),
                   "clip_acc": acc(rows), "seg_acc": seg_acc, "rows": rows,
                   "seconds": time.time() - t0}
            if args.save_seg:
                np.savez_compressed(out_dir / f"{tag}__{name}.npz", **seg)
            json.dump(out, open(out_dir / f"{tag}__{name}.json", "w"))
            print(f"[eval_sets] {tag} {name}: clip {out['clip_acc']:.4f} "
                  f"seg {seg_acc:.4f} n={len(rows)} {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
