"""Reproducible splits for EMIR.

Two protocols are supported:

1. **paper_split**: stratified random 70/10/20 train/val/test over recordings
   (clips), following the protocol of Retta et al. 2023, which partitions the
   recordings before they are cut into 3-second chunks. Chunks of one
   recording therefore never appear in both train and test.

2. **stratified_kfold**: 5-fold stratified at the clip level, for the
   cross-validation results.

Both protocols are class-stratified, so that every partition keeps the class
proportions of the corpus.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .manifest import CLASS_NAMES, ClipMeta, load_manifest


@dataclass
class Split:
    train: List[str]   # list of WAV paths
    val: List[str]
    test: List[str]


def chunk_level_split(index_path: str, seed: int = 1234,
                      train_frac: float = 0.70, val_frac: float = 0.10,
                      test_frac: float = 0.20) -> Split:
    """Random 70/10/20 at the CHUNK (segment) level, the segment-level
    pitfall measured for comparison: segments of the same clip can fall
    into both train and test. The Split's train/val/test lists are wav_paths of chunks
    (not source_paths). EmirChunks must be loaded with a chunk-aware
    source filter when this split is used."""
    import json
    with open(index_path) as f:
        idx = json.load(f)
    paths = np.array([c["wav_path"] for c in idx])
    y = np.array([int(c["class_idx"]) for c in idx])
    rest_idx, test_idx = train_test_split(
        np.arange(len(idx)), test_size=test_frac, random_state=seed, stratify=y,
    )
    val_share_of_rest = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_share_of_rest, random_state=seed,
        stratify=y[rest_idx],
    )
    return Split(
        train=paths[train_idx].tolist(),
        val=paths[val_idx].tolist(),
        test=paths[test_idx].tolist(),
    )


def paper_split(meta: List[ClipMeta], seed: int = 1234,
                train_frac: float = 0.70, val_frac: float = 0.10,
                test_frac: float = 0.20) -> Split:
    """Stratified random 70/10/20 split at the clip level."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    paths = np.array([m.path for m in meta])
    y = np.array([m.class_idx for m in meta])

    # First peel off test
    rest_idx, test_idx = train_test_split(
        np.arange(len(meta)),
        test_size=test_frac,
        random_state=seed,
        stratify=y,
    )
    # Then split remaining into train / val keeping the global val fraction
    val_share_of_rest = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=val_share_of_rest,
        random_state=seed,
        stratify=y[rest_idx],
    )
    return Split(
        train=paths[train_idx].tolist(),
        val=paths[val_idx].tolist(),
        test=paths[test_idx].tolist(),
    )


def stratified_kfold(meta: List[ClipMeta], n_splits: int = 5, seed: int = 1234
                     ) -> List[Tuple[List[str], List[str]]]:
    paths = np.array([m.path for m in meta])
    y = np.array([m.class_idx for m in meta])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for trval_idx, test_idx in skf.split(paths, y):
        folds.append((paths[trval_idx].tolist(), paths[test_idx].tolist()))
    return folds


def save_split(split: Split, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(split), f, indent=2)


def load_split(path: Path) -> Split:
    with path.open() as f:
        d = json.load(f)
    return Split(**d)


def split_summary(split: Split, meta: List[ClipMeta]) -> Dict[str, Dict[str, int]]:
    by_path = {m.path: m for m in meta}
    out = {}
    for name, paths in (("train", split.train), ("val", split.val), ("test", split.test)):
        counts = {c: 0 for c in CLASS_NAMES}
        for p in paths:
            if p in by_path:
                counts[by_path[p].class_name] += 1
            else:
                # chunk-level path: infer class from parent dir name
                from pathlib import Path as _P
                cn = _P(p).parent.name
                if cn in counts:
                    counts[cn] += 1
        counts["total"] = len(paths)
        out[name] = counts
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    meta = load_manifest(Path(args.manifest))
    sp = paper_split(meta, seed=args.seed)
    save_split(sp, Path(args.out))
    print(json.dumps(split_summary(sp, meta), indent=2))
