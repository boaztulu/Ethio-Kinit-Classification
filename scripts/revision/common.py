"""Shared helpers for the revision analyses (CPU-friendly re-inference)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import EmirChunks, make_loader
from src.eval import load_ckpt

PROJ = Path(__file__).resolve().parents[2]
RESULTS = PROJ / "results"
CKPT = PROJ / "checkpoints"
SEEDS = (1234, 2024, 7777)
CLASSES = ["Tizita", "Bati", "Ambassel", "Anchihoye"]

# configurations reported in the main results table
MODELS = ["effnet_b0_mel", "effnet_b2_mel", "ast", "wav2vec2", "hubert",
          "ekm_mel", "ekm_mfcc"]


def pin_cpus(mask: Optional[int] = None) -> None:
    """Pin this process to a set of logical CPUs (Windows), e.g. 0xFFFF for
    the P-cores of a hybrid CPU. Reads EMIR_AFFINITY when mask is None."""
    mask = mask if mask is not None else int(os.environ.get("EMIR_AFFINITY", "0"), 0)
    if mask and os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(mask))


def work_dir() -> Path:
    return Path(os.environ["EMIR_WORK"])


def local_index() -> str:
    return str(work_dir() / "cache" / "index.json")


def split(name: str = "split_paper.json") -> Dict[str, List[str]]:
    return json.load(open(RESULTS / name))


def load_model(tag: str):
    model, mode, cfg, _ = load_ckpt(CKPT / tag / "best.pt")
    model.eval()
    return model, mode, cfg


@torch.no_grad()
def predict_segments(model, mode: str, sources: List[str],
                     raw_perturb: Optional[Callable] = None,
                     index: Optional[str] = None, batch_size: int = 32,
                     num_workers: int = 0, dataset_cls=EmirChunks
                     ) -> Dict[str, np.ndarray]:
    """Segment posteriors for every segment of `sources`.

    Returns dict with probs (N,4), y (N,), src (N,) in dataset order.
    """
    ds = dataset_cls(index or local_index(), sources, mode=mode,
                     raw_perturb=raw_perturb)
    dl = make_loader(ds, batch_size, shuffle=False, num_workers=num_workers)
    P, Y, S = [], [], []
    for x, y, s in dl:
        P.append(F.softmax(model(x), dim=1).numpy())
        Y.append(y.numpy())
        S.extend(list(s))
    return {"probs": np.concatenate(P), "y": np.concatenate(Y), "src": np.array(S)}


def pool_clips(seg: Dict[str, np.ndarray]) -> List[Dict]:
    """Mean-pool segment posteriors to one prediction per recording."""
    rows = []
    for s in dict.fromkeys(seg["src"].tolist()):
        m = seg["src"] == s
        p = seg["probs"][m].mean(0)
        rows.append({"source": s, "y": int(seg["y"][m][0]),
                     "pred": int(p.argmax()), "probs": p.tolist()})
    return rows


def acc(rows: List[Dict]) -> float:
    return float(np.mean([r["y"] == r["pred"] for r in rows]))


def stored_rows(tag: str) -> List[Dict]:
    return json.load(open(RESULTS / f"{tag}.json"))["test"]["rows"]


class FixedScaleChunks(EmirChunks):
    """Log-mel input mapped to [0, 1] by the fixed rule (S + 80) / 80 instead of
    the per-segment min-max. On clean audio both rules are identical (every
    segment spans exactly -80 to 0 dB), so a trained model sees the same clean
    input; under noise only the min-max rule adapts to the raised floor."""

    def __getitem__(self, idx):
        if self.mode != "mel":
            return super().__getitem__(idx)
        rec = self.items[idx]
        S = (self._load_mel(rec["mel_path"]) + 80.0) / 80.0
        return torch.from_numpy(S).unsqueeze(0), int(rec["class_idx"]), rec["source_path"]
