"""Build a manifest of every WAV in EMIR_Dataset_V1 with (path, class, duration, sr)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import soundfile as sf

CLASS_NAMES = ["Tizita", "Bati", "Ambassel", "Anchihoye"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


@dataclass
class ClipMeta:
    path: str
    class_name: str
    class_idx: int
    duration_sec: float
    samplerate: int
    n_samples: int
    n_channels: int


def iter_wavs(root: Path) -> Iterable[Path]:
    for cls in CLASS_NAMES:
        d = root / cls
        if not d.is_dir():
            raise FileNotFoundError(f"Missing class folder: {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() == ".wav":
                yield p


def build_manifest(root: Path) -> List[ClipMeta]:
    meta = []
    for p in iter_wavs(root):
        try:
            info = sf.info(str(p))
        except Exception as e:
            raise RuntimeError(f"Could not read {p}: {e}")
        cls = p.parent.name
        meta.append(
            ClipMeta(
                path=str(p),
                class_name=cls,
                class_idx=CLASS_TO_IDX[cls],
                duration_sec=info.duration,
                samplerate=info.samplerate,
                n_samples=info.frames,
                n_channels=info.channels,
            )
        )
    return meta


def save_manifest(meta: List[ClipMeta], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([asdict(m) for m in meta], f, indent=2)


def load_manifest(path: Path) -> List[ClipMeta]:
    with path.open() as f:
        rows = json.load(f)
    return [ClipMeta(**r) for r in rows]
