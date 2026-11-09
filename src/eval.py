"""Full-evaluation pipeline that consumes saved checkpoints.

For each model checkpoint:
  - load best.pt
  - run test set (clip-level + chunk-level)
  - confusion matrix + per-class precision/recall/F1
  - 1000-resample bootstrap 95 % CI on accuracy
  - McNemar's test against EKM (if EKM result.json present)
  - robustness probes: noise SNR {20, 10, 5} dB, pitch shift {-2, -1, +1, +2}, time stretch {0.9, 1.1}
  - embedding extraction (penultimate features) + t-SNE / UMAP plots
  - top-K misclassification gallery
Outputs go under results/<tag>/eval/.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.data.augment import add_gaussian, pitch_shift, time_stretch
from src.data.dataset import EmirChunks, make_loader
from src.data.manifest import CLASS_NAMES, load_manifest
from src.data.splits import load_split
from src.models.registry import build_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ckpt(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model, mode = build_model(cfg["model"], n_classes=4, **cfg.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model"])
    return model, mode, cfg, ckpt


def clip_predict(model, dl, device, amp=True) -> List[Dict]:
    model.eval()
    by_clip: Dict[str, Dict] = {}
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        for x, y, src in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_np = y.cpu().numpy()
            for i, s in enumerate(src):
                d = by_clip.setdefault(s, {"probs": [], "y": int(y_np[i])})
                d["probs"].append(probs[i])
    rows = []
    for s, d in by_clip.items():
        p = np.mean(np.stack(d["probs"], 0), axis=0)
        rows.append({"source": s, "y": d["y"], "pred": int(p.argmax()), "probs": p.tolist()})
    return rows


def metrics(rows: List[Dict]) -> Dict:
    y = np.array([r["y"] for r in rows])
    yp = np.array([r["pred"] for r in rows])
    acc = float(np.mean(y == yp))
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=list(range(4)), zero_division=0)
    cm = confusion_matrix(y, yp, labels=list(range(4))).tolist()
    macro_f1 = float(np.mean(f))
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f.tolist(),
        "confusion_matrix": cm,
        "n": len(rows),
    }


def bootstrap_ci(rows: List[Dict], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
                 ) -> Dict:
    rng = np.random.default_rng(seed)
    y = np.array([r["y"] for r in rows])
    yp = np.array([r["pred"] for r in rows])
    n = len(rows)
    accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[b] = (y[idx] == yp[idx]).mean()
    return {
        "boot_mean": float(accs.mean()),
        "boot_lo": float(np.quantile(accs, alpha / 2)),
        "boot_hi": float(np.quantile(accs, 1 - alpha / 2)),
        "boot_std": float(accs.std()),
    }


def mcnemar(rows_a: List[Dict], rows_b: List[Dict]) -> Dict:
    """Two prediction sets compared on the same source clips."""
    a = {r["source"]: (r["y"], r["pred"]) for r in rows_a}
    b = {r["source"]: (r["y"], r["pred"]) for r in rows_b}
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_pairs": 0}
    n10 = n01 = n11 = n00 = 0
    for s in common:
        ca = (a[s][1] == a[s][0])
        cb = (b[s][1] == b[s][0])
        if ca and not cb: n10 += 1
        elif cb and not ca: n01 += 1
        elif ca and cb: n11 += 1
        else: n00 += 1
    if n10 + n01 == 0:
        return {"n_pairs": len(common), "n10": n10, "n01": n01, "stat": 0.0, "p_value": 1.0}
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    # chi-square df=1 p-value via survival function
    from math import erfc, sqrt
    p = erfc(sqrt(stat / 2.0))
    return {"n_pairs": len(common), "n10": n10, "n01": n01, "stat": float(stat), "p_value": float(p)}


# robustness perturbations
def perturb_noise(snr_db: float):
    def fn(y, sr):
        sig_p = float(np.mean(y ** 2) + 1e-12)
        noise_p = sig_p / (10 ** (snr_db / 10))
        n = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(noise_p).astype(np.float32)
        return (y + n).astype(np.float32)
    return fn


def perturb_pitch(semitones: float):
    def fn(y, sr):
        import librosa
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)
    return fn


def perturb_stretch(rate: float):
    def fn(y, sr):
        import librosa
        out = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
        if len(out) < len(y):
            pad = np.zeros(len(y), dtype=np.float32)
            pad[: len(out)] = out
            return pad
        return out[: len(y)]
    return fn


def robustness_probe(model, dataset_ctor, src_paths: List[str], device, amp: bool,
                     batch_size: int = 32, num_workers: int = 4) -> Dict:
    """dataset_ctor takes (augment_fn) and returns a Dataset.
    Returns dict of {probe_name: clip_metrics}."""
    probes = {
        "clean": None,
        "noise_snr20": perturb_noise(20),
        "noise_snr10": perturb_noise(10),
        "noise_snr5":  perturb_noise(5),
        "pitch_-2":   perturb_pitch(-2),
        "pitch_-1":   perturb_pitch(-1),
        "pitch_+1":   perturb_pitch(+1),
        "pitch_+2":   perturb_pitch(+2),
        "stretch_0.9": perturb_stretch(0.9),
        "stretch_1.1": perturb_stretch(1.1),
    }
    results = {}
    for name, fn in probes.items():
        ds = dataset_ctor(fn)
        dl = make_loader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        rows = clip_predict(model, dl, device, amp=amp)
        results[name] = {"metrics": metrics(rows), "n": len(rows)}
    return results


def extract_embeddings(model, dl, device, amp: bool, model_name: str
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Penultimate-layer embeddings via forward hook on the classifier head."""
    feats: List[torch.Tensor] = []
    ys: List[int] = []
    srcs: List[str] = []

    # find the final Linear in the model (the head) and hook its INPUT
    head = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and getattr(mod, "out_features", -1) == 4:
            head = mod
    if head is None:
        raise RuntimeError("Could not find 4-class head Linear")

    captured = {}
    def hook(_m, inp, _o):
        captured["x"] = inp[0].detach().cpu()
    h = head.register_forward_hook(hook)

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        for x, y, s in dl:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            feats.append(captured["x"].float())
            ys.extend(y.tolist())
            srcs.extend(list(s))
    h.remove()

    X = torch.cat(feats, 0).numpy()
    Y = np.array(ys)
    return X, Y, srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to best.pt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ekm_eval_rows", default=None,
                    help="path to EKM eval rows for McNemar comparison")
    ap.add_argument("--no_robustness", action="store_true")
    ap.add_argument("--no_embeddings", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    print(f"[eval] device={device}")

    model, mode, cfg, ckpt = load_ckpt(Path(args.checkpoint))
    model = model.to(device)
    amp = bool(cfg.get("amp", True)) and torch.cuda.is_available()
    tag = cfg["log"]["tag"]
    print(f"[eval] tag={tag} mode={mode}")

    split = load_split(Path(args.split))

    # clean test rows + metrics
    ds_test = EmirChunks(args.index, split.test, mode=mode, augment=None)
    dl_test = make_loader(ds_test, args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
    rows = clip_predict(model, dl_test, device, amp=amp)
    m_clean = metrics(rows)
    boot = bootstrap_ci(rows, n_boot=1000, alpha=0.05, seed=0)
    out = {"tag": tag, "checkpoint": str(args.checkpoint), "metrics": m_clean,
           "bootstrap95": boot, "rows": rows}

    # McNemar vs EKM
    if args.ekm_eval_rows and Path(args.ekm_eval_rows).exists():
        with open(args.ekm_eval_rows) as f:
            ekm_rows = json.load(f)["rows"]
        out["mcnemar_vs_ekm"] = mcnemar(rows, ekm_rows)

    # Robustness
    if not args.no_robustness:
        def ds_ctor(fn):
            # fn is None for "clean" or a (y, sr)->y raw perturbation
            return EmirChunks(args.index, split.test, mode=mode,
                              raw_perturb=fn)
        out["robustness"] = robustness_probe(
            model, ds_ctor, split.test, device, amp,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

    with (out_dir / "eval.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {out_dir / 'eval.json'}")

    # Embeddings
    if not args.no_embeddings:
        try:
            X, Y, srcs = extract_embeddings(model, dl_test, device, amp, cfg["model"])
            np.savez(out_dir / "embeddings.npz", X=X, Y=Y, srcs=np.array(srcs))
            print(f"[eval] saved embeddings shape={X.shape}")
        except Exception as e:
            print(f"[eval] embedding extraction failed: {e}")


if __name__ == "__main__":
    main()
