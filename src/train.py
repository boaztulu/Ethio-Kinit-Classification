"""Generic training driver.

Usage:
    python -m src.train --config configs/<name>.yaml [--seed N] [--fold K] [--out_dir ...]

A config has:
  model:        registry name (string)
  model_kwargs: dict
  data:         { manifest, index, split_path, win_sec, batch_size, num_workers }
  augment:      { raw: {...}, mel: {...}, mixup_alpha }
  optim:        { name, lr, weight_decay, warmup_epochs, epochs, label_smoothing,
                  use_class_weight, sampler }
  amp:          bool
  early_stop_patience: int
  log:          { tag }
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.augment import build_mel_augment, build_raw_augment, mixup_batch
from src.data.dataset import EmirChunks, class_balanced_sampler, make_loader
from src.data.manifest import CLASS_NAMES, load_manifest
from src.data.splits import load_split, paper_split, save_split, split_summary
from src.models.registry import build_model
from src.utils.seed import set_seed


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _epoch(loader: DataLoader, model: nn.Module, optim, sched,
           device: torch.device, train: bool,
           label_smoothing: float, mixup_alpha: float, amp: bool, scaler):
    model.train(train)
    losses, total, correct = 0.0, 0, 0
    n_classes = None
    all_logits, all_y = [], []
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    for batch in loader:
        x, y, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if n_classes is None:
            n_classes = 4  # fixed for EMIR
        if train and mixup_alpha > 0:
            x, ya, yb, lam = mixup_batch(x, y, alpha=mixup_alpha)
        else:
            ya = yb = y
            lam = 1.0

        if train:
            optim.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)

        if train:
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            if sched is not None:
                sched.step()

        losses += loss.item() * x.size(0)
        total += x.size(0)
        if not train:
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            all_logits.append(logits.detach().cpu())
            all_y.append(y.detach().cpu())

    avg_loss = losses / max(total, 1)
    out = {"loss": avg_loss}
    if not train:
        all_logits = torch.cat(all_logits)
        all_y = torch.cat(all_y)
        out["chunk_acc"] = correct / max(total, 1)
        out["logits"] = all_logits
        out["y"] = all_y
    return out


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int = 4) -> float:
    f1s = []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def clip_level_eval(loader: DataLoader, model: nn.Module, device: torch.device,
                    amp: bool) -> Dict:
    """Aggregate chunk logits to clip-level by mean-pooling per source_path."""
    model.eval()
    by_clip: Dict[str, Dict[str, list]] = {}
    with torch.no_grad():
        for x, y, src in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y = y.cpu().numpy()
            for i, s in enumerate(src):
                d = by_clip.setdefault(s, {"probs": [], "y": int(y[i])})
                d["probs"].append(probs[i])
    rows = []
    for s, d in by_clip.items():
        p = np.mean(np.stack(d["probs"], 0), axis=0)
        pred = int(p.argmax())
        rows.append({"source": s, "y": d["y"], "pred": pred, "probs": p.tolist()})
    y_true = torch.tensor([r["y"] for r in rows])
    y_pred = torch.tensor([r["pred"] for r in rows])
    acc = float((y_true == y_pred).float().mean().item())
    f1 = macro_f1(y_true, y_pred, 4)
    return {"clip_acc": acc, "clip_macro_f1": f1, "n_clips": len(rows), "rows": rows}


def build_dataloaders(cfg: dict, split, augment_raw, augment_mel, mode: str
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    idx = cfg["data"]["index"]
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)

    ds_train = EmirChunks(idx, split.train, mode=mode, augment=augment_raw if mode == "raw"
                          else augment_mel if mode == "mel" else augment_raw)
    ds_val = EmirChunks(idx, split.val, mode=mode, augment=None)
    ds_test = EmirChunks(idx, split.test, mode=mode, augment=None)

    sampler = None
    if cfg["optim"].get("sampler") == "class_balanced":
        sampler = class_balanced_sampler(ds_train, list(range(4)))
    dl_train = make_loader(ds_train, bs, shuffle=True, num_workers=nw,
                           drop_last=True, weighted_sampler=sampler)
    dl_val = make_loader(ds_val, bs, shuffle=False, num_workers=nw)
    dl_test = make_loader(ds_test, bs, shuffle=False, num_workers=nw)
    return dl_train, dl_val, dl_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="overrides cfg.seed if set")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--tag_suffix", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else cfg.get("seed", 1234)
    set_seed(seed)

    tag = cfg["log"]["tag"] + f"_s{seed}" + args.tag_suffix
    out_dir = Path(args.out_dir or f"checkpoints/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    device = _device()
    print(f"[train] tag={tag} device={device} seed={seed}")

    # splits
    sp_path = Path(cfg["data"]["split_path"])
    if sp_path.exists():
        split = load_split(sp_path)
    else:
        meta = load_manifest(Path(cfg["data"]["manifest"]))
        split = paper_split(meta, seed=cfg.get("split_seed", 1234))
        save_split(split, sp_path)
    print("[train] split sizes:",
          {k: len(v) for k, v in [("train", split.train), ("val", split.val), ("test", split.test)]})

    # model
    model, mode = build_model(cfg["model"], n_classes=4, **cfg.get("model_kwargs", {}))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] model={cfg['model']} mode={mode} n_params={n_params:,}")

    # augment
    aug_cfg = cfg.get("augment", {})
    augment_raw = build_raw_augment(aug_cfg.get("raw", {})) if aug_cfg.get("raw") else None
    augment_mel = build_mel_augment(aug_cfg.get("mel", {})) if aug_cfg.get("mel") else None

    # loaders
    dl_train, dl_val, dl_test = build_dataloaders(cfg, split, augment_raw, augment_mel, mode)

    # optim
    optim_cfg = cfg["optim"]
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )
    steps_per_epoch = len(dl_train)
    epochs = int(optim_cfg["epochs"])
    warmup_epochs = float(optim_cfg.get("warmup_epochs", 2))
    total_steps = steps_per_epoch * epochs

    def lr_lambda(step):
        warm_steps = max(1, int(warmup_epochs * steps_per_epoch))
        if step < warm_steps:
            return step / warm_steps
        prog = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    amp = bool(cfg.get("amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # train loop
    best = {"epoch": -1, "val_macro_f1": -1.0, "val_chunk_acc": -1.0,
            "val_clip_acc": -1.0, "val_clip_f1": -1.0}
    history = []
    patience = int(cfg.get("early_stop_patience", 12))
    bad = 0
    label_smoothing = float(optim_cfg.get("label_smoothing", 0.1))
    mixup_alpha = float(aug_cfg.get("mixup_alpha", 0.0))

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        et = time.time()
        tr = _epoch(dl_train, model, optim, sched, device, True,
                    label_smoothing, mixup_alpha, amp, scaler)
        with torch.no_grad():
            va = _epoch(dl_val, model, optim, None, device, False,
                        label_smoothing, 0.0, amp, scaler)
        val_f1 = macro_f1(va["y"], va["logits"].argmax(1), 4)
        clip_eval = clip_level_eval(dl_val, model, device, amp)
        rec = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_chunk_acc": va["chunk_acc"],
            "val_chunk_f1": val_f1,
            "val_clip_acc": clip_eval["clip_acc"],
            "val_clip_f1": clip_eval["clip_macro_f1"],
            "lr": optim.param_groups[0]["lr"],
            "time": time.time() - et,
        }
        history.append(rec)
        print(f"[ep {epoch:03d}] tr_loss={tr['loss']:.4f} va_loss={va['loss']:.4f} "
              f"va_chunk_acc={va['chunk_acc']:.4f} va_clip_acc={clip_eval['clip_acc']:.4f} "
              f"va_clip_f1={clip_eval['clip_macro_f1']:.4f} t={rec['time']:.1f}s")

        score = clip_eval["clip_macro_f1"]
        if score > best["val_clip_f1"]:
            best.update({
                "epoch": epoch,
                "val_chunk_acc": va["chunk_acc"],
                "val_macro_f1": val_f1,
                "val_clip_acc": clip_eval["clip_acc"],
                "val_clip_f1": clip_eval["clip_macro_f1"],
            })
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best": best,
                "cfg": cfg,
                "seed": seed,
                "mode": mode,
            }, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[early stop] {patience} epochs without improvement")
                break

    train_secs = time.time() - t0

    # final test
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_chunk = _epoch(dl_test, model, optim, None, device, False,
                        label_smoothing, 0.0, amp, scaler)
    test_chunk_f1 = macro_f1(test_chunk["y"], test_chunk["logits"].argmax(1), 4)
    test_clip = clip_level_eval(dl_test, model, device, amp)
    print(f"[TEST] chunk_acc={test_chunk['chunk_acc']:.4f} chunk_f1={test_chunk_f1:.4f} "
          f"clip_acc={test_clip['clip_acc']:.4f} clip_f1={test_clip['clip_macro_f1']:.4f} "
          f"train_time={train_secs:.1f}s")

    result = {
        "tag": tag,
        "model": cfg["model"],
        "seed": seed,
        "n_params": n_params,
        "best_val": best,
        "test": {
            "chunk_acc": test_chunk["chunk_acc"],
            "chunk_macro_f1": test_chunk_f1,
            "clip_acc": test_clip["clip_acc"],
            "clip_macro_f1": test_clip["clip_macro_f1"],
            "n_clips": test_clip["n_clips"],
            "rows": test_clip["rows"],
        },
        "history": history,
        "train_seconds": train_secs,
        "split_summary": split_summary(split, load_manifest(Path(cfg["data"]["manifest"]))),
    }
    with (out_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (Path("results") / f"{tag}.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[train] wrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
