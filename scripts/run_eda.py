"""Phase 1 - Data verification & EDA.

Reads EMIR_Dataset_V1/, writes:
  results/manifest.json               full per-clip metadata
  results/data_stats.json             aggregate stats
  data_card.md                        human-readable summary
  figures/eda_duration_hist.{png,pdf} duration histogram per class
  figures/eda_class_dist.{png,pdf}    class count bar chart
  figures/eda_rms.{png,pdf}           per-class RMS box plot
  figures/eda_samples.{png,pdf}       waveform + mel-spec for 2 samples per class
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.data.manifest import CLASS_NAMES, build_manifest, save_manifest
from src.utils.seed import set_seed
from src.utils.style import CLASS_COLORS, apply_style


def fmt_dur(s: float) -> str:
    m = int(s // 60)
    sec = s - 60 * m
    return f"{m:d}:{sec:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="EMIR_Dataset_V1")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    apply_style()

    root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    results_dir = out_dir / "results"
    figs_dir = out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EDA] Building manifest from {root}")
    meta = build_manifest(root)
    save_manifest(meta, results_dir / "manifest.json")
    print(f"[EDA] {len(meta)} clips total")

    # per-class stats
    by_class = {c: [m for m in meta if m.class_name == c] for c in CLASS_NAMES}
    sr_set = sorted({m.samplerate for m in meta})
    ch_set = sorted({m.n_channels for m in meta})

    stats = {
        "n_total": len(meta),
        "samplerates_observed": sr_set,
        "channels_observed": ch_set,
        "per_class": {},
    }
    for c, items in by_class.items():
        durs = np.array([m.duration_sec for m in items], dtype=np.float64)
        stats["per_class"][c] = {
            "count": len(items),
            "duration_total_sec": float(durs.sum()),
            "duration_mean_sec": float(durs.mean()),
            "duration_std_sec": float(durs.std()),
            "duration_min_sec": float(durs.min()),
            "duration_max_sec": float(durs.max()),
        }

    # RMS energy (computed on the fly)
    rms_by_class = {c: [] for c in CLASS_NAMES}
    for m in meta:
        y, sr = sf.read(m.path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        rms_by_class[m.class_name].append(float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)))
    stats["rms_by_class"] = {
        c: {
            "mean": float(np.mean(rms_by_class[c])),
            "std": float(np.std(rms_by_class[c])),
            "min": float(np.min(rms_by_class[c])),
            "max": float(np.max(rms_by_class[c])),
        }
        for c in CLASS_NAMES
    }

    with (results_dir / "data_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # =================== FIGURES ===================
    # 1) class distribution
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    counts = [stats["per_class"][c]["count"] for c in CLASS_NAMES]
    ax.bar(CLASS_NAMES, counts, color=[CLASS_COLORS[c] for c in CLASS_NAMES])
    for i, v in enumerate(counts):
        ax.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("number of recordings")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_title("EMIR class distribution (N = 600)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"eda_class_dist.{ext}")
    plt.close(fig)

    # 2) duration histogram per class
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    all_durs = []
    for c in CLASS_NAMES:
        ds = [m.duration_sec for m in by_class[c]]
        all_durs.append(ds)
    ax.hist(all_durs, bins=20, stacked=False, label=CLASS_NAMES,
            color=[CLASS_COLORS[c] for c in CLASS_NAMES], alpha=0.75)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("count")
    ax.set_title("clip duration per class")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"eda_duration_hist.{ext}")
    plt.close(fig)

    # 3) RMS box plot
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    data = [rms_by_class[c] for c in CLASS_NAMES]
    bp = ax.boxplot(data, labels=CLASS_NAMES, patch_artist=True, widths=0.55)
    for patch, c in zip(bp["boxes"], CLASS_NAMES):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.65)
    ax.set_ylabel("RMS amplitude")
    ax.set_title("per-class RMS distribution")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"eda_rms.{ext}")
    plt.close(fig)

    # 4) waveform + mel-spec samples (2 per class)
    rng = np.random.default_rng(args.seed)
    n_per_class = 2
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class * 2,
                             figsize=(2.6 * n_per_class * 2, 1.7 * len(CLASS_NAMES)))
    for row, c in enumerate(CLASS_NAMES):
        clips = by_class[c]
        chosen = rng.choice(len(clips), size=n_per_class, replace=False)
        for col, ci in enumerate(chosen):
            m = clips[int(ci)]
            y, sr = sf.read(m.path, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            # take first 5 s for plotting
            n5 = min(len(y), int(5 * sr))
            y5 = y[:n5]
            t = np.arange(n5) / sr
            # waveform
            ax_w = axes[row, 2 * col]
            ax_w.plot(t, y5, color=CLASS_COLORS[c], linewidth=0.5)
            ax_w.set_xlim(0, n5 / sr)
            ax_w.set_ylim(-1, 1)
            if row == 0:
                ax_w.set_title("waveform")
            if col == 0:
                ax_w.set_ylabel(c, fontsize=10)
            ax_w.set_xticks([])
            ax_w.set_yticks([])
            # mel-spec
            ax_s = axes[row, 2 * col + 1]
            S = librosa.feature.melspectrogram(y=y5, sr=sr, n_mels=128, n_fft=400, hop_length=160)
            S_db = librosa.power_to_db(S, ref=np.max)
            ax_s.imshow(S_db, aspect="auto", origin="lower", cmap="magma")
            if row == 0:
                ax_s.set_title("log-mel spec")
            ax_s.set_xticks([])
            ax_s.set_yticks([])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"eda_samples.{ext}")
    plt.close(fig)

    # =================== data_card.md ===================
    lines = ["# EMIR Data Card",
             "",
             f"Generated from `{root}` with `scripts/run_eda.py`.",
             "",
             "## Summary",
             f"- Total clips: **{stats['n_total']}**",
             f"- Sample rates observed: {sr_set}",
             f"- Channels observed: {ch_set}",
             "",
             "## Per-class statistics",
             "",
             "| Class | Count | Total dur | Mean dur | Std | Min | Max |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in CLASS_NAMES:
        pc = stats["per_class"][c]
        lines.append(f"| {c} | {pc['count']} | {fmt_dur(pc['duration_total_sec'])} | "
                     f"{pc['duration_mean_sec']:.2f}s | {pc['duration_std_sec']:.2f}s | "
                     f"{pc['duration_min_sec']:.2f}s | {pc['duration_max_sec']:.2f}s |")
    lines.append("")
    lines.append("## RMS amplitude")
    lines.append("")
    lines.append("| Class | Mean | Std | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in CLASS_NAMES:
        r = stats["rms_by_class"][c]
        lines.append(f"| {c} | {r['mean']:.4f} | {r['std']:.4f} | {r['min']:.4f} | {r['max']:.4f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Source: Retta et al. 2023 (EMIR v1.0).")
    lines.append("- Split: recording-level stratified 70/10/20 partition in `results/split_paper.json`.")

    (out_dir / "data_card.md").write_text("\n".join(lines))

    print("[EDA] done.")


if __name__ == "__main__":
    main()
