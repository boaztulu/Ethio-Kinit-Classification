"""Gradio web app for EMIR Kiñit classification.

Usage (inside emir-env):
    python -m web.app --checkpoint checkpoints/effnet_b0_mel_s1234/best.pt \
                      --share --port 7860

Or with the 3-seed EfficientNet ensemble:
    python -m web.app --ensemble \
        checkpoints/effnet_b0_mel_s1234/best.pt \
        checkpoints/effnet_b0_mel_s2024/best.pt \
        checkpoints/effnet_b0_mel_s7777/best.pt \
        --share

Launch via SLURM with `slurm/08_webapp.sbatch`.
"""
from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.manifest import CLASS_NAMES
from src.inference import KinitEnsemble, KinitInferer
from src.utils.style import CLASS_COLORS, apply_style

# Starlette 1.2.x changed the TemplateResponse signature from
#     TemplateResponse(name, context, ...)        # old (Starlette <0.40)
# to  TemplateResponse(request, name, context, ...)  # new (Starlette 1.x)
# Gradio 4.44.x still uses the old signature, so on the new Starlette its
# index-page request crashes with `TypeError: unhashable type: 'dict'`
# (the context dict gets passed as the template name and ends up in the
# Jinja2 LRU cache). Patch the method to accept the old call shape and
# remap args before Gradio binds it.
def _patch_old_template_response() -> None:
    from starlette.requests import Request
    from starlette.templating import Jinja2Templates
    _orig = Jinja2Templates.TemplateResponse

    def _patched(self, *args, **kwargs):
        if args and not isinstance(args[0], Request):
            first = args[0]
            if isinstance(first, str):
                name = first
                if len(args) >= 2 and isinstance(args[1], dict):
                    ctx = args[1]
                    rest = args[2:]
                else:
                    ctx = kwargs.pop("context", {}) or {}
                    rest = args[1:]
                req = ctx.get("request") if isinstance(ctx, dict) else None
                return _orig(self, req, name, ctx, *rest, **kwargs)
        return _orig(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = _patched

_patch_old_template_response()

try:
    import gradio as gr
    # Workaround for Gradio 4.44.x bug: url_ok() probe can fail on compute
    # nodes even when the server is reachable from the same host, causing
    # `launch(share=False)` to refuse to start. Patch it to always return True.
    from gradio import networking as _networking
    _networking.url_ok = lambda url: True
except ImportError:
    raise SystemExit("Gradio is not installed; run `pip install gradio` inside the emir-env.")

apply_style()

# Will be populated by main()
INFERER = None


def _bar_plot(probs: dict) -> "np.ndarray":
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    classes = list(CLASS_NAMES)
    vals = [probs[c] for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]
    ax.bar(classes, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("probability")
    ax.set_title("clip-level Kiñit posterior")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    import PIL.Image
    return np.array(PIL.Image.open(buf))


def _chunk_plot(per_chunk):
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    n = len(per_chunk)
    starts = [c["start_sec"] for c in per_chunk]
    for cname in CLASS_NAMES:
        ys = [c["probs"][cname] for c in per_chunk]
        ax.plot(starts, ys, marker="o", label=cname, color=CLASS_COLORS[cname])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("probability")
    ax.set_title("per-chunk posteriors")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    import PIL.Image
    return np.array(PIL.Image.open(buf))


def predict(audio):
    """Gradio audio input is (sample_rate, ndarray) or a filepath string."""
    if audio is None:
        return ("⚠️  Upload audio or record from your microphone.", None, None, "")
    if isinstance(audio, str):
        out = INFERER.predict_file(audio)
    else:
        sr, y = audio
        out = INFERER.predict_audio(y.astype(np.float32), sr)
    if "error" in out:
        return (f"⚠️  {out['error']}", None, None, "")
    pred = out["prediction"]
    conf = out["confidence"] * 100
    headline = f"## **{pred}**  ({conf:.1f}% confidence)\n\n"
    headline += f"clip duration {out['duration_sec']:.1f}s · {out['n_chunks']} chunks · model `{out['model']}`"
    if "per_member" in out:
        headline += "\n\n**Per-member predictions:**\n"
        for tag, m in out["per_member"].items():
            headline += f"- `{tag}` → {m['prediction']} ({m['confidence']*100:.1f}%)\n"
    bar = _bar_plot(out["probs"])
    chunk_img = _chunk_plot(out["per_chunk"])
    rows_md = "| start–end (s) | predicted | conf | top probs |\n|---|---|---:|---|\n"
    for c in out["per_chunk"]:
        top = sorted(c["probs"].items(), key=lambda x: -x[1])[:2]
        top_s = " · ".join(f"{k} {v*100:.0f}%" for k, v in top)
        rows_md += f"| {c['start_sec']:.1f}–{c['end_sec']:.1f} | {c['pred']} | {c['probs'][c['pred']]*100:.0f}% | {top_s} |\n"
    return headline, bar, chunk_img, rows_md


def build_ui():
    with gr.Blocks(title="EMIR Kiñit Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Ethiopian Music: Kiñit Classifier\n"
            "Upload or record a clip of Ethiopian music. The model classifies it into "
            "one of the four pentatonic Kiñits: **Tizita**, **Bati**, **Ambassel**, **Anchihoye**.\n\n"
            "Built on the EMIR v1.0 dataset (Retta et al., PLOS ONE 2023)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(label="Audio clip (any sample rate; will be resampled to 16 kHz)",
                                 type="numpy", sources=["upload", "microphone"])
                btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Markdown()
                bar = gr.Image(label="Class probabilities", show_label=False)
        chunk_plot = gr.Image(label="Per-chunk posteriors over time", show_label=True)
        chunk_table = gr.Markdown()

        btn.click(predict, inputs=audio,
                  outputs=[summary, bar, chunk_plot, chunk_table])

        # Example clips (one per class) if EMIR is present
        emir_root = Path("EMIR_Dataset_V1")
        if emir_root.is_dir():
            examples = []
            for c in CLASS_NAMES:
                d = emir_root / c
                if d.is_dir():
                    files = sorted(d.glob("*.wav"))
                    if files:
                        examples.append([str(files[0])])
            if examples:
                gr.Examples(examples=examples, inputs=audio, label="Sample clips (one per class)")

        gr.Markdown(
            "---\n"
            f"Model: **{INFERER.cfg['model'] if hasattr(INFERER,'cfg') else 'ensemble'}**.\n"
            "Code: `src/inference.py` · `web/app.py`."
        )
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/effnet_b0_mel_s1234/best.pt",
                    help="Single-model checkpoint")
    ap.add_argument("--ensemble", nargs="+", default=None,
                    help="Multiple checkpoints to ensemble (overrides --checkpoint)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Use Gradio share link (public)")
    ap.add_argument("--server_name", default="0.0.0.0")
    args = ap.parse_args()

    global INFERER
    if args.ensemble:
        print(f"[app] loading ensemble of {len(args.ensemble)} models …", flush=True)
        INFERER = KinitEnsemble(args.ensemble)
        INFERER.cfg = {"model": "ensemble"}  # for UI
    else:
        print(f"[app] loading checkpoint {args.checkpoint}", flush=True)
        INFERER = KinitInferer(args.checkpoint)
    print("[app] ready", flush=True)

    demo = build_ui()
    demo.launch(server_name=args.server_name, server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
