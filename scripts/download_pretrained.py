"""Pre-download HF model weights (AST, HuBERT, wav2vec2) and EfficientNet
into HF_HOME and TORCH_HOME. Run from a CPU burst job once.

This is the safest way to handle slow / firewalled HF Hub access on compute
nodes - once cached, subsequent train jobs hit local disk only.
"""
from __future__ import annotations

import os
import time


def t(label, fn):
    t0 = time.time()
    print(f"[dl] start {label}", flush=True)
    fn()
    print(f"[dl] done  {label} in {time.time() - t0:.1f}s", flush=True)


def main():
    print("[dl] HF_HOME    =", os.environ.get("HF_HOME"), flush=True)
    print("[dl] TORCH_HOME =", os.environ.get("TORCH_HOME"), flush=True)

    def _ast():
        from transformers import ASTModel, ASTFeatureExtractor
        _ = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        _ = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def _hub():
        from transformers import HubertModel
        _ = HubertModel.from_pretrained("facebook/hubert-base-ls960")

    def _w2v():
        from transformers import Wav2Vec2Model
        _ = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def _eff_b0():
        import timm
        _ = timm.create_model("efficientnet_b0", pretrained=True, in_chans=1, num_classes=4)

    def _eff_b2():
        import timm
        _ = timm.create_model("efficientnet_b2", pretrained=True, in_chans=1, num_classes=4)

    def _rn50():
        import timm
        _ = timm.create_model("resnet50", pretrained=True, in_chans=1, num_classes=4)

    for label, fn in [
        ("ast",            _ast),
        ("hubert-base",    _hub),
        ("wav2vec2-base",  _w2v),
        ("efficientnet_b0", _eff_b0),
        ("efficientnet_b2", _eff_b2),
        ("resnet50",       _rn50),
    ]:
        try:
            t(label, fn)
        except Exception as e:
            print(f"[dl] FAILED {label}: {e}", flush=True)


if __name__ == "__main__":
    main()
