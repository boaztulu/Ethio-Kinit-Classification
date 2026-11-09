"""Quick post-env smoke test: imports, dataset listing, model forward pass."""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    print(f"[sanity] python  : {sys.version.split()[0]}")
    import torch
    print(f"[sanity] torch   : {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[sanity] gpu     : {torch.cuda.get_device_name(0)}")

    import numpy, scipy, librosa, transformers, timm, soundfile, pyloudnorm
    print(f"[sanity] numpy   : {numpy.__version__}")
    print(f"[sanity] scipy   : {scipy.__version__}")
    print(f"[sanity] librosa : {librosa.__version__}")
    print(f"[sanity] HF      : {transformers.__version__}")
    print(f"[sanity] timm    : {timm.__version__}")

    # data
    from src.data.manifest import build_manifest
    root = Path("EMIR_Dataset_V1")
    meta = build_manifest(root)
    from collections import Counter
    c = Counter(m.class_name for m in meta)
    print(f"[sanity] dataset : {len(meta)} clips, classes={dict(c)}")
    assert len(meta) == 600, f"expected 600 clips, got {len(meta)}"
    assert c["Tizita"] == 162 and c["Bati"] == 144 and c["Ambassel"] == 147 and c["Anchihoye"] == 147

    # EKM forward pass
    from src.models.ekm import EKM, n_params
    m = EKM(n_classes=4)
    x = torch.randn(2, 1, 13, 301)
    out = m(x)
    assert out.shape == (2, 4)
    print(f"[sanity] EKM     : output {tuple(out.shape)} params {n_params(m):,}")

    # EfficientNet
    from src.models.spec_cnn import TimmSpecCNN
    enet = TimmSpecCNN("efficientnet_b0", n_classes=4, pretrained=False)
    out = enet(torch.randn(2, 1, 128, 301))
    print(f"[sanity] effnetb0: output {tuple(out.shape)}")

    print("[sanity] OK")


if __name__ == "__main__":
    main()
