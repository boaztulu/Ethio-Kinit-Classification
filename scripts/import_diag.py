"""Diagnostic: time individual imports to identify what hangs on the cluster."""
from __future__ import annotations
import time, sys

def t(label, code):
    t0 = time.time()
    sys.stdout.flush()
    exec(code, globals())
    dt = time.time() - t0
    print(f"[diag] {label}: {dt:.2f}s", flush=True)

print(f"[diag] python {sys.version.split()[0]}", flush=True)
t("torch",        "import torch")
t("torchaudio",   "import torchaudio")
t("numpy",        "import numpy")
t("scipy",        "import scipy")
t("librosa",      "import librosa")
t("transformers", "import transformers")
t("timm",         "import timm")
t("ASTConfig",    "from transformers import ASTConfig, ASTModel, ASTFeatureExtractor")
t("HubertModel",  "from transformers import HubertModel, HubertConfig")
t("create effnet (no pretrained)",  "import timm; m = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=4)")
t("cuda",         "import torch; x = torch.zeros(1).cuda(); print('cuda ok', x.device)")
print("[diag] DONE", flush=True)
