"""Single-clip inference utility for the EMIR Kiñit classifier.

Two entry points:
  - KinitInferer(checkpoint).predict_file(wav_path) -> dict
  - KinitInferer(checkpoint).predict_audio(audio, sr) -> dict

The returned dict is JSON-serializable and contains:
  {
    "prediction": "Tizita",         # argmax class name
    "confidence": 0.93,              # softmax probability of argmax
    "probs": {"Tizita":..., "Bati":..., "Ambassel":..., "Anchihoye":...},
    "per_chunk": [{"start":..., "end":..., "probs":{...}}, ...],
    "n_chunks": 19,
    "duration_sec": 30.0,
  }

Inference policy:
  1. Resample to 16 kHz mono.
  2. Loudness-normalize to -23 LUFS (best effort).
  3. Trim silence (30 dB top).
  4. Chunk into 3-s windows with 50 % overlap.
  5. Forward each chunk through the model.
  6. Mean-pool softmax posteriors across chunks → clip-level prediction.

Supports either a single checkpoint or an *ensemble* of checkpoints
(KinitEnsemble) that averages posteriors across multiple models.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.data.manifest import CLASS_NAMES
from src.data.preprocess import (chunk, load_audio, logmel, loudness_normalize,
                                 trim_silence, TARGET_SR)
from src.models.registry import build_model


class KinitInferer:
    """Load a checkpoint and predict on a single audio clip."""

    def __init__(self, checkpoint_path: Union[str, Path], device: str = "auto"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        self.cfg = cfg
        self.model_name = cfg["model"]
        self.tag = cfg["log"]["tag"]
        model, mode = build_model(self.model_name, n_classes=4, **cfg.get("model_kwargs", {}))
        model.load_state_dict(ckpt["model"])
        model.eval()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.mode = mode
        self.sr = TARGET_SR
        self.win_sec = 3.0
        self.overlap = 0.5

    def _features_from_chunk(self, y_chunk: np.ndarray) -> torch.Tensor:
        if self.mode == "raw":
            x = torch.from_numpy(y_chunk).float().unsqueeze(0).unsqueeze(0)  # (1,1,T)
        elif self.mode == "mel":
            S = logmel(y_chunk, self.sr, n_mels=128)
            S = (S - S.min()) / (S.max() - S.min() + 1e-6)
            x = torch.from_numpy(S).float().unsqueeze(0).unsqueeze(0)        # (1,1,M,T)
        elif self.mode == "mfcc":
            import librosa
            M = librosa.feature.mfcc(y=y_chunk, sr=self.sr, n_mfcc=13,
                                     n_mels=40, n_fft=400, hop_length=160).astype(np.float32)
            M = (M - M.mean()) / (M.std() + 1e-6)
            x = torch.from_numpy(M).float().unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return x.to(self.device, non_blocking=True)

    @torch.no_grad()
    def _chunk_probs(self, chunks: List[np.ndarray]) -> np.ndarray:
        probs = []
        for ch in chunks:
            x = self._features_from_chunk(ch)
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                logits = self.model(x)
            p = F.softmax(logits, dim=1).float().squeeze(0).cpu().numpy()
            probs.append(p)
        return np.stack(probs, 0)

    def predict_audio(self, audio: np.ndarray, sr: int) -> Dict:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sr:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.sr)
        y = loudness_normalize(audio.astype(np.float32), self.sr)
        y = trim_silence(y, top_db=30.0)
        if len(y) < self.sr * 0.5:
            return {"prediction": None, "error": "audio too short after trimming",
                    "duration_sec": float(len(audio) / sr)}
        chunks = chunk(y, self.sr, win_sec=self.win_sec, overlap=self.overlap)
        probs = self._chunk_probs(chunks)            # (n_chunks, 4)
        mean = probs.mean(axis=0)
        cls = int(mean.argmax())
        hop_sec = self.win_sec * (1 - self.overlap)
        per_chunk = []
        for i, p in enumerate(probs):
            per_chunk.append({
                "start_sec": i * hop_sec,
                "end_sec": i * hop_sec + self.win_sec,
                "probs": {CLASS_NAMES[c]: float(p[c]) for c in range(4)},
                "pred": CLASS_NAMES[int(p.argmax())],
            })
        return {
            "prediction": CLASS_NAMES[cls],
            "confidence": float(mean[cls]),
            "probs": {CLASS_NAMES[c]: float(mean[c]) for c in range(4)},
            "per_chunk": per_chunk,
            "n_chunks": len(chunks),
            "duration_sec": float(len(y) / self.sr),
            "model": self.tag,
        }

    def predict_file(self, wav_path: Union[str, Path]) -> Dict:
        y = load_audio(str(wav_path), self.sr)
        return self.predict_audio(y, self.sr)


class KinitEnsemble:
    """Ensemble of multiple KinitInferers - averages softmax posteriors."""

    def __init__(self, checkpoints: List[Union[str, Path]], device: str = "auto"):
        self.members = [KinitInferer(c, device=device) for c in checkpoints]

    def predict_audio(self, audio: np.ndarray, sr: int) -> Dict:
        outs = [m.predict_audio(audio, sr) for m in self.members]
        if any("error" in o for o in outs):
            return outs[0]
        # average member clip-level probs
        prob_mat = np.stack([
            np.array([o["probs"][c] for c in CLASS_NAMES])
            for o in outs
        ], 0)
        mean = prob_mat.mean(axis=0)
        cls = int(mean.argmax())
        return {
            "prediction": CLASS_NAMES[cls],
            "confidence": float(mean[cls]),
            "probs": {CLASS_NAMES[c]: float(mean[c]) for c in range(4)},
            "per_chunk": outs[0]["per_chunk"],  # representative chunking
            "n_chunks": outs[0]["n_chunks"],
            "duration_sec": outs[0]["duration_sec"],
            "model": "ensemble(" + ",".join(m.tag for m in self.members) + ")",
            "per_member": {m.tag: {"prediction": o["prediction"], "confidence": o["confidence"]}
                           for m, o in zip(self.members, outs)},
        }

    def predict_file(self, wav_path: Union[str, Path]) -> Dict:
        from src.data.preprocess import load_audio
        y = load_audio(str(wav_path), TARGET_SR)
        return self.predict_audio(y, TARGET_SR)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/effnet_b0_mel_s1234/best.pt")
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()
    t0 = time.time()
    inf = KinitInferer(args.checkpoint)
    print(f"[infer] loaded model in {time.time()-t0:.1f}s on {inf.device}", flush=True)
    out = inf.predict_file(args.audio)
    print(json.dumps(out, indent=2))
