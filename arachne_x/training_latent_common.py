"""
Shared helpers for ARACHNE-X latent training: sample validation and batch collation.
Used by ``scripts/train.py`` and ``scripts/train_lora_avatar.py``.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import torch


def validate_latent_sample(sample: Dict[str, Any], *, require_audio: bool, source: str = "") -> Dict[str, torch.Tensor]:
    required = {"latents", "prompt_embeds", "prompt_mask", "timesteps", "noise"}
    missing = required - set(sample.keys())
    if missing:
        raise KeyError(f"{source}missing keys: {sorted(missing)}")
    if require_audio and "audio_embs" not in sample:
        raise KeyError(f"{source}avatar training requires audio_embs")
    return sample  # type: ignore[return-value]


def collate_latent_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack per-field tensors into a batch dict (same layout as default DataLoader for tensor values)."""
    if not samples:
        raise ValueError("empty batch")
    out: Dict[str, torch.Tensor] = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        out[k] = torch.stack(vals, dim=0)
    return out


def decode_wds_sample_pt(sample: Dict[str, Any], *, require_audio: bool) -> Dict[str, torch.Tensor]:
    """Decode WebDataset sample with binary field ``sample.pt`` (torch.save bytes)."""
    raw = sample.get("sample.pt")
    if raw is None:
        raise KeyError("WebDataset sample missing sample.pt")
    buf = io.BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else io.BytesIO(bytes(raw))
    obj = torch.load(buf, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"sample.pt must be a dict, got {type(obj)}")
    return validate_latent_sample(obj, require_audio=require_audio, source="")
