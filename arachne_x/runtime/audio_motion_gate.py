"""
Silence-aware scaling for audio CFG (Stability OS).

Reduces ``audio_guidance_scale`` when the audio conditioning window is near-silent
so the face latent does not wander between speech segments.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def _temporal_rms(audio_emb: torch.Tensor) -> float:
    x = audio_emb.detach().float()
    if x.numel() == 0:
        return 0.0
    if x.dim() >= 3:
        # [B, T, ...] — aggregate over batch and feature dims
        while x.dim() > 2:
            x = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)
        per_t = x.pow(2).mean(dim=0).sqrt()
        return float(per_t.mean().item())
    return float(x.pow(2).mean().sqrt().item())


def apply_audio_motion_gate(
    audio_emb: torch.Tensor,
    base_audio_guidance_scale: float,
    *,
    silence_rms_threshold: float = 0.03,
    min_scale_ratio: float = 0.18,
    enabled: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Return ``(effective_audio_guidance_scale, metadata)``.
    """
    base = float(base_audio_guidance_scale)
    meta: Dict[str, Any] = {
        "audio_guidance_scale_base": base,
        "silence_gate_enabled": bool(enabled),
    }
    if not enabled:
        meta["audio_guidance_scale_effective"] = base
        meta["silence_ratio"] = 0.0
        return base, meta

    rms = _temporal_rms(audio_emb)
    meta["audio_emb_rms"] = round(rms, 6)
    if rms >= silence_rms_threshold:
        meta["audio_guidance_scale_effective"] = base
        meta["silence_ratio"] = 0.0
        return base, meta

    ratio = max(min_scale_ratio, rms / max(silence_rms_threshold, 1e-8))
    effective = base * ratio
    meta["audio_guidance_scale_effective"] = round(effective, 4)
    meta["silence_ratio"] = round(1.0 - ratio, 4)
    return effective, meta
