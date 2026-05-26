"""Safetensors persistence for audio-conditioning adapters."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from .adapter import AudioConditioningAdapter, AudioConditioningAdapterConfig


def save_audio_conditioning_adapter(
    adapter: AudioConditioningAdapter,
    path: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)

    payload_meta = {
        "version": 1,
        "timestamp": time.time(),
        "config": adapter.config.to_dict(),
    }
    if metadata:
        payload_meta.update(metadata)

    tensors = adapter.state_dict()
    save_file(tensors, path, metadata={k: json.dumps(v) if not isinstance(v, str) else v for k, v in payload_meta.items()})
    sidecar = Path(path).with_suffix(".meta.json")
    sidecar.write_text(json.dumps(payload_meta, indent=2), encoding="utf-8")
    return path


def load_audio_conditioning_adapter(
    path: str,
    *,
    device: str = "cpu",
    strict: bool = True,
) -> AudioConditioningAdapter:
    sidecar = Path(path).with_suffix(".meta.json")
    config: Optional[AudioConditioningAdapterConfig] = None
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if "config" in meta:
            config = AudioConditioningAdapterConfig.from_dict(meta["config"])

    adapter = AudioConditioningAdapter(config=config)
    state = load_file(path, device=device)
    incompatible = adapter.load_state_dict(state, strict=strict)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"Adapter load mismatch missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    return adapter
