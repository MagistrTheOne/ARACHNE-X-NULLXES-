"""faster-whisper ASR (loads model once per process)."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model = None
_model_key: Optional[tuple[Any, ...]] = None


def _get_model(cfg: Dict[str, Any]):
    global _model, _model_key
    model_id = str(cfg.get("model_id") or "base")
    device = str(cfg.get("device") or "cpu")
    compute_type = str(cfg.get("compute_type") or "int8")
    language = cfg.get("language")
    key = (model_id, device, compute_type, language)
    with _model_lock:
        if _model is not None and _model_key == key:
            return _model
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
        )
        _model_key = key
        logger.info("faster-whisper loaded model_id=%s device=%s", model_id, device)
        return _model


def transcribe_f32_mono_16k(
    samples: np.ndarray,
    cfg: Dict[str, Any],
) -> str:
    """
    Transcribe float32 mono PCM at 16 kHz. Returns stripped text or empty string.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size < 400:
        return ""
    model = _get_model(cfg)
    kwargs: Dict[str, Any] = {
        "beam_size": int(cfg.get("beam_size", 1)),
        "vad_filter": bool(cfg.get("vad_filter", False)),
    }
    lang = cfg.get("language")
    if lang:
        kwargs["language"] = str(lang)
    segments, _info = model.transcribe(x, **kwargs)
    parts = [s.text.strip() for s in segments if s.text and s.text.strip()]
    return " ".join(parts).strip()
