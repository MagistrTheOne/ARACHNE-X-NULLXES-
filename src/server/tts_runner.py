"""
Orchestrator TTS seam.

The in-tree TTS backends (``arachne_x.tts`` / ``arachne_x.speech``) were removed.
Speech synthesis is now expected to come from an external service: either the
caller supplies pre-rendered PCM, or an external TTS endpoint is wired here.

Until an external backend is configured, ``synthesize_pcm_f32_16k`` raises so the
realtime loop degrades to ``text_only`` (see ``realtime_avatar_loop``) instead of
emitting silent/fake audio.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def synthesize_pcm_f32_16k(text: str, tts_cfg: Dict[str, Any]) -> np.ndarray:
    """
    Synthesize speech; returns float32 mono 16 kHz PCM.

    Empty/blank text returns an empty buffer. Any non-empty text raises because
    no TTS backend is wired — the caller is responsible for degraded handling.
    """
    text = (text or "").strip()
    if not text:
        return np.zeros((0,), dtype=np.float32)

    raise RuntimeError(
        "Internal TTS removed (arachne_x.tts / arachne_x.speech deleted). "
        "Supply pre-rendered PCM or wire an external TTS service in tts_runner."
    )
