from __future__ import annotations

import logging
import time
from typing import Optional

from arachne_x.prompt_compiler.templates import merge_avatar_defaults
from arachne_x.utils.prompt_enhancer import enhance_prompt_i2v, enhance_prompt_t2v

logger = logging.getLogger(__name__)

_AVATAR_MODES = frozenset({"ai2v", "at2v", "streaming_ai2v", "avc"})


def expand_with_openai(
    user_text: str,
    *,
    mode: str,
    image_path: Optional[str] = None,
    locale: str = "auto",
) -> tuple[str, float]:
    """
    Prod path: OpenAI chat completion (same stack as prompt_enhancer).
    Returns (expanded_positive, latency_ms).
    """
    t0 = time.perf_counter()
    text = (user_text or "").strip()
    if not text:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return "", latency_ms

    if mode in ("i2v", "ai2v", "avc") and image_path:
        expanded = enhance_prompt_i2v(image_path, text, force=True)
    else:
        expanded = enhance_prompt_t2v(text, force=True)

    pos, _ = merge_avatar_defaults(expanded, "", mode=mode, locale=locale)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "prompt_compiler openai mode=%s latency_ms=%.1f chars_in=%d chars_out=%d",
        mode,
        latency_ms,
        len(text),
        len(pos),
    )
    return pos, latency_ms
