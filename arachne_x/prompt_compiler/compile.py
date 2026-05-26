from __future__ import annotations

import logging
import os
import time
from typing import Optional

from arachne_x.prompt_compiler.avatar_turn_plan import AvatarInferMode, AvatarTurnPlan, CompilerBackend
from arachne_x.prompt_compiler.templates import merge_avatar_defaults, truncate_for_log

logger = logging.getLogger(__name__)

_VALID_BACKENDS = frozenset({"off"})
_VALID_FALLBACK = frozenset({"off"})


def resolve_compiler_backend(cli_value: Optional[str] = None) -> CompilerBackend:
    """CLI flag overrides ARACHNE_PROMPT_COMPILER env; default off."""
    raw = (cli_value or os.environ.get("ARACHNE_PROMPT_COMPILER") or "off").strip().lower()
    if raw not in _VALID_BACKENDS:
        logger.warning("Prompt compiler backend %r is not shipped in this runtime; using off", raw)
        return "off"
    return raw  # type: ignore[return-value]


def resolve_compiler_fallback(cli_value: Optional[str] = None) -> CompilerBackend:
    raw = (cli_value or os.environ.get("ARACHNE_COMPILER_FALLBACK") or "off").strip().lower()
    if raw not in _VALID_FALLBACK:
        return "off"
    return raw  # type: ignore[return-value]


def compile_avatar_turn(
    user_text: str,
    *,
    mode: AvatarInferMode,
    image_path: Optional[str] = None,
    audio_duration_sec: Optional[float] = None,
    backend: CompilerBackend = "off",
    fallback: CompilerBackend = "off",
    locale: str = "auto",
    negative_prompt: str = "",
    emotion_id: Optional[int] = None,
) -> AvatarTurnPlan:
    """
    Compile user intent into UMT5-ready positive/negative strings.

    ``audio_duration_sec`` is reserved for Phase B planning (logged only today).
    """
    del audio_duration_sec  # Phase B hook

    t0 = time.perf_counter()
    source = (user_text or "").strip()
    neg_in = (negative_prompt or "").strip()
    del image_path, fallback
    effective_backend: CompilerBackend = "off"
    positive = source

    positive, negative = merge_avatar_defaults(
        positive,
        neg_in,
        mode=mode,
        locale=locale,
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "prompt_compiler backend=%s effective=%s latency_ms=%.1f "
        "chars_before=%d chars_after=%d pos_preview=%r",
        backend,
        effective_backend,
        latency_ms,
        len(source),
        len(positive),
        truncate_for_log(positive),
    )

    return AvatarTurnPlan(
        positive_prompt=positive,
        negative_prompt=negative,
        emotion_id=emotion_id,
        compiler_backend=effective_backend,
        compiler_latency_ms=latency_ms,
        source_user_text=source,
    )
