"""Wire prompt compiler into inference / serving (ARACHNE-X only)."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from arachne_x.inference_frames import audio_duration_sec
from arachne_x.prompt_compiler import compile_avatar_turn, resolve_compiler_backend
from arachne_x.prompt_compiler.compile import resolve_compiler_fallback


def _infer_mode_for_compiler(mode: str) -> str:
    allowed = (
        "ai2v",
        "at2v",
        "streaming_ai2v",
        "t2v",
        "i2v",
        "vc",
        "avc",
        "audio_i2v",
        "imagine_i2v",
    )
    return mode if mode in allowed else "ai2v"


def resolve_imagine_compiler_backend(args: argparse.Namespace) -> str:
    """Default Gemma for imagine_i2v when CLI/env compiler not set."""
    if getattr(args, "prompt_compiler", None):
        return resolve_compiler_backend(args.prompt_compiler)
    env_default = (os.environ.get("ARACHNE_IMAGINE_PROMPT_COMPILER") or "gemma").strip().lower()
    return resolve_compiler_backend(env_default)


def apply_prompt_compiler(
    args: argparse.Namespace,
    *,
    wav_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Mutate ``args.prompt`` / ``args.negative_prompt`` from compiled turn plan.
    Returns metadata dict for run-metadata sidecar.
    """
    backend = resolve_compiler_backend(getattr(args, "prompt_compiler", None))
    fallback = resolve_compiler_fallback(getattr(args, "prompt_compiler_fallback", None))
    if backend == "off" and not (args.prompt or "").strip():
        plan = compile_avatar_turn(
            "",
            mode=_infer_mode_for_compiler(args.mode),  # type: ignore[arg-type]
            backend="off",
            negative_prompt=args.negative_prompt or "",
        )
        args.prompt = plan.positive_prompt
        args.negative_prompt = plan.negative_prompt
        return {
            "compiler_backend": plan.compiler_backend,
            "compiler_latency_ms": plan.compiler_latency_ms,
            "prompt_chars_before": 0,
            "prompt_chars_after": len(plan.positive_prompt),
        }

    audio_dur: Optional[float] = None
    if wav_path:
        try:
            audio_dur = audio_duration_sec(wav_path)
        except Exception:
            audio_dur = None

    emotion_id = getattr(args, "emotion_id", None)
    if isinstance(emotion_id, str) and emotion_id.strip().lstrip("-").isdigit():
        emotion_id = int(emotion_id.strip())

    plan = compile_avatar_turn(
        args.prompt or "",
        mode=_infer_mode_for_compiler(args.mode),  # type: ignore[arg-type]
        image_path=getattr(args, "image", None),
        audio_duration_sec=audio_dur,
        backend=backend,
        fallback=fallback,
        negative_prompt=args.negative_prompt or "",
        emotion_id=emotion_id if isinstance(emotion_id, int) else None,
    )
    chars_before = len((args.prompt or "").strip())
    args.prompt = plan.positive_prompt
    args.negative_prompt = plan.negative_prompt
    return {
        "compiler_backend": plan.compiler_backend,
        "compiler_latency_ms": plan.compiler_latency_ms,
        "prompt_chars_before": chars_before,
        "prompt_chars_after": len(plan.positive_prompt),
        "source_user_text": plan.source_user_text,
    }


def compile_prompt_for_job(
    prompt: str,
    *,
    negative_prompt: str = "",
    mode: str = "streaming_ai2v",
    image_path: Optional[str] = None,
    audio_duration_sec: Optional[float] = None,
    compiler: Optional[str] = None,
) -> tuple[str, str]:
    """Non-argparse helper for avatar_serving jobs."""
    backend = resolve_compiler_backend(compiler)
    plan = compile_avatar_turn(
        prompt,
        mode=mode,  # type: ignore[arg-type]
        image_path=image_path,
        audio_duration_sec=audio_duration_sec,
        backend=backend,
        fallback=resolve_compiler_fallback(None),
        negative_prompt=negative_prompt,
    )
    return plan.positive_prompt, plan.negative_prompt
