"""
Frame budget helpers for avatar inference (4n+1 rule, audio-embedding sync cap).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple

import librosa

DEFAULT_EMBEDDING_FPS = 16 * 4  # matches inference_audio: 16 * vae_scale_factor_temporal
DEFAULT_MUX_FPS = 30
VAE_TEMPORAL_STRIDE = 4
MAX_EMBEDDING_FPS = 128

NumFramesMode = Literal["explicit", "sync", "duration", "min"]


def round_to_4n_plus_1(n: int) -> int:
    n = max(1, int(n))
    return ((n - 1) // 4) * 4 + 1


def audio_duration_sec(path: str, sample_rate: int = 16000) -> float:
    try:
        from arachne_x.audio_process.torch_utils import get_audio_duration

        return float(get_audio_duration(path))
    except Exception:
        speech, _ = librosa.load(path, sr=sample_rate)
        return len(speech) / float(sample_rate)


def embedding_timesteps(duration_sec: float, embedding_fps: float) -> int:
    return max(1, int(duration_sec * embedding_fps))


def max_sync_frames(
    duration_sec: float,
    embedding_fps: float,
    vae_stride: int = VAE_TEMPORAL_STRIDE,
) -> int:
    """Max num_frames before audio window indices clamp to the last embedding step."""
    t_emb = embedding_timesteps(duration_sec, embedding_fps)
    raw = ((t_emb - 1) // vae_stride) // vae_stride * vae_stride + 1
    return max(1, raw)


def duration_frames(duration_sec: float, mux_fps: float = DEFAULT_MUX_FPS) -> int:
    raw = max(1, int(round(duration_sec * mux_fps)))
    return round_to_4n_plus_1(raw)


def suggest_embedding_fps(
    duration_sec: float,
    num_frames: int,
    base_fps: float = DEFAULT_EMBEDDING_FPS,
    vae_stride: int = VAE_TEMPORAL_STRIDE,
    max_fps: float = MAX_EMBEDDING_FPS,
) -> float:
    """Raise embedding fps so ``num_frames`` does not rely on heavy index clamp."""
    sync_at_base = max_sync_frames(duration_sec, base_fps, vae_stride)
    if num_frames <= sync_at_base:
        return base_fps
    needed = math.ceil(4.0 * (num_frames - 1) / max(duration_sec, 1e-6))
    return min(max_fps, max(base_fps, float(needed)))


def resolve_num_frames(
    mode: str,
    duration_sec: float,
    embedding_fps: float,
    explicit: Optional[int] = None,
    mux_fps: float = DEFAULT_MUX_FPS,
    vae_stride: int = VAE_TEMPORAL_STRIDE,
) -> Tuple[int, Dict[str, Any]]:
    sync_max = max_sync_frames(duration_sec, embedding_fps, vae_stride)
    dur_frames = duration_frames(duration_sec, mux_fps)
    info: Dict[str, Any] = {
        "duration_sec": round(duration_sec, 4),
        "embedding_fps": embedding_fps,
        "sync_max_frames": sync_max,
        "duration_frames": dur_frames,
        "mode": mode,
    }
    if mode in (None, "explicit"):
        chosen = explicit if explicit is not None else dur_frames
        chosen = round_to_4n_plus_1(chosen)
        info["chosen"] = chosen
        return chosen, info
    if mode == "sync":
        info["chosen"] = sync_max
        return sync_max, info
    if mode == "duration":
        info["chosen"] = dur_frames
        return dur_frames, info
    if mode == "min":
        chosen = min(sync_max, dur_frames)
        info["chosen"] = chosen
        return chosen, info
    raise ValueError(f"Unknown num_frames_mode: {mode!r} (use explicit|sync|duration|min)")
