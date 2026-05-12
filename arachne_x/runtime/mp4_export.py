"""
Production MP4 export: mux generated RGB frames with an audio track via ffmpeg.

Used by the Inference Worker and any path that must not ship silent MP4 when audio exists.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np


def export_avatar_mp4_bytes(
    frames_uint8: np.ndarray,
    audio_path: Optional[str],
    *,
    fps: int = 30,
    embed_audio: bool = True,
    quiet: bool = True,
) -> bytes:
    """
    ``frames_uint8``: ``uint8`` array ``[T, H, W, 3]``.
    If ``embed_audio`` and ``audio_path`` is a readable file, final MP4 contains AAC muxed from WAV.
    Otherwise returns silent MP4 (same as imageio-only path, but via ``save_video_ffmpeg``).
    """
    vid = np.asarray(frames_uint8)
    if vid.ndim != 4 or vid.shape[-1] != 3:
        raise ValueError(f"frames must be [T,H,W,3], got shape={vid.shape!r}")
    if vid.dtype != np.uint8:
        vid = np.clip(vid, 0.0, 255.0).astype(np.uint8)

    mux_audio: Optional[str] = None
    if embed_audio:
        if not audio_path or not os.path.isfile(str(audio_path)):
            raise FileNotFoundError(f"embed_audio=True requires existing audio_path, got {audio_path!r}")
        mux_audio = str(audio_path)

    fd, tmp_target = tempfile.mkstemp(suffix=".mp4", prefix="nx_mux_")
    os.close(fd)
    try:
        os.unlink(tmp_target)
    except OSError:
        pass

    from arachne_x.audio_process.torch_utils import save_video_ffmpeg

    save_video_ffmpeg(vid, tmp_target, audio_path=mux_audio, fps=int(fps), quality=5, high_quality_save=False, quiet=quiet)

    final_path = tmp_target if tmp_target.lower().endswith(".mp4") else tmp_target + ".mp4"
    if not os.path.isfile(final_path):
        raise RuntimeError(f"ffmpeg export did not produce {final_path!r}")
    try:
        with open(final_path, "rb") as f:
            return f.read()
    finally:
        try:
            if os.path.isfile(final_path):
                os.unlink(final_path)
        except OSError:
            pass
