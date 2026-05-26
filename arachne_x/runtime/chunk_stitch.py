"""
Chunk frame scheduling and pixel-space overlap blending for chunked avatar inference.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np

from arachne_x.inference_frames import round_to_4n_plus_1


def normalize_chunk_frames(chunk_frames: int) -> int:
    return round_to_4n_plus_1(max(1, int(chunk_frames)))


def iter_chunk_frame_ranges(
    total_frames: int,
    chunk_frames: int,
    overlap: int,
) -> Iterator[Tuple[int, int, int]]:
    """
  Yield ``(start, end, num_frames)`` in pixel/video frame indices.

  ``end`` is exclusive. ``num_frames`` is ``end - start`` (may be < chunk_frames on tail).
    """
    total = max(1, int(total_frames))
    chunk = normalize_chunk_frames(chunk_frames)
    ov = max(0, min(int(overlap), chunk - 1))
    step = max(1, chunk - ov)
    start = 0
    while start < total:
        end = min(start + chunk, total)
        n = end - start
        if n <= 0:
            break
        yield start, end, n
        if end >= total:
            break
        start += step


def cosine_blend_weights(overlap: int) -> np.ndarray:
    """Weights in [0,1] for blending chunk B over accumulated tail (length ``overlap``)."""
    ov = max(1, int(overlap))
    t = np.linspace(0.0, 1.0, ov, dtype=np.float32)
    # smoothstep-like cosine ramp
    return 0.5 - 0.5 * np.cos(np.pi * t)


def stitch_chunk_videos(
    chunks: List[np.ndarray],
    overlap: int,
) -> np.ndarray:
    """
    Stitch list of chunk videos ``[T,H,W,C]`` uint8 or float with cosine overlap on boundaries.
    """
    if not chunks:
        raise ValueError("stitch_chunk_videos: empty chunks")
    if len(chunks) == 1:
        out = np.asarray(chunks[0])
        if out.dtype != np.uint8:
            out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        return out

    ov = max(0, int(overlap))
    acc = np.asarray(chunks[0], dtype=np.float32)
    if acc.max() > 1.5:
        acc = acc / 255.0

    for nxt in chunks[1:]:
        cur = np.asarray(nxt, dtype=np.float32)
        if cur.max() > 1.5:
            cur = cur / 255.0
        if ov <= 0 or acc.shape[0] < ov or cur.shape[0] < ov:
            acc = np.concatenate([acc, cur], axis=0)
            continue
        w = cosine_blend_weights(ov).reshape(ov, 1, 1, 1)
        blended = acc[-ov:] * (1.0 - w) + cur[:ov] * w
        acc = np.concatenate([acc[:-ov], blended, cur[ov:]], axis=0)

    return (np.clip(acc, 0.0, 1.0) * 255.0).astype(np.uint8)


def slice_audio_emb_temporal(audio_emb, start: int, end: int):
    """Slice DiT-ready audio_emb ``[B,T,...]`` along temporal dim."""
    import torch

    if not torch.is_tensor(audio_emb):
        audio_emb = torch.as_tensor(audio_emb)
    if audio_emb.dim() < 2:
        raise ValueError(f"audio_emb rank too low for chunk slice: {tuple(audio_emb.shape)}")
    return audio_emb[:, int(start) : int(end)].contiguous()
