"""
Cross-chunk KV cache helpers (Sprint 1.5 wedge).

Seeds ``pipe.kv_cache_dict`` from the tail of a decoded chunk so a future
``generate_ai2v(use_kv_cache=True)`` pass can reuse temporal context.
Chunked ai2v does not consume KV yet; enable with ``ARACHNE_CHUNK_KV=1``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


def chunk_kv_enabled() -> bool:
    import os

    return os.environ.get("ARACHNE_CHUNK_KV", "").strip().lower() in ("1", "true", "yes")


def seed_kv_from_chunk_tail(
    pipe: Any,
    chunk_video: np.ndarray,
    *,
    audio_emb_slice: torch.Tensor,
    kv_keep_last: int = 24,
    max_sequence_length: int = 512,
) -> bool:
    """
    Encode the last ``kv_keep_last`` pixel frames of ``chunk_video`` and run a
    conditioning forward to populate ``pipe.kv_cache_dict`` (trimmed).
    """
    if chunk_video is None or int(chunk_video.shape[0]) < 1:
        return False
    keep = max(1, int(kv_keep_last))
    tail = np.asarray(chunk_video[-keep:], dtype=np.float32)
    if tail.max() > 1.5:
        tail = tail / 255.0
    # [1, C, T, H, W]
    vid = torch.from_numpy(tail).permute(3, 0, 1, 2).unsqueeze(0)
    vid = vid.to(device=pipe.device, dtype=pipe.vae.dtype)
    from arachne_x.pipeline_arachne_x_video_avatar import retrieve_latents

    latents = pipe.normalize_latents(
        retrieve_latents(pipe.vae.encode(vid), generator=None, sample_mode="argmax")
    )
    n_cond = int(latents.shape[2])
    audio_cache = audio_emb_slice[:, :n_cond] if audio_emb_slice.shape[1] >= n_cond else audio_emb_slice
    pipe._cache_clean_latents(
        latents,
        max_sequence_length,
        offload_kv_cache=False,
        device=pipe.device,
        dtype=pipe.dit.dtype,
        audio_embs=audio_cache,
        num_cond_latents=n_cond,
        num_ref_latents=0,
        ref_img_index=None,
    )
    kv = pipe._get_kv_cache_dict()
    if kv:
        trimmed, _ = pipe._compress_kv_cache_dict_temporal(kv, n_cond, 0)
        pipe._update_kv_cache_dict(trimmed)
        rsm = getattr(pipe, "runtime_sampling_metrics", None)
        if rsm is not None:
            rsm.kv_cache_hits += 1
        return True
    return False
