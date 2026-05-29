"""
Cross-chunk KV cache helpers (Stability OS).

Seeds ``pipe.kv_cache_dict`` from the tail of a decoded chunk for the next
``generate_ai2v(use_kv_cache=True, reuse_kv_cache=True)`` pass.
"""

from __future__ import annotations

from typing import Any, Optional

import loguru
import numpy as np
import torch


def chunk_kv_enabled() -> bool:
    import os

    v = os.environ.get("ARACHNE_CHUNK_KV", "1").strip().lower()
    if v in ("0", "false", "no"):
        return False
    return v in ("1", "true", "yes", "")


def seed_kv_from_chunk_tail(
    pipe: Any,
    chunk_video: np.ndarray,
    *,
    audio_emb_slice: torch.Tensor,
    kv_keep_last: int = 24,
    max_sequence_length: int = 512,
    chunk_idx: Optional[int] = None,
) -> bool:
    """
    Encode the last pixel frame of ``chunk_video`` and run a conditioning forward
    to populate ``pipe.kv_cache_dict`` for the next chunked pass.

    Cross-chunk ai2v uses the image-to-video path (``num_cond_latents=1``).
    Multi-frame seed (``kv_keep_last`` > 1) requires video-continuation ref
    semantics and is not used here.
    """
    del kv_keep_last  # API retained; seed always uses a single cond latent.
    if chunk_video is None or int(chunk_video.shape[0]) < 1:
        return False

    tail = np.asarray(chunk_video[-1:], dtype=np.float32)
    if tail.max() > 1.5:
        tail = tail / 255.0
    # [1, C, T, H, W]
    vid = torch.from_numpy(tail).permute(3, 0, 1, 2).unsqueeze(0)
    vid = vid.to(device=pipe.device, dtype=pipe.vae.dtype)
    from arachne_x.pipeline_arachne_x_video_avatar import retrieve_latents

    latents = pipe.normalize_latents(
        retrieve_latents(pipe.vae.encode(vid), generator=None, sample_mode="argmax")
    )
    if int(latents.shape[2]) > 1:
        latents = latents[:, :, -1:].contiguous()
    n_cond = 1
    audio_frames = 1
    audio_cache = (
        audio_emb_slice[:, :audio_frames]
        if audio_emb_slice.shape[1] >= audio_frames
        else audio_emb_slice
    )
    effective_cond = pipe._cache_clean_latents(
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
    pipe._cross_chunk_kv_active_cond = int(effective_cond)
    kv = pipe._get_kv_cache_dict()
    if kv:
        trimmed, _ = pipe._compress_kv_cache_dict_temporal(kv, n_cond, 0)
        pipe._update_kv_cache_dict(trimmed)
        rsm = getattr(pipe, "runtime_sampling_metrics", None)
        if rsm is not None:
            rsm.kv_cache_hits += 1
            rsm.cross_chunk_kv_frames += int(n_cond)
        loguru.logger.info(
            "chunk_kv_seed_ok chunk_idx={} n_cond={} effective_cond={} kv_layers={}",
            chunk_idx,
            n_cond,
            int(effective_cond),
            len(trimmed),
        )
        return True
    return False
