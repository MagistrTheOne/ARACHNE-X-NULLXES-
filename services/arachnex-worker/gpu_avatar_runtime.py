"""
Inference Worker entrypoints — lazy import of ``arachne_x.runtime.avatar_serving``.

Keeps uvicorn importable without PYTHONPATH to repo root; ARACHNE-X loads on first GPU call.
"""

from __future__ import annotations

from typing import Any, Iterator


def get_avatar_pipeline():
    from arachne_x.runtime.avatar_serving import get_avatar_pipeline as _impl

    return _impl()


def stream_avatar_frames_raw_sync(
    *,
    image_bytes: bytes,
    prompt: str,
    audio_f32: Any,
    negative_prompt: str = "",
    num_inference_steps: int = 8,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    resolution: str = "480p",
    num_frames: int = 93,
    runtime_profile: str | None = None,
    chunk_frames: int | None = None,
    chunk_overlap: int | None = None,
    use_chunked_denoise: bool | None = None,
    use_distill: bool | None = None,
) -> Iterator[tuple[int, bytes, int, int]]:
    from arachne_x.runtime.avatar_serving import stream_avatar_frames_raw_sync as _impl

    yield from _impl(
        image_bytes=image_bytes,
        prompt=prompt,
        audio_f32=audio_f32,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        resolution=resolution,
        num_frames=num_frames,
        runtime_profile=runtime_profile,
        chunk_frames=chunk_frames,
        chunk_overlap=chunk_overlap,
        use_chunked_denoise=use_chunked_denoise,
        use_distill=use_distill,
    )


def generate_mp4_bytes_from_job(job: dict[str, Any]) -> bytes:
    from arachne_x.runtime.avatar_serving import generate_mp4_bytes_from_job as _impl

    return _impl(job)


def audio_chunks_from_f32(*args: Any, **kwargs: Any):
    from arachne_x.runtime.avatar_serving import audio_chunks_from_f32 as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "get_avatar_pipeline",
    "stream_avatar_frames_raw_sync",
    "generate_mp4_bytes_from_job",
    "audio_chunks_from_f32",
]
