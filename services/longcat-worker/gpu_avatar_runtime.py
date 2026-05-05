"""
In-process ARACHNE-X avatar pipeline (single GPU lock, multi-session queue).
Requires repo root on PYTHONPATH and NULLXES_CHECKPOINT_DIR or ARACHNE_CHECKPOINT_DIR.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipe = None
_pipe_key: Optional[str] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_syspath() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _checkpoint_dir() -> str:
    for key in ("NULLXES_CHECKPOINT_DIR", "ARACHNE_CHECKPOINT_DIR"):
        v = os.environ.get(key, "").strip()
        if v and os.path.isdir(v):
            return v
    raise RuntimeError(
        "Set NULLXES_CHECKPOINT_DIR or ARACHNE_CHECKPOINT_DIR to avatar weights (tokenizer, vae, dit, avatar_single, audio)."
    )


def get_avatar_pipeline():
    global _pipe, _pipe_key
    ckpt = _checkpoint_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("Avatar inference requires CUDA; no GPU visible to PyTorch.")
    key = f"{ckpt}|{device}"
    with _lock:
        if _pipe is not None and _pipe_key == key:
            return _pipe
        _ensure_syspath()
        from arachne_x.loader import load_avatar_pipeline

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _pipe = load_avatar_pipeline(ckpt, variant="single", device=device, torch_dtype=dtype)
        _pipe_key = key
        logger.info("Avatar pipeline loaded from %s", ckpt)
        return _pipe


def _audio_chunks_from_f32(audio_f32: np.ndarray, chunk_samples: int = 3200) -> Generator[np.ndarray, None, None]:
    x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    for i in range(0, x.size, chunk_samples):
        yield x[i : i + chunk_samples]


def _generate_frames_numpy(
    *,
    image_bytes: bytes,
    prompt: str,
    audio_f32: np.ndarray,
    num_inference_steps: int = 8,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    resolution: str = "480p",
    num_frames: int = 93,
) -> List[np.ndarray]:
    from PIL import Image

    if audio_f32.size == 0:
        raise ValueError("audio_f32 is empty")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipe = get_avatar_pipeline()

    def audio_stream():
        for c in _audio_chunks_from_f32(audio_f32):
            if c.size > 0:
                yield c

    out: List[np.ndarray] = []
    with _lock:
        for frame in pipe.generate_streaming_ai2v(
            image=img,
            prompt=prompt or "A person speaking clearly to camera.",
            audio_stream=audio_stream(),
            resolution=resolution if resolution in ("480p", "720p") else "480p",
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            text_guidance_scale=float(text_guidance_scale),
            audio_guidance_scale=float(audio_guidance_scale),
        ):
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            out.append(arr)
    return out


def stream_avatar_frames_raw_sync(
    *,
    image_bytes: bytes,
    prompt: str,
    audio_f32: np.ndarray,
    negative_prompt: str = "",
    num_inference_steps: int = 8,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    resolution: str = "480p",
    num_frames: int = 93,
) -> Iterator[tuple[int, bytes, int, int]]:
    from PIL import Image

    del negative_prompt
    if audio_f32.size == 0:
        raise ValueError("audio_f32 is empty")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipe = get_avatar_pipeline()

    def audio_stream():
        for c in _audio_chunks_from_f32(audio_f32):
            if c.size > 0:
                yield c

    seq = 0
    with _lock:
        for frame in pipe.generate_streaming_ai2v(
            image=img,
            prompt=prompt or "A person speaking clearly to camera.",
            audio_stream=audio_stream(),
            resolution=resolution if resolution in ("480p", "720p") else "480p",
            num_frames=int(num_frames),
            num_inference_steps=int(num_inference_steps),
            text_guidance_scale=float(text_guidance_scale),
            audio_guidance_scale=float(audio_guidance_scale),
        ):
            seq += 1
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            # Expect RGB uint8 HWC
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            h, w, _c = arr.shape
            raw = np.ascontiguousarray(arr).tobytes()
            yield seq, raw, int(w), int(h)


def generate_mp4_bytes_from_job(job: dict[str, Any]) -> bytes:
    """
    Synchronous full clip for legacy job API: audio-image-to-video → MP4 bytes (temp encode file only).
    """
    import tempfile

    import torch as T
    from torchvision.io import write_video

    task = str(job.get("task") or "")
    if task not in ("audio-image-to-video", "audio-text-to-video"):
        raise RuntimeError(f"in-process GPU path does not support task={task!r}")
    prompt = str(job.get("prompt") or "")
    image_path = job.get("image_path")
    audio_path = job.get("audio_path")
    if not image_path or not os.path.isfile(str(image_path)):
        raise ValueError("job missing image_path")
    if not audio_path or not os.path.isfile(str(audio_path)):
        raise ValueError("job missing audio_path")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    import librosa

    speech, _sr = librosa.load(str(audio_path), sr=16000, mono=True)
    audio_f32 = np.asarray(speech, dtype=np.float32)
    frames = _generate_frames_numpy(
        image_bytes=image_bytes,
        prompt=prompt,
        audio_f32=audio_f32,
    )
    if not frames:
        raise RuntimeError("avatar produced zero frames")
    vid = np.stack(frames, axis=0)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="nx_avatar_")
    os.close(fd)
    try:
        write_video(
            tmp_path,
            T.from_numpy(vid),
            fps=30,
            video_codec="libx264",
            options={"crf": "23"},
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
