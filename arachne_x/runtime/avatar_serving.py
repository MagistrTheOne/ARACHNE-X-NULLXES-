"""
Avatar GPU serving — shared by Inference Worker and callable from CLI/runtime.

Single implementation for ``load_avatar_pipeline`` caching, streaming frames, and MP4 jobs.
Worker package re-exports from here to keep one schema truth (NIGHT FURY V2).
"""

from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Generator, Iterator, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipe: Any = None
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
    """Singleton avatar pipeline for the current checkpoint dir (CUDA required)."""
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


def _job_get(job: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in job and job[k] is not None:
            return job[k]
    return default


def synthesize_speak_text_to_wav(
    text: str,
    *,
    work_dir: str,
    tts_provider: str,
    input_json: Optional[dict[str, Any]] = None,
) -> str:
    """MODE B: TTS -> WAV on disk (16 kHz conditioning via librosa downstream)."""
    import torch

    from arachne_x.tts import create_speech_synthesizer

    ij = dict(input_json or {})

    def pick(*ks: str) -> Any:
        for k in ks:
            if k in ij and ij[k] not in (None, ""):
                return ij[k]
        return None

    provider = (tts_provider or "qwen").strip().lower()
    dm = pick("ttsDeviceMap", "tts_device_map") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    mid = pick("ttsModel", "tts_model")
    language = str(pick("ttsLanguage", "tts_language") or "English")
    speaker = str(pick("ttsSpeaker", "tts_speaker") or "Ryan")
    instruct = pick("ttsInstruct", "tts_instruct")
    attn = pick("ttsAttn", "tts_attn", "tts_attn_implementation")
    synth = create_speech_synthesizer(
        provider,
        model_id=mid,
        device_map=dm,
        language=language,
        speaker=speaker,
        instruct=instruct,
        attn_implementation=attn,
        audiodit_nfe=pick("audioditNfe", "audiodit_nfe"),
        audiodit_guidance_strength=pick("audioditGuidanceStrength", "audiodit_guidance_strength"),
        audiodit_guidance_method=pick("audioditGuidanceMethod", "audiodit_guidance_method"),
        audiodit_prompt_audio=pick("audioditPromptAudio", "audiodit_prompt_audio"),
        audiodit_prompt_text=pick("audioditPromptText", "audiodit_prompt_text"),
        audiodit_seed=pick("audioditSeed", "audiodit_seed"),
    )
    out = os.path.join(work_dir, "tts_generated.wav")
    synth.synthesize_to_path(text.strip(), out)
    return out


def audio_chunks_from_f32(audio_f32: np.ndarray, chunk_samples: int = 3200) -> Generator[np.ndarray, None, None]:
    x = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
    for i in range(0, x.size, chunk_samples):
        yield x[i : i + chunk_samples]


def generate_frames_numpy(
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
        for c in audio_chunks_from_f32(audio_f32):
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
        for c in audio_chunks_from_f32(audio_f32):
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
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            h, w, _c = arr.shape
            raw = np.ascontiguousarray(arr).tobytes()
            yield seq, raw, int(w), int(h)


def generate_mp4_bytes_from_job(job: dict[str, Any]) -> bytes:
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
    num_inference_steps = int(_job_get(job, "num_inference_steps", "numInferenceSteps", default=8))
    text_guidance_scale = float(_job_get(job, "text_guidance_scale", "textGuidanceScale", default=4.0))
    audio_guidance_scale = float(_job_get(job, "audio_guidance_scale", "audioGuidanceScale", default=4.0))
    resolution = str(_job_get(job, "resolution", default="480p"))
    num_frames = int(_job_get(job, "num_frames", "numFrames", default=93))
    frames = generate_frames_numpy(
        image_bytes=image_bytes,
        prompt=prompt,
        audio_f32=audio_f32,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        resolution=resolution,
        num_frames=num_frames,
    )
    if not frames:
        raise RuntimeError("avatar produced zero frames")
    vid = np.stack(frames, axis=0)
    out_mode = str(_job_get(job, "output_mode", "outputMode", default="mp4") or "mp4").lower()
    if out_mode != "mp4":
        raise ValueError(f"unsupported output_mode: {out_mode!r}")
    embed_audio = bool(_job_get(job, "embed_audio", "embedAudio", default=True))
    fps = int(_job_get(job, "fps", default=30))

    from arachne_x.runtime.mp4_export import export_avatar_mp4_bytes

    return export_avatar_mp4_bytes(
        vid,
        str(audio_path) if embed_audio else None,
        fps=fps,
        embed_audio=embed_audio,
        quiet=True,
    )
