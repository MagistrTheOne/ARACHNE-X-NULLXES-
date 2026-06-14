"""
Avatar GPU serving — shared by Inference Worker and callable from CLI/runtime.

Single implementation for ``load_avatar_pipeline`` caching, streaming frames, and MP4 jobs.
Worker package re-exports from here to keep one schema truth (NIGHT FURY V2).
"""

from __future__ import annotations

import argparse
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


def _default_identity_bank_path() -> Optional[str]:
    for env_key in ("NULLXES_IDENTITY_BANK_PATH", "ARACHNE_IDENTITY_BANK_PATH"):
        p = (os.environ.get(env_key) or "").strip()
        if p and os.path.isfile(p):
            return p
    return None


def ensure_identity_bank_loaded(pipe: Any, bank_path: Optional[str] = None) -> None:
    path = (bank_path or "").strip() or _default_identity_bank_path()
    if path and os.path.isfile(path):
        pipe.load_identity_bank(path, strict=False)
        logger.info("Identity bank loaded from %s", path)


def load_mouth_mask_from_base64(mask_b64: Optional[str]) -> Optional[torch.Tensor]:
    if not mask_b64 or not str(mask_b64).strip():
        return None
    import base64
    from PIL import Image

    raw = base64.b64decode(str(mask_b64).strip())
    img = Image.open(io.BytesIO(raw)).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def configure_realtime_pipe(pipe: Any, mouth_mask_tensor: Optional[torch.Tensor] = None) -> None:
    if mouth_mask_tensor is not None:
        pipe.hybrid_renderer_enabled = True


def get_avatar_pipeline():
    """Singleton avatar pipeline for the current checkpoint dir (CUDA required)."""
    global _pipe, _pipe_key
    ckpt = _checkpoint_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("Avatar inference requires CUDA; no GPU visible to PyTorch.")
    if torch.cuda.is_available():
        vram_gb = float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
        if vram_gb <= 45.0:
            raise RuntimeError(
                f"Full avatar inference requires >45GB VRAM (detected {vram_gb:.1f} GiB). "
                "A100-40 / similar tiers are train/LoRA/latent-export only — "
                "use scripts/train_lora_avatar.py on this pod."
            )
    key = f"{ckpt}|{device}"
    with _lock:
        if _pipe is not None and _pipe_key == key:
            return _pipe
        _ensure_syspath()
        from arachne_x.loader import load_avatar_pipeline

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _pipe = load_avatar_pipeline(ckpt, variant="single", device=device, torch_dtype=dtype)
        ensure_identity_bank_loaded(_pipe)
        _pipe_key = key
        logger.info("Avatar pipeline loaded from %s", ckpt)
        return _pipe


def _job_get(job: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in job and job[k] is not None:
            return job[k]
    return default


def _streaming_sampling_args(
    *,
    runtime_profile: Optional[str] = None,
    num_inference_steps: int = 8,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    resolution: str = "480p",
    num_frames: int = 93,
    chunk_frames: Optional[int] = None,
    first_chunk_frames: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_chunked_denoise: Optional[bool] = None,
    use_distill: Optional[bool] = None,
) -> argparse.Namespace:
    """Build args namespace and apply operational/cinematic profile (explicit job fields win)."""
    from arachne_x.runtime.sampling_profiles import (
        apply_sampling_profile,
        cuda_vram_gb,
        resolve_operational_resolution,
        resolve_use_distill,
    )

    profile = (
        runtime_profile
        or os.environ.get("ARACHNE_RUNTIME_PROFILE")
        or os.environ.get("NULLXES_RUNTIME_PROFILE")
        or "operational"
    )
    ns = argparse.Namespace(
        runtime_profile=str(profile).strip().lower(),
        num_inference_steps=int(num_inference_steps),
        text_guidance_scale=float(text_guidance_scale),
        audio_guidance_scale=float(audio_guidance_scale),
        resolution=resolution if resolution in ("480p", "720p") else "480p",
        num_frames=int(num_frames),
        chunk_frames=chunk_frames if chunk_frames is not None else 33,
        first_chunk_frames=first_chunk_frames,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else 8,
        use_chunked_denoise=use_chunked_denoise,
        use_distill=use_distill,
        num_frames_mode="sync",
    )
    apply_sampling_profile(ns, argv=[])
    profile_name = str(getattr(ns, "_sampling_profile_name", profile) or profile).lower()
    ns.resolution = resolve_operational_resolution(cuda_vram_gb(), profile_name, str(ns.resolution))
    if use_chunked_denoise is not None:
        ns.use_chunked_denoise = bool(use_chunked_denoise)
    if use_distill is not None:
        ns.use_distill = bool(use_distill)
    if chunk_frames is not None:
        ns.chunk_frames = int(chunk_frames)
    if first_chunk_frames is not None:
        ns.first_chunk_frames = int(first_chunk_frames)
    elif str(getattr(ns, "_sampling_profile_name", "") or "").lower() == "operational":
        env_first = (os.environ.get("ARACHNE_FIRST_CHUNK_FRAMES") or "").strip()
        ns.first_chunk_frames = int(env_first) if env_first else 9
    else:
        ns.first_chunk_frames = None
    if chunk_overlap is not None:
        ns.chunk_overlap = int(chunk_overlap)
    if ns.use_distill is None:
        ns.use_distill = resolve_use_distill(ns)
    return ns


def _attach_pipe_sampling_metrics(pipe, profile_name: Optional[str]) -> None:
    from arachne_x.runtime.sampling_metrics import RuntimeSamplingMetrics

    rsm = RuntimeSamplingMetrics(runtime_profile=profile_name)
    rsm.mark_start()
    pipe.runtime_sampling_metrics = rsm


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
    negative_prompt: str = "",
    prompt_compiler: Optional[str] = None,
    num_inference_steps: int = 8,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    resolution: str = "480p",
    num_frames: int = 93,
    runtime_profile: Optional[str] = None,
    chunk_frames: Optional[int] = None,
    first_chunk_frames: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_chunked_denoise: Optional[bool] = None,
    use_distill: Optional[bool] = None,
    identity_id: Optional[int] = None,
    identity_bank_path: Optional[str] = None,
    mouth_mask_base64: Optional[str] = None,
) -> List[np.ndarray]:
    from PIL import Image

    if audio_f32.size == 0:
        raise ValueError("audio_f32 is empty")
    from arachne_x.runtime.prompt_compiler_runtime import compile_prompt_for_job

    samp = _streaming_sampling_args(
        runtime_profile=runtime_profile,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        resolution=resolution,
        num_frames=num_frames,
        chunk_frames=chunk_frames,
        first_chunk_frames=first_chunk_frames,
        chunk_overlap=chunk_overlap,
        use_chunked_denoise=use_chunked_denoise,
        use_distill=use_distill,
    )

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipe = get_avatar_pipeline()
    ensure_identity_bank_loaded(pipe, identity_bank_path)
    mouth_mask = load_mouth_mask_from_base64(mouth_mask_base64)
    configure_realtime_pipe(pipe, mouth_mask)
    _attach_pipe_sampling_metrics(pipe, getattr(samp, "_sampling_profile_name", None))
    audio_dur = float(audio_f32.size) / 16000.0 if audio_f32.size > 0 else None
    compiled_pos, compiled_neg = compile_prompt_for_job(
        prompt or "A person speaking clearly to camera.",
        negative_prompt=negative_prompt,
        mode="streaming_ai2v",
        audio_duration_sec=audio_dur,
        compiler=prompt_compiler,
    )

    def audio_stream():
        for c in audio_chunks_from_f32(audio_f32):
            if c.size > 0:
                yield c

    out: List[np.ndarray] = []
    for frame in pipe.generate_streaming_ai2v(
        image=img,
        prompt=compiled_pos,
        negative_prompt=compiled_neg,
        audio_stream=audio_stream(),
        resolution=samp.resolution,
        num_frames=samp.num_frames,
        num_inference_steps=samp.num_inference_steps,
        use_distill=bool(samp.use_distill),
        text_guidance_scale=float(samp.text_guidance_scale),
        audio_guidance_scale=float(samp.audio_guidance_scale),
        chunk_frames=int(samp.chunk_frames),
        first_chunk_frames=(
            int(samp.first_chunk_frames) if getattr(samp, "first_chunk_frames", None) is not None else None
        ),
        chunk_overlap=int(samp.chunk_overlap),
        use_chunked_denoise=bool(samp.use_chunked_denoise),
        identity_id=identity_id,
        mouth_zone_masks=mouth_mask,
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
    prompt_compiler: Optional[str] = None,
    num_inference_steps: int = 8,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    resolution: str = "480p",
    num_frames: int = 93,
    runtime_profile: Optional[str] = None,
    chunk_frames: Optional[int] = None,
    first_chunk_frames: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    use_chunked_denoise: Optional[bool] = None,
    use_distill: Optional[bool] = None,
    identity_id: Optional[int] = None,
    identity_bank_path: Optional[str] = None,
    mouth_mask_base64: Optional[str] = None,
) -> Iterator[tuple[int, bytes, int, int]]:
    from PIL import Image
    import time

    from arachne_x.runtime.prompt_compiler_runtime import compile_prompt_for_job

    if audio_f32.size == 0:
        raise ValueError("audio_f32 is empty")

    samp = _streaming_sampling_args(
        runtime_profile=runtime_profile,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        resolution=resolution,
        num_frames=num_frames,
        chunk_frames=chunk_frames,
        first_chunk_frames=first_chunk_frames,
        chunk_overlap=chunk_overlap,
        use_chunked_denoise=use_chunked_denoise,
        use_distill=use_distill,
    )

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipe = get_avatar_pipeline()
    ensure_identity_bank_loaded(pipe, identity_bank_path)
    mouth_mask = load_mouth_mask_from_base64(mouth_mask_base64)
    configure_realtime_pipe(pipe, mouth_mask)
    _attach_pipe_sampling_metrics(pipe, getattr(samp, "_sampling_profile_name", None))
    audio_dur = float(audio_f32.size) / 16000.0
    compiled_pos, compiled_neg = compile_prompt_for_job(
        prompt or "A person speaking clearly to camera.",
        negative_prompt=negative_prompt,
        mode="streaming_ai2v",
        audio_duration_sec=audio_dur,
        compiler=prompt_compiler,
    )

    def audio_stream():
        for c in audio_chunks_from_f32(audio_f32):
            if c.size > 0:
                yield c

    seq = 0
    stream_start = time.perf_counter()
    first_frame_logged = False
    logger.info(
        "avatar_frames stream start profile=%s mode=streaming_ai2v resolution=%s frames=%s steps=%s "
        "chunked=%s chunk_frames=%s first_chunk_frames=%s chunk_overlap=%s distill=%s audio_samples=%s identity_id=%s",
        getattr(samp, "_sampling_profile_name", None),
        samp.resolution,
        int(samp.num_frames),
        int(samp.num_inference_steps),
        bool(samp.use_chunked_denoise),
        int(samp.chunk_frames),
        getattr(samp, "first_chunk_frames", None),
        int(samp.chunk_overlap),
        bool(samp.use_distill),
        int(audio_f32.size),
        identity_id,
    )
    for frame in pipe.generate_streaming_ai2v(
        image=img,
        prompt=compiled_pos,
        negative_prompt=compiled_neg,
        audio_stream=audio_stream(),
        resolution=samp.resolution,
        num_frames=samp.num_frames,
        num_inference_steps=samp.num_inference_steps,
        use_distill=bool(samp.use_distill),
        text_guidance_scale=float(samp.text_guidance_scale),
        audio_guidance_scale=float(samp.audio_guidance_scale),
        chunk_frames=int(samp.chunk_frames),
        first_chunk_frames=(
            int(samp.first_chunk_frames) if getattr(samp, "first_chunk_frames", None) is not None else None
        ),
        chunk_overlap=int(samp.chunk_overlap),
        use_chunked_denoise=bool(samp.use_chunked_denoise),
        identity_id=identity_id,
        mouth_zone_masks=mouth_mask,
    ):
        seq += 1
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            continue
        h, w, _c = arr.shape
        raw = np.ascontiguousarray(arr).tobytes()
        if not first_frame_logged:
            first_frame_logged = True
            elapsed = time.perf_counter() - stream_start
            sampling = getattr(pipe, "runtime_sampling_metrics", None)
            logger.info(
                "avatar_frames first_frame seq=%s ttff_sec=%.4f sampling=%s",
                seq,
                elapsed,
                sampling.to_dict() if sampling is not None else None,
            )
        yield seq, raw, int(w), int(h)


def generate_mp4_bytes_from_job(job: dict[str, Any]) -> bytes:
    task = str(job.get("task") or "")
    if task not in ("audio-image-to-video", "audio-text-to-video"):
        raise RuntimeError(f"in-process GPU path does not support task={task!r}")
    prompt = str(job.get("prompt") or "")
    prompt_compiler = _job_get(job, "promptCompiler", "prompt_compiler", default=None)
    if prompt_compiler is not None:
        prompt_compiler = str(prompt_compiler).strip().lower() or None
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
    runtime_profile = _job_get(job, "runtimeProfile", "runtime_profile", default=None)
    num_inference_steps = int(_job_get(job, "num_inference_steps", "numInferenceSteps", default=8))
    text_guidance_scale = float(_job_get(job, "text_guidance_scale", "textGuidanceScale", default=4.0))
    audio_guidance_scale = float(_job_get(job, "audio_guidance_scale", "audioGuidanceScale", default=4.0))
    resolution = str(_job_get(job, "resolution", default="480p"))
    num_frames = int(_job_get(job, "num_frames", "numFrames", default=93))
    chunk_frames = _job_get(job, "chunkFrames", "chunk_frames", default=None)
    chunk_overlap = _job_get(job, "chunkOverlap", "chunk_overlap", default=None)
    use_chunked = _job_get(job, "useChunkedDenoise", "use_chunked_denoise", default=None)
    use_distill = _job_get(job, "useDistill", "use_distill", default=None)
    identity_id = _job_get(job, "identityId", "identity_id", default=None)
    identity_bank_path = _job_get(job, "identityBankPath", "identity_bank_path", default=None)
    mouth_mask_b64 = _job_get(job, "mouthMaskBase64", "mouth_mask_base64", default=None)
    negative_prompt = str(_job_get(job, "negative_prompt", "negativePrompt", default="") or "")
    frames = generate_frames_numpy(
        image_bytes=image_bytes,
        prompt=prompt,
        audio_f32=audio_f32,
        negative_prompt=negative_prompt,
        prompt_compiler=prompt_compiler,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        resolution=resolution,
        num_frames=num_frames,
        runtime_profile=str(runtime_profile) if runtime_profile else None,
        chunk_frames=int(chunk_frames) if chunk_frames is not None else None,
        chunk_overlap=int(chunk_overlap) if chunk_overlap is not None else None,
        use_chunked_denoise=bool(use_chunked) if use_chunked is not None else None,
        use_distill=bool(use_distill) if use_distill is not None else None,
        identity_id=int(identity_id) if identity_id is not None else None,
        identity_bank_path=str(identity_bank_path) if identity_bank_path else None,
        mouth_mask_base64=str(mouth_mask_b64) if mouth_mask_b64 else None,
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
