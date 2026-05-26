"""
Avatar inference HTTP worker: in-process GPU (no torchrun subprocess).
Streaming: POST /v1/realtime/avatar_frames → application/x-ndjson
Legacy jobs: POST /v1/infer/jobs (MP4 result file, temp disk for mux only).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Iterator, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gpu_avatar_runtime import generate_mp4_bytes_from_job, stream_avatar_frames_raw_sync
from job_queue import JobStatus, job_queue
from streaming_queue import (
    StreamingQueueRejected,
    StreamingQueueTimeout,
    WorkerLifecycle,
    streaming_queue,
)

logger = logging.getLogger(__name__)

EXPECTED_KEY_HEADER = "x-nullxes-avatar-inference-key"

# Core vision engines (NULLXES DiT/VAE path).
_CORE_VISION_ENGINES = frozenset({"", "arachne", "nullxes", "core"})
# NULLXES HR AI gateway (`resolveArachnePodEngine`) sends these when VIDEO_ENGINE is ultra-branded;
# realtime NDJSON still uses the same audio-driven avatar pipeline as `arachne`.
_ULTRA_AVATAR_FRAME_ALIASES = frozenset({"arachne_ultra_avatar", "arachne_ultra_video"})


def _normalize_avatar_frames_engine(raw: str) -> str:
    e = (raw or "").strip().lower()
    if e in _ULTRA_AVATAR_FRAME_ALIASES:
        return "arachne"
    return e


def _inference_service_key_expected() -> Optional[str]:
    for env_name in (
        "NULLXES_INFERENCE_SERVICE_KEY",
        "NULLXES_AVATAR_INFERENCE_SERVICE_KEY",
    ):
        v = os.environ.get(env_name, "").strip()
        if v:
            return v
    return None


TaskLiteral = Literal[
    "text-to-video",
    "image-to-video",
    "video-continuation",
    "audio-text-to-video",
    "audio-image-to-video",
]


class GenerateBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: TaskLiteral = "text-to-video"
    prompt: str = Field(..., max_length=16000)
    sessionId: str = Field(..., max_length=512)
    imageBase64: Optional[str] = None
    audioBase64: Optional[str] = None
    continuationState: Optional[str] = None
    numSegments: Optional[int] = Field(default=None, ge=1, le=256)
    refImgIndex: Optional[int] = Field(default=None, ge=0)
    negative_prompt: Optional[str] = Field(default=None, max_length=8000)
    inputJson: Optional[dict[str, Any]] = None
    embedAudio: bool = Field(default=True)
    outputMode: str = Field(default="mp4", max_length=32)
    numInferenceSteps: int = Field(default=8, ge=1, le=64)
    textGuidanceScale: float = Field(default=4.0)
    audioGuidanceScale: float = Field(default=4.0)
    resolution: str = Field(default="480p", max_length=16)
    numFrames: int = Field(default=93, ge=1, le=256)


class StreamFramesBody(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    sessionId: str = Field(..., max_length=512)
    prompt: str = Field("", max_length=16000)
    imageBase64: str = Field(..., min_length=8)
    # Preferred realtime audio contract (MVP): PCM16 mono 16kHz base64 (20-40ms chunking upstream).
    audioPcm16Base64: Optional[str] = Field(default=None, min_length=4)
    # Backwards compatibility: float32 mono 16kHz base64.
    audioFloat32Base64: Optional[str] = Field(default=None, min_length=4)
    negativePrompt: Optional[str] = Field(default=None, max_length=8000)
    numInferenceSteps: int = Field(default=8, ge=1, le=64)
    textGuidanceScale: float = Field(default=4.0)
    audioGuidanceScale: float = Field(default=4.0)
    resolution: str = Field(default="480p")
    numFrames: int = Field(default=93, ge=1, le=256)
    runtimeProfile: Optional[str] = Field(
        default=None,
        max_length=32,
        description="operational (default) | cinematic. Env ARACHNE_RUNTIME_PROFILE when omitted.",
    )
    chunkFrames: Optional[int] = Field(default=None, ge=5, le=256)
    firstChunkFrames: Optional[int] = Field(default=None, ge=5, le=65)
    chunkOverlap: Optional[int] = Field(default=None, ge=0, le=64)
    useChunkedDenoise: Optional[bool] = Field(default=None)
    useDistill: Optional[bool] = Field(default=None)
    identityId: Optional[int] = Field(default=None, ge=0, le=1023)
    identityBankPath: Optional[str] = Field(default=None, max_length=4096)
    mouthMaskBase64: Optional[str] = Field(default=None)
    engine: str = Field(default="arachne", max_length=64)


def _check_key(expected: Optional[str], hdr: Optional[str]) -> None:
    if not expected:
        return
    if (hdr or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid inference key")


def _validate_body(body: GenerateBody) -> None:
    if body.task == "image-to-video" and not (body.imageBase64 or "").strip():
        raise ValueError("imageBase64 required for image-to-video")
    if body.task == "audio-text-to-video":
        if not (body.audioBase64 or "").strip():
            raise ValueError("audioBase64 required for audio-text-to-video")
    if body.task == "audio-image-to-video":
        if not (body.audioBase64 or "").strip():
            raise ValueError("audioBase64 required for audio-image-to-video")
        if not (body.imageBase64 or "").strip():
            raise ValueError("imageBase64 required for audio-image-to-video")
    if body.task == "video-continuation" and not (body.continuationState or "").strip():
        raise ValueError("continuationState (base64-encoded mp4) required for video-continuation")


def run_generate_sync(body: GenerateBody) -> bytes:
    from PIL import Image

    _validate_body(body)
    task = str(body.task)
    if task in ("audio-image-to-video", "audio-text-to-video"):
        work = tempfile.mkdtemp(prefix="nx_inf_")
        try:
            out_mp4 = os.path.join(work, "result.mp4")
            job: dict[str, Any] = {
                "task": task,
                "prompt": body.prompt,
                "output_mp4": out_mp4,
                "embed_audio": body.embedAudio,
                "output_mode": body.outputMode,
                "num_inference_steps": body.numInferenceSteps,
                "text_guidance_scale": body.textGuidanceScale,
                "audio_guidance_scale": body.audioGuidanceScale,
                "resolution": body.resolution,
                "num_frames": body.numFrames,
                "fps": 30,
            }
            if body.negative_prompt:
                job["negative_prompt"] = body.negative_prompt
            if body.task == "audio-image-to-video":
                raw = base64.b64decode(body.imageBase64 or "")
                ip = os.path.join(work, "in.png")
                with open(ip, "wb") as f:
                    f.write(raw)
                job["image_path"] = ip
            else:
                ip = os.path.join(work, "neutral.png")
                Image.new("RGB", (832, 480), (40, 40, 48)).save(ip, format="PNG")
                job["image_path"] = ip
            raw_a = base64.b64decode(body.audioBase64 or "")
            ap = os.path.join(work, "in.wav")
            with open(ap, "wb") as f:
                f.write(raw_a)
            job["audio_path"] = ap
            return generate_mp4_bytes_from_job(job)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    raise RuntimeError(
        f"Task {task!r} is not supported by the in-process worker "
        "(only audio-image-to-video and audio-text-to-video). "
        "Use a dedicated batch service for text-to-video / image-to-video / continuation."
    )


def execute_inference_from_dict(d: dict[str, Any]) -> bytes:
    body = GenerateBody.model_validate(d)
    return run_generate_sync(body)


def _ndjson_stream(body: StreamFramesBody) -> Iterator[bytes]:
    import numpy as np
    import time

    raw_engine = (body.engine or "arachne").strip().lower()
    if raw_engine == "wan_s2v":
        err = json.dumps(
            {
                "error": (
                    "wan_s2v engine is not deployed on this worker "
                    "(use engine=arachne / omit engine, or run a dedicated Wan S2V inference service)"
                )
            },
            ensure_ascii=False,
        )
        yield (err + "\n").encode("utf-8")
        return
    engine = _normalize_avatar_frames_engine(raw_engine)
    if engine not in _CORE_VISION_ENGINES:
        err = json.dumps(
            {
                "error": (
                    f"unknown engine {raw_engine!r}; supported: arachne, nullxes, "
                    "core, arachne_ultra_avatar, arachne_ultra_video (HR aliases → arachne)"
                )
            },
            ensure_ascii=False,
        )
        yield (err + "\n").encode("utf-8")
        return

    try:
        img = base64.b64decode(body.imageBase64)
        if body.audioPcm16Base64:
            raw16 = base64.b64decode(body.audioPcm16Base64)
            pcm16 = np.frombuffer(raw16, dtype=np.int16).astype(np.float32)
            audio = np.clip(pcm16 / 32768.0, -1.0, 1.0).copy()
        elif body.audioFloat32Base64:
            raw = base64.b64decode(body.audioFloat32Base64)
            audio = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError("audioPcm16Base64 or audioFloat32Base64 is required")
        t0_ns = time.monotonic_ns()
        first_line_logged = False
        for seq, frame_bytes, w, h in stream_avatar_frames_raw_sync(
            image_bytes=img,
            prompt=body.prompt,
            audio_f32=audio,
            negative_prompt=body.negativePrompt or "",
            num_inference_steps=body.numInferenceSteps,
            text_guidance_scale=body.textGuidanceScale,
            audio_guidance_scale=body.audioGuidanceScale,
            resolution=body.resolution,
            num_frames=body.numFrames,
            runtime_profile=body.runtimeProfile,
            chunk_frames=body.chunkFrames,
            first_chunk_frames=body.firstChunkFrames,
            chunk_overlap=body.chunkOverlap,
            use_chunked_denoise=body.useChunkedDenoise,
            use_distill=body.useDistill,
            identity_id=body.identityId,
            identity_bank_path=body.identityBankPath,
            mouth_mask_base64=body.mouthMaskBase64,
        ):
            # Use monotonic clock to timestamp frames for sync downstream.
            ts_ms = int((time.monotonic_ns() - t0_ns) / 1_000_000)
            if not first_line_logged:
                first_line_logged = True
                logger.info(
                    "avatar_frames ndjson_first_line session_id=%s seq=%s ttff_ms=%s width=%s height=%s",
                    body.sessionId,
                    seq,
                    ts_ms,
                    w,
                    h,
                )
            frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
            line = (
                json.dumps(
                    {
                        "seq": seq,
                        "tsMs": ts_ms,
                        "encoding": "rgb24_base64",
                        "width": w,
                        "height": h,
                        "frameBase64": frame_b64,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            yield line.encode("utf-8")
    except Exception as e:
        err = json.dumps({"error": str(e)[:4000]}, ensure_ascii=False) + "\n"
        yield err.encode("utf-8")


@asynccontextmanager
async def _lifespan(_app: object):
    job_queue.set_executor(execute_inference_from_dict)
    job_queue.start_worker()
    streaming_queue.mark_ready()
    yield
    await job_queue.stop_worker()
    streaming_queue.set_lifecycle(WorkerLifecycle.offline)


def _worker_busy_http(exc: StreamingQueueRejected) -> HTTPException:
    return HTTPException(status_code=503, detail=exc.to_detail())


def _queue_timeout_http(exc: StreamingQueueTimeout) -> HTTPException:
    return HTTPException(status_code=503, detail=exc.to_detail())


def _vram_used_mb() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def _gpu_visible() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _runtime_metrics_payload() -> dict[str, Any]:
    snap = streaming_queue.snapshot()
    snap["gpuVisible"] = _gpu_visible()
    vram = _vram_used_mb()
    if vram is not None:
        snap["vramUsedMb"] = vram
    snap["mp4JobQueueDepth"] = job_queue.queue_depth()
    return snap


app = FastAPI(title="NULLXES Avatar inference worker", version="2.0.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    snap = streaming_queue.snapshot()
    lifecycle = snap.get("lifecycle", "active")
    gpu_ok = _gpu_visible()
    status = "ok" if lifecycle == "active" and gpu_ok else "degraded"
    if lifecycle in ("offline", "draining"):
        status = lifecycle
    return {
        "status": status,
        "lifecycle": lifecycle,
        "gpuVisible": gpu_ok,
        "activeStreamingJobs": snap.get("activeStreamingJobs", 0),
        "queueDepth": snap.get("queueDepth", 0),
        "vramUsedMb": _vram_used_mb(),
        "uptimeSec": snap.get("uptimeSec", 0),
        "lastJobFinishedAt": snap.get("lastJobFinishedAt"),
    }


@app.get("/v1/runtime/metrics")
async def runtime_metrics(
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> dict[str, Any]:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    return _runtime_metrics_payload()


@app.post("/v1/admin/drain")
async def admin_drain(
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> dict[str, str]:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    streaming_queue.set_lifecycle(WorkerLifecycle.draining)
    logger.warning("worker lifecycle -> draining")
    return {"lifecycle": WorkerLifecycle.draining.value}


@app.post("/v1/admin/activate")
async def admin_activate(
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> dict[str, str]:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    streaming_queue.set_lifecycle(WorkerLifecycle.active)
    logger.info("worker lifecycle -> active")
    return {"lifecycle": WorkerLifecycle.active.value}


@app.post("/v1/realtime/avatar_frames")
async def realtime_avatar_frames(
    body: StreamFramesBody,
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> StreamingResponse:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    try:
        ticket = streaming_queue.try_admit(body.sessionId)
    except StreamingQueueRejected as exc:
        logger.warning(
            "avatar_frames reject session_id=%s error=%s queue_depth=%s",
            body.sessionId,
            exc.error,
            exc.queue_depth,
        )
        raise _worker_busy_http(exc) from exc

    try:
        await asyncio.to_thread(streaming_queue.wait_for_active, ticket.ticket_id)
    except StreamingQueueRejected as exc:
        logger.warning(
            "avatar_frames wait_reject session_id=%s ticket=%s error=%s",
            body.sessionId,
            ticket.ticket_id,
            exc.error,
        )
        raise _worker_busy_http(exc) from exc
    except StreamingQueueTimeout as exc:
        logger.warning(
            "avatar_frames queue_timeout session_id=%s ticket=%s waited_ms=%s queue_depth=%s",
            body.sessionId,
            ticket.ticket_id,
            exc.waited_ms,
            exc.queue_depth,
        )
        raise _queue_timeout_http(exc) from exc

    def _stream_with_release() -> Iterator[bytes]:
        try:
            yield from _ndjson_stream(body)
        finally:
            streaming_queue.release_active(ticket.ticket_id)

    try:
        return StreamingResponse(
            _stream_with_release(),
            media_type="application/x-ndjson",
        )
    except ValueError as e:
        streaming_queue.release_active(ticket.ticket_id)
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/v1/arachne/generate")
async def generate(
    body: GenerateBody,
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> Response:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    try:
        data = await asyncio.to_thread(run_generate_sync, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        msg = str(e)
        if "not supported" in msg or "CUDA" in msg or "checkpoint" in msg.lower():
            raise HTTPException(status_code=503, detail=msg[:8000]) from e
        raise HTTPException(status_code=502, detail=msg[:8000]) from e
    return Response(content=data, media_type="video/mp4")


@app.post("/v1/infer/jobs")
async def create_infer_job(
    body: GenerateBody,
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> dict[str, Any]:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    try:
        _validate_body(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        job_id = job_queue.enqueue(body.model_dump(mode="json", exclude_none=True))
    except RuntimeError as ex:
        if str(ex) == "queue_full":
            raise HTTPException(status_code=503, detail="inference queue full") from ex
        raise
    return {"jobId": job_id, "status": JobStatus.queued.value}


@app.get("/v1/infer/jobs/{job_id}")
async def get_infer_job(
    job_id: str,
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> dict[str, Any]:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    rec = job_queue.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    out: dict[str, Any] = {"jobId": job_id, "status": rec.status.value}
    if rec.error:
        out["error"] = rec.error
    return out


@app.get("/v1/infer/jobs/{job_id}/result")
async def get_infer_job_result(
    job_id: str,
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> Response:
    expected = _inference_service_key_expected()
    _check_key(expected, x_nullxes_avatar_inference_key)
    rec = job_queue.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    if rec.status == JobStatus.failed:
        raise HTTPException(status_code=502, detail=rec.error or "inference failed")
    if rec.status != JobStatus.done:
        raise HTTPException(status_code=409, detail=f"job not ready: {rec.status.value}")
    path = job_queue.pop_result_path(job_id)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="result missing")
    try:
        with open(path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    job_queue.delete_job(job_id)
    return Response(content=data, media_type="video/mp4")


@app.post("/v1/arachne/debug/decode-image")
async def debug_image(body: GenerateBody) -> dict[str, int]:
    if not body.imageBase64:
        raise HTTPException(400, "no imageBase64")
    raw = base64.b64decode(body.imageBase64)
    return {"image_bytes": len(raw)}
