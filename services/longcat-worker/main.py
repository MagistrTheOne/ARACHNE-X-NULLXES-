"""
Avatar / video inference HTTP worker for ARACHNE-X.

Production: ARACHNE_VIDEO_REPO + ARACHNE_CHECKPOINT_DIR (ARACHNE-X ULTRA weights),
or LONGCAT_VIDEO_REPO + LONGCAT_CHECKPOINT_DIR (legacy LongCat-Video).

Dev mocks (static mp4 / base64) only if ALLOW_INFERENCE_DEV_MOCK=1. Async jobs: POST /v1/infer/jobs.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from arachne_subprocess import run_arachne_inference
from job_queue import JobStatus, job_queue
from longcat_subprocess import run_longcat_inference

EXPECTED_KEY_HEADER = "x-nullxes-avatar-inference-key"

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


def _check_key(expected: Optional[str], hdr: Optional[str]) -> None:
    if not expected:
        return
    if (hdr or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid inference key")


def _allow_dev_mock() -> bool:
    return os.environ.get("ALLOW_INFERENCE_DEV_MOCK", "").strip().lower() in ("1", "true", "yes")


def _use_arachne() -> bool:
    r = os.environ.get("ARACHNE_VIDEO_REPO", "").strip()
    c = os.environ.get("ARACHNE_CHECKPOINT_DIR", "").strip()
    return bool(r and c)


def _use_longcat() -> bool:
    r = os.environ.get("LONGCAT_VIDEO_REPO", "").strip()
    c = os.environ.get("LONGCAT_CHECKPOINT_DIR", "").strip()
    return bool(r and c)


def _validate_body(body: GenerateBody) -> None:
    if body.task == "image-to-video" and not (body.imageBase64 or "").strip():
        raise ValueError("imageBase64 required for image-to-video")
    if body.task == "audio-text-to-video" and not (body.audioBase64 or "").strip():
        raise ValueError("audioBase64 required for audio-text-to-video")
    if body.task == "audio-image-to-video":
        if not (body.audioBase64 or "").strip():
            raise ValueError("audioBase64 required for audio-image-to-video")
        if not (body.imageBase64 or "").strip():
            raise ValueError("imageBase64 required for audio-image-to-video")
    if body.task == "video-continuation" and not (body.continuationState or "").strip():
        raise ValueError("continuationState (base64-encoded mp4) required for video-continuation")


def run_generate_sync(body: GenerateBody) -> bytes:
    _validate_body(body)
    if _allow_dev_mock():
        mock_path = os.environ.get("LONGCAT_MOCK_MP4_PATH", "").strip()
        if mock_path and os.path.isfile(mock_path):
            with open(mock_path, "rb") as f:
                return f.read()
    arachne = _use_arachne()
    legacy = _use_longcat()
    if not arachne and not legacy:
        if _allow_dev_mock():
            demo_b64 = os.environ.get("LONGCAT_MOCK_VIDEO_BASE64", "").strip()
            if demo_b64:
                return json.dumps({"videoBase64": demo_b64}).encode("utf-8")
        raise RuntimeError(
            "Configure ARACHNE_VIDEO_REPO + ARACHNE_CHECKPOINT_DIR or LONGCAT_*; "
            "or ALLOW_INFERENCE_DEV_MOCK=1 for dev mocks."
        )
    work = tempfile.mkdtemp(prefix="nx_inf_api_")
    try:
        out_mp4 = os.path.join(work, "result.mp4")
        job: dict[str, Any] = {
            "task": body.task,
            "prompt": body.prompt,
            "output_mp4": out_mp4,
        }
        if body.negative_prompt:
            job["negative_prompt"] = body.negative_prompt
        if body.inputJson:
            job["input_json_overlay"] = dict(body.inputJson)
        if body.numSegments is not None:
            job["num_segments"] = int(body.numSegments)
        if body.refImgIndex is not None:
            job["ref_img_index"] = int(body.refImgIndex)
        if body.task in ("image-to-video", "audio-image-to-video"):
            raw = base64.b64decode(body.imageBase64 or "")
            img_path = os.path.join(work, "input_image.png")
            with open(img_path, "wb") as f:
                f.write(raw)
            job["image_path"] = img_path
        if body.task in ("audio-text-to-video", "audio-image-to-video"):
            raw = base64.b64decode(body.audioBase64 or "")
            audio_path = os.path.join(work, "input_audio.bin")
            with open(audio_path, "wb") as f:
                f.write(raw)
            job["audio_path"] = audio_path
        if body.task == "video-continuation":
            raw = base64.b64decode(body.continuationState or "")
            vid_path = os.path.join(work, "condition.mp4")
            with open(vid_path, "wb") as f:
                f.write(raw)
            job["video_path"] = vid_path
        if arachne:
            return run_arachne_inference(job)
        if body.task not in ("text-to-video", "image-to-video", "video-continuation"):
            raise RuntimeError("This task requires ARACHNE_VIDEO_REPO + ARACHNE_CHECKPOINT_DIR")
        lc_job: dict[str, Any] = {
            "task": body.task,
            "prompt": body.prompt,
            "output_mp4": out_mp4,
        }
        if body.negative_prompt:
            lc_job["negative_prompt"] = body.negative_prompt
        if body.task == "image-to-video":
            lc_job["image_path"] = job["image_path"]
        if body.task == "video-continuation":
            lc_job["video_path"] = job["video_path"]
        return run_longcat_inference(lc_job)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def execute_inference_from_dict(d: dict[str, Any]) -> bytes:
    body = GenerateBody.model_validate(d)
    data = run_generate_sync(body)
    if data[:1] == b"{" and b"videoBase64" in data[:220]:
        obj = json.loads(data.decode("utf-8"))
        b64 = obj.get("videoBase64") or obj.get("video_base64")
        if isinstance(b64, str) and b64.strip():
            return base64.b64decode(b64)
    return data


@asynccontextmanager
async def _lifespan(app):
    job_queue.set_executor(execute_inference_from_dict)
    job_queue.start_worker()
    yield
    await job_queue.stop_worker()


app = FastAPI(title="NULLXES Avatar inference worker", version="1.2.0", lifespan=_lifespan)
@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/longcat/generate")
async def generate(
    body: GenerateBody,
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> Response:
    expected = os.environ.get("LONGCAT_INFERENCE_SERVICE_KEY", "").strip() or None
    _check_key(expected, x_nullxes_avatar_inference_key)
    try:
        data = await asyncio.to_thread(run_generate_sync, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        msg = str(e)
        if "Configure ARACHNE" in msg or "LONGCAT" in msg or "ALLOW_INFERENCE" in msg:
            raise HTTPException(status_code=503, detail=msg[:8000]) from e
        raise HTTPException(status_code=502, detail=msg[:8000]) from e
    if data[:1] == b"{":
        try:
            obj = json.loads(data.decode("utf-8"))
            if isinstance(obj, dict) and (obj.get("videoBase64") or obj.get("video_base64")):
                return Response(content=data, media_type="application/json")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return Response(content=data, media_type="video/mp4")


@app.post("/v1/infer/jobs")
async def create_infer_job(
    body: GenerateBody,
    x_nullxes_avatar_inference_key: Optional[str] = Header(default=None, alias=EXPECTED_KEY_HEADER),
) -> dict[str, Any]:
    expected = os.environ.get("LONGCAT_INFERENCE_SERVICE_KEY", "").strip() or None
    _check_key(expected, x_nullxes_avatar_inference_key)
    try:
        _validate_body(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        job_id = job_queue.enqueue(body.model_dump(mode="json"))
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
    expected = os.environ.get("LONGCAT_INFERENCE_SERVICE_KEY", "").strip() or None
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
    expected = os.environ.get("LONGCAT_INFERENCE_SERVICE_KEY", "").strip() or None
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


@app.post("/v1/longcat/debug/decode-image")
async def debug_image(body: GenerateBody) -> dict[str, int]:
    if not body.imageBase64:
        raise HTTPException(400, "no imageBase64")
    raw = base64.b64decode(body.imageBase64)
    return {"image_bytes": len(raw)}
