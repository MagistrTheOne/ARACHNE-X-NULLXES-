"""HTTP client for GPU avatar inference worker: returns generated MP4 bytes."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any, Optional

import aiohttp


INFERENCE_URL_ENV = "NULLXES_AVATAR_INFERENCE_URL"
INFERENCE_PATH_ENV = "NULLXES_AVATAR_INFERENCE_PATH"
INFERENCE_KEY_ENV = "NULLXES_AVATAR_INFERENCE_SERVICE_KEY"
INFERENCE_KEY_HEADER = "X-NULLXES-Avatar-Inference-Key"
INFERENCE_TIMEOUT_ENV = "NULLXES_AVATAR_INFERENCE_TIMEOUT_SEC"
INFERENCE_ASYNC_ENV = "NULLXES_AVATAR_INFERENCE_ASYNC"
INFERENCE_JOBS_PATH_ENV = "NULLXES_AVATAR_INFERENCE_JOBS_PATH"
INFERENCE_POLL_MS_ENV = "NULLXES_AVATAR_INFERENCE_POLL_MS"


def inference_base_url() -> str:
    return os.environ.get(INFERENCE_URL_ENV, "").strip().rstrip("/")


def inference_generate_path() -> str:
    p = os.environ.get(INFERENCE_PATH_ENV, "/v1/arachne/generate").strip() or "/v1/arachne/generate"
    return p if p.startswith("/") else f"/{p}"


def inference_jobs_path() -> str:
    p = os.environ.get(INFERENCE_JOBS_PATH_ENV, "/v1/infer/jobs").strip() or "/v1/infer/jobs"
    return p if p.startswith("/") else f"/{p}"


def _timeout_sec() -> int:
    try:
        return max(30, min(7200, int(os.environ.get(INFERENCE_TIMEOUT_ENV, "600"))))
    except ValueError:
        return 600


def _poll_ms() -> int:
    try:
        return max(100, min(10000, int(os.environ.get(INFERENCE_POLL_MS_ENV, "500"))))
    except ValueError:
        return 500


def _service_key() -> Optional[str]:
    for env_name in (
        "NULLXES_INFERENCE_SERVICE_KEY",
        INFERENCE_KEY_ENV,
        "LONGCAT_INFERENCE_SERVICE_KEY",
    ):
        k = os.environ.get(env_name, "").strip()
        if k:
            return k
    return None


def _use_async_jobs() -> bool:
    return os.environ.get(INFERENCE_ASYNC_ENV, "").strip().lower() in ("1", "true", "yes")


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json, video/mp4"}
    sk = _service_key()
    if sk:
        headers[INFERENCE_KEY_HEADER] = sk
    return headers


def _build_payload(
    *,
    prompt: str,
    session_id: str,
    task: str,
    image_base64: Optional[str],
    audio_base64: Optional[str],
    continuation_state: Optional[str],
    num_segments: Optional[int],
    ref_img_index: Optional[int],
    negative_prompt: Optional[str],
    input_json: Optional[dict[str, Any]],
    embed_audio: Optional[bool] = None,
    output_mode: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    text_guidance_scale: Optional[float] = None,
    audio_guidance_scale: Optional[float] = None,
    resolution: Optional[str] = None,
    num_frames: Optional[int] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": task,
        "prompt": prompt[:8000],
        "sessionId": session_id,
    }
    if image_base64:
        payload["imageBase64"] = image_base64
    if audio_base64:
        payload["audioBase64"] = audio_base64
    if continuation_state:
        payload["continuationState"] = continuation_state
    if num_segments is not None:
        payload["numSegments"] = int(num_segments)
    if ref_img_index is not None:
        payload["refImgIndex"] = int(ref_img_index)
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if input_json is not None:
        payload["inputJson"] = input_json
    if embed_audio is not None:
        payload["embedAudio"] = bool(embed_audio)
    if output_mode:
        payload["outputMode"] = output_mode
    if num_inference_steps is not None:
        payload["numInferenceSteps"] = int(num_inference_steps)
    if text_guidance_scale is not None:
        payload["textGuidanceScale"] = float(text_guidance_scale)
    if audio_guidance_scale is not None:
        payload["audioGuidanceScale"] = float(audio_guidance_scale)
    if resolution:
        payload["resolution"] = resolution
    if num_frames is not None:
        payload["numFrames"] = int(num_frames)
    return payload


async def _avatar_via_job_queue(
    session: aiohttp.ClientSession,
    base: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    total_timeout_sec: int,
) -> bytes:
    jobs_url = base + inference_jobs_path()
    deadline = time.monotonic() + float(total_timeout_sec)
    async with session.post(jobs_url, json=payload, headers=headers) as resp:
        body = await resp.read()
        if resp.status >= 400:
            raise RuntimeError(f"avatar infer job enqueue HTTP {resp.status}: {body[:500]!r}")
        obj = json.loads(body.decode("utf-8"))
        job_id = obj.get("jobId") or obj.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("avatar infer: missing jobId in response")
    status_path = f"/v1/infer/jobs/{job_id}"
    result_path = f"/v1/infer/jobs/{job_id}/result"
    poll_s = _poll_ms() / 1000.0
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError("avatar infer: timeout waiting for job")
        async with session.get(base + status_path, headers=headers) as r:
            sb = await r.read()
            if r.status != 200:
                raise RuntimeError(f"avatar infer status HTTP {r.status}: {sb[:300]!r}")
            st = json.loads(sb.decode("utf-8"))
        status = st.get("status")
        if status == "failed":
            err = st.get("error") or "inference failed"
            raise RuntimeError(f"avatar infer job failed: {err[:2000]}")
        if status == "done":
            break
        await asyncio.sleep(poll_s)
    async with session.get(base + result_path, headers={**headers, "Accept": "video/mp4"}) as r2:
        mp4 = await r2.read()
        if r2.status != 200:
            raise RuntimeError(f"avatar infer result HTTP {r2.status}: {mp4[:500]!r}")
    return mp4


async def avatar_generate_mp4_bytes(
    client_session: Optional[aiohttp.ClientSession],
    *,
    prompt: str,
    session_id: str,
    task: str = "text-to-video",
    image_base64: Optional[str] = None,
    audio_base64: Optional[str] = None,
    continuation_state: Optional[str] = None,
    num_segments: Optional[int] = None,
    ref_img_index: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    input_json: Optional[dict[str, Any]] = None,
    embed_audio: Optional[bool] = None,
    output_mode: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    text_guidance_scale: Optional[float] = None,
    audio_guidance_scale: Optional[float] = None,
    resolution: Optional[str] = None,
    num_frames: Optional[int] = None,
) -> bytes:
    """POST to worker: sync generate or async job queue (see NULLXES_AVATAR_INFERENCE_ASYNC)."""
    base = inference_base_url()
    if not base:
        raise RuntimeError(f"{INFERENCE_URL_ENV} is not set")

    payload = _build_payload(
        prompt=prompt,
        session_id=session_id,
        task=task,
        image_base64=image_base64,
        audio_base64=audio_base64,
        continuation_state=continuation_state,
        num_segments=num_segments,
        ref_img_index=ref_img_index,
        negative_prompt=negative_prompt,
        input_json=input_json,
        embed_audio=embed_audio,
        output_mode=output_mode,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        resolution=resolution,
        num_frames=num_frames,
    )
    headers = _auth_headers()
    total_timeout = _timeout_sec()

    close_session = False
    sess = client_session
    if sess is None or sess.closed:
        sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=total_timeout + 30))
        close_session = True
    try:
        if _use_async_jobs():
            return await _avatar_via_job_queue(sess, base, payload, headers, total_timeout)

        url = base + inference_generate_path()
        async with sess.post(
            url,
            json=payload,
            headers={**headers, "Accept": "video/mp4, application/json"},
            timeout=aiohttp.ClientTimeout(total=total_timeout),
        ) as resp:
            body = await resp.read()
            if resp.status >= 400:
                raise RuntimeError(
                    f"avatar inference HTTP {resp.status}: {body[:500]!r}"
                )
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "json" in ct:
                obj = json.loads(body.decode("utf-8"))
                if not isinstance(obj, dict):
                    raise RuntimeError("avatar inference: JSON body is not an object")
                b64 = obj.get("videoBase64") or obj.get("video_base64")
                if not isinstance(b64, str) or not b64.strip():
                    raise RuntimeError("avatar inference: missing videoBase64 in JSON")
                return base64.b64decode(b64)
            return body
    finally:
        if close_session:
            await sess.close()


async def longcat_generate_mp4_bytes(
    client_session: Optional[aiohttp.ClientSession],
    *,
    prompt: str,
    session_id: str,
    task: str = "text-to-video",
    image_base64: Optional[str] = None,
    continuation_state: Optional[str] = None,
    audio_base64: Optional[str] = None,
    num_segments: Optional[int] = None,
    ref_img_index: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    input_json: Optional[dict[str, Any]] = None,
    embed_audio: Optional[bool] = None,
    output_mode: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    text_guidance_scale: Optional[float] = None,
    audio_guidance_scale: Optional[float] = None,
    resolution: Optional[str] = None,
    num_frames: Optional[int] = None,
) -> bytes:
    """Legacy alias (historical import name); delegates to avatar_generate_mp4_bytes."""
    return await avatar_generate_mp4_bytes(
        client_session,
        prompt=prompt,
        session_id=session_id,
        task=task,
        image_base64=image_base64,
        audio_base64=audio_base64,
        continuation_state=continuation_state,
        num_segments=num_segments,
        ref_img_index=ref_img_index,
        negative_prompt=negative_prompt,
        input_json=input_json,
        embed_audio=embed_audio,
        output_mode=output_mode,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        resolution=resolution,
        num_frames=num_frames,
    )
