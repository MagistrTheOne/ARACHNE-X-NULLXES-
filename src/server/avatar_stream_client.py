"""HTTP client: streaming avatar JPEG frames (NDJSON) from GPU worker — no MP4."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import aiohttp

from src.server.avatar_worker_router import route_worker_base_url

logger = logging.getLogger(__name__)

INFERENCE_URL_ENV = "NULLXES_AVATAR_INFERENCE_URL"
INFERENCE_FRAMES_PATH_ENV = "NULLXES_AVATAR_INFERENCE_FRAMES_PATH"
INFERENCE_KEY_ENV = "NULLXES_AVATAR_INFERENCE_SERVICE_KEY"
INFERENCE_KEY_HEADER = "X-NULLXES-Avatar-Inference-Key"
INFERENCE_TIMEOUT_ENV = "NULLXES_AVATAR_INFERENCE_TIMEOUT_SEC"
INFERENCE_RETRY_MAX_ENV = "NULLXES_AVATAR_INFERENCE_RETRY_MAX"
INFERENCE_RETRY_JITTER_MS_ENV = "NULLXES_AVATAR_INFERENCE_RETRY_JITTER_MS"


def inference_base_url(session_id: Optional[str] = None) -> str:
    if session_id:
        try:
            return route_worker_base_url(session_id)
        except RuntimeError:
            pass
    return os.environ.get(INFERENCE_URL_ENV, "").strip().rstrip("/")


def inference_frames_path() -> str:
    p = os.environ.get(INFERENCE_FRAMES_PATH_ENV, "/v1/realtime/avatar_frames").strip()
    return p if p.startswith("/") else f"/{p}"


def _timeout_sec() -> int:
    try:
        return max(60, min(7200, int(os.environ.get(INFERENCE_TIMEOUT_ENV, "900"))))
    except ValueError:
        return 900


def _retry_max() -> int:
    try:
        return max(0, min(8, int(os.environ.get(INFERENCE_RETRY_MAX_ENV, "3"))))
    except ValueError:
        return 3


def _retry_jitter_ms() -> int:
    try:
        return max(0, min(5000, int(os.environ.get(INFERENCE_RETRY_JITTER_MS_ENV, "250"))))
    except ValueError:
        return 250


def _service_key() -> str:
    for env_name in (
        "NULLXES_INFERENCE_SERVICE_KEY",
        INFERENCE_KEY_ENV,
    ):
        v = os.environ.get(env_name, "").strip()
        if v:
            return v
    return ""


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/x-ndjson, application/json"}
    sk = _service_key()
    if sk:
        headers[INFERENCE_KEY_HEADER] = sk
    return headers


def _parse_error_detail(raw: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"error": raw[:500].decode("utf-8", errors="replace")}
    detail = obj.get("detail")
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, str):
        return {"error": detail}
    if isinstance(obj, dict) and obj.get("error"):
        return obj
    return {"error": str(obj)[:2000]}


def _retry_after_ms(detail: dict[str, Any], *, default_ms: int = 8000) -> int:
    try:
        return max(100, min(120_000, int(detail.get("retryAfterMs", default_ms))))
    except (TypeError, ValueError):
        return default_ms


def _should_retry_status(status: int, detail: dict[str, Any]) -> bool:
    if status not in (429, 503):
        return False
    err = str(detail.get("error") or "")
    return err in (
        "worker_busy",
        "worker_draining",
        "worker_offline",
        "queue_timeout",
    ) or status == 429


async def _sleep_retry(retry_after_ms: int) -> None:
    jitter = random.randint(0, _retry_jitter_ms())
    await asyncio.sleep((retry_after_ms + jitter) / 1000.0)


async def _iter_ndjson_frames(resp: aiohttp.ClientResponse) -> AsyncIterator[Tuple[int, Dict[str, Any]]]:
    buf = b""
    async for chunk in resp.content.iter_chunked(65536):
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            if obj.get("error"):
                raise RuntimeError(str(obj.get("error"))[:2000])
            seq = int(obj.get("seq", 0))
            payload: Dict[str, Any] = {
                "encoding": obj.get("encoding") or ("jpeg_base64" if obj.get("jpegBase64") else None),
                "data": obj.get("frameBase64") or obj.get("jpegBase64") or obj.get("data"),
                "width": obj.get("width"),
                "height": obj.get("height"),
                "tsMs": obj.get("tsMs"),
            }
            if isinstance(payload.get("data"), str) and payload["data"]:
                yield seq, payload
    tail = buf.strip()
    if tail:
        obj = json.loads(tail.decode("utf-8"))
        if obj.get("error"):
            raise RuntimeError(str(obj.get("error"))[:2000])
        seq = int(obj.get("seq", 0))
        payload2: Dict[str, Any] = {
            "encoding": obj.get("encoding") or ("jpeg_base64" if obj.get("jpegBase64") else None),
            "data": obj.get("frameBase64") or obj.get("jpegBase64") or obj.get("data"),
            "width": obj.get("width"),
            "height": obj.get("height"),
            "tsMs": obj.get("tsMs"),
        }
        if isinstance(payload2.get("data"), str) and payload2["data"]:
            yield seq, payload2


async def stream_avatar_frames(
    client_session: Optional[aiohttp.ClientSession],
    *,
    prompt: str,
    session_id: str,
    image_base64: str,
    audio_pcm16_base64: Optional[str] = None,
    audio_float32_base64: Optional[str] = None,
    negative_prompt: str = "",
    num_inference_steps: int = 8,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
    resolution: str = "480p",
    num_frames: int = 93,
    engine: Optional[str] = None,
    worker_base_url: Optional[str] = None,
) -> AsyncIterator[Tuple[int, Dict[str, Any]]]:
    """
    POST NDJSON stream. Yields (seq, frame_payload).
    Retries worker_busy / queue_timeout with retryAfterMs from worker JSON body.
    """
    base = (worker_base_url or inference_base_url(session_id)).strip().rstrip("/")
    if not base:
        raise RuntimeError(f"{INFERENCE_URL_ENV} is not set")

    body: Dict[str, Any] = {
        "sessionId": session_id,
        "prompt": prompt[:8000],
        "imageBase64": image_base64,
        "numInferenceSteps": int(num_inference_steps),
        "textGuidanceScale": float(text_guidance_scale),
        "audioGuidanceScale": float(audio_guidance_scale),
        "resolution": resolution,
        "numFrames": int(num_frames),
    }
    if audio_pcm16_base64:
        body["audioPcm16Base64"] = audio_pcm16_base64
    elif audio_float32_base64:
        body["audioFloat32Base64"] = audio_float32_base64
    else:
        raise ValueError("audio_pcm16_base64 or audio_float32_base64 is required")
    if negative_prompt:
        body["negativePrompt"] = negative_prompt[:8000]
    if engine:
        body["engine"] = str(engine).strip().lower()

    url = base + inference_frames_path()
    headers = _auth_headers()
    timeout = aiohttp.ClientTimeout(total=_timeout_sec())
    close_session = False
    sess = client_session
    if sess is None or sess.closed:
        sess = aiohttp.ClientSession(timeout=timeout)
        close_session = True

    attempt = 0
    max_attempts = _retry_max() + 1
    try:
        while attempt < max_attempts:
            attempt += 1
            try:
                async with sess.post(url, json=body, headers=headers, timeout=timeout) as resp:
                    if resp.status >= 400:
                        raw = await resp.read()
                        detail = _parse_error_detail(raw)
                        if attempt < max_attempts and _should_retry_status(resp.status, detail):
                            retry_ms = _retry_after_ms(detail)
                            logger.warning(
                                "avatar_frames retry session_id=%s attempt=%s/%s status=%s error=%s retry_ms=%s",
                                session_id,
                                attempt,
                                max_attempts,
                                resp.status,
                                detail.get("error"),
                                retry_ms,
                            )
                            await _sleep_retry(retry_ms)
                            continue
                        err_msg = detail.get("error") or repr(raw[:500])
                        raise RuntimeError(f"avatar frames HTTP {resp.status}: {err_msg}")
                    async for item in _iter_ndjson_frames(resp):
                        yield item
                    return
            except aiohttp.ClientError as e:
                if attempt < max_attempts:
                    logger.warning(
                        "avatar_frames transport_retry session_id=%s attempt=%s/%s err=%s",
                        session_id,
                        attempt,
                        max_attempts,
                        e,
                    )
                    await _sleep_retry(8000)
                    continue
                raise RuntimeError(f"avatar frames transport error: {e}") from e
        raise RuntimeError("avatar frames exhausted retries")
    finally:
        if close_session and not sess.closed:
            await sess.close()
