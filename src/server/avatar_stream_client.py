"""HTTP client: streaming avatar JPEG frames (NDJSON) from GPU worker — no MP4."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

INFERENCE_URL_ENV = "NULLXES_AVATAR_INFERENCE_URL"
INFERENCE_FRAMES_PATH_ENV = "NULLXES_AVATAR_INFERENCE_FRAMES_PATH"
INFERENCE_KEY_ENV = "NULLXES_AVATAR_INFERENCE_SERVICE_KEY"
INFERENCE_KEY_HEADER = "X-NULLXES-Avatar-Inference-Key"
INFERENCE_TIMEOUT_ENV = "NULLXES_AVATAR_INFERENCE_TIMEOUT_SEC"


def inference_base_url() -> str:
    return os.environ.get(INFERENCE_URL_ENV, "").strip().rstrip("/")


def inference_frames_path() -> str:
    p = os.environ.get(INFERENCE_FRAMES_PATH_ENV, "/v1/realtime/avatar_frames").strip()
    return p if p.startswith("/") else f"/{p}"


def _timeout_sec() -> int:
    try:
        return max(60, min(7200, int(os.environ.get(INFERENCE_TIMEOUT_ENV, "900"))))
    except ValueError:
        return 900


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
) -> AsyncIterator[Tuple[int, Dict[str, Any]]]:
    """
    POST NDJSON stream. Yields (seq, frame_payload).
    """
    base = inference_base_url()
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
    try:
        async with sess.post(url, json=body, headers=headers, timeout=timeout) as resp:
            if resp.status >= 400:
                raw = await resp.read()
                raise RuntimeError(f"avatar frames HTTP {resp.status}: {raw[:500]!r}")
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
    finally:
        if close_session and not sess.closed:
            await sess.close()
