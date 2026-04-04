"""HTTP client for GPU avatar inference worker: returns generated MP4 bytes."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

import aiohttp


INFERENCE_URL_ENV = "NULLXES_AVATAR_INFERENCE_URL"
INFERENCE_PATH_ENV = "NULLXES_AVATAR_INFERENCE_PATH"
INFERENCE_KEY_ENV = "NULLXES_AVATAR_INFERENCE_SERVICE_KEY"
INFERENCE_KEY_HEADER = "X-NULLXES-Avatar-Inference-Key"
INFERENCE_TIMEOUT_ENV = "NULLXES_AVATAR_INFERENCE_TIMEOUT_SEC"


def inference_base_url() -> str:
    return os.environ.get(INFERENCE_URL_ENV, "").strip().rstrip("/")


def inference_generate_path() -> str:
    p = os.environ.get(INFERENCE_PATH_ENV, "/v1/longcat/generate").strip() or "/v1/longcat/generate"
    return p if p.startswith("/") else f"/{p}"


def _timeout_sec() -> int:
    try:
        return max(30, min(7200, int(os.environ.get(INFERENCE_TIMEOUT_ENV, "600"))))
    except ValueError:
        return 600


def _service_key() -> Optional[str]:
    k = os.environ.get(INFERENCE_KEY_ENV, "").strip()
    return k or None


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
) -> bytes:
    """POST JSON to worker; expect video/mp4 body or JSON with videoBase64."""
    base = inference_base_url()
    if not base:
        raise RuntimeError(f"{INFERENCE_URL_ENV} is not set")

    url = base + inference_generate_path()
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

    headers: dict[str, str] = {"Accept": "video/mp4, application/json"}
    sk = _service_key()
    if sk:
        headers[INFERENCE_KEY_HEADER] = sk

    timeout = aiohttp.ClientTimeout(total=_timeout_sec())
    close_session = False
    session = client_session
    if session is None or session.closed:
        session = aiohttp.ClientSession(timeout=timeout)
        close_session = True
    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
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
            await session.close()


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
) -> bytes:
    """Backward-compatible name; delegates to avatar_generate_mp4_bytes."""
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
    )
