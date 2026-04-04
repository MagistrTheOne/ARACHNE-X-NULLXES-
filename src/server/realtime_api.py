"""Dashboard realtime: mint token, WebSocket, optional HTTP chat (line B MVP)."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from aiohttp import web

import base64

import numpy as np

from src.server.avatar_inference_client import inference_base_url
from src.server.avatar_stream_client import stream_avatar_jpeg_frames
from src.server.avatar_ws_frames import load_jpeg_frames_from_mp4
from src.server.tts_runner import synthesize_pcm_f32_16k
from src.server.realtime_store import RealtimeTokenStore

logger = logging.getLogger(__name__)

JSON_DECODER = json.JSONDecoder()
SERVICE_KEY_ENV = "NULLXES_REALTIME_SERVICE_KEY"
SERVICE_KEY_HEADER = "X-NULLXES-Realtime-Service-Key"
TTL_ENV = "NULLXES_REALTIME_TOKEN_TTL_SEC"
PUBLIC_HTTP_ENV = "NULLXES_PUBLIC_HTTP_BASE"
PUBLIC_WS_ENV = "NULLXES_PUBLIC_WS_BASE"
CORS_ORIGIN_ENV = "NULLXES_CORS_ORIGIN"
AVATAR_PREVIEW_VIDEO_ENV = "NULLXES_AVATAR_PREVIEW_VIDEO_URL"
AVATAR_PREVIEW_ASSET_ENV = "NULLXES_AVATAR_PREVIEW_ASSET_PATH"
AVATAR_PREVIEW_PROFILE_ENV = "NULLXES_ARACHNE_OUTPUT_PROFILE"
# Same-origin mp4 for <video src>; build URL with NULLXES_PUBLIC_HTTP_BASE + this path.
AVATAR_PREVIEW_ASSET_URL_PATH = "/v1/avatar/preview/asset.mp4"
BOOTSTRAP_PREVIEW_COOLDOWN_ENV = "NULLXES_AVATAR_BOOTSTRAP_PREVIEW_COOLDOWN_SEC"
WS_AVATAR_STREAM_STUB_ENV = "NULLXES_WS_AVATAR_STREAM_STUB"
WS_AVATAR_STREAM_MODE_ENV = "NULLXES_WS_AVATAR_STREAM_MODE"
WS_AVATAR_STREAM_CHUNK_MS_ENV = "NULLXES_WS_AVATAR_STREAM_CHUNK_MS"
WS_AVATAR_STREAM_NUM_CHUNKS_ENV = "NULLXES_WS_AVATAR_STREAM_NUM_CHUNKS"
AVATAR_INFERENCE_TASK_ENV = "NULLXES_AVATAR_INFERENCE_TASK"
AVATAR_INFERENCE_IMAGE_ENV = "NULLXES_AVATAR_INFERENCE_IMAGE_BASE64"
WS_AUTH_TIMEOUT_SEC = 12.0
WS_CLOSE_AUTH = 4401
PROTOCOL_VERSION = 1

CHAT_ASSISTANT_FIXED_REPLY_ENV = "NULLXES_CHAT_ASSISTANT_FIXED_REPLY"
WS_CHAT_ASSISTANT_FIXED_REPLY_ENV = "NULLXES_WS_CHAT_ASSISTANT_FIXED_REPLY"
ROUTE_VIA_SESSION_WORKER_ENV = "NULLXES_REALTIME_ROUTE_VIA_SESSION_WORKER"


def _http_chat_assistant_text(content: str) -> str:
    fixed = os.environ.get(CHAT_ASSISTANT_FIXED_REPLY_ENV, "").strip()
    if fixed:
        return fixed[:8000]
    return (content or "")[:8000]


def _ws_assistant_reply_text(user_text: str) -> str:
    fixed = os.environ.get(WS_CHAT_ASSISTANT_FIXED_REPLY_ENV, "").strip()
    if fixed:
        return fixed[:8000]
    return (user_text or "")[:8000]


def _service_key(app: web.Application) -> Optional[str]:
    k = os.environ.get(SERVICE_KEY_ENV, "").strip()
    return k or None


def _verify_service_request(request: web.Request) -> bool:
    expected = _service_key(request.app)
    if not expected:
        logger.warning("%s unset — realtime token/chat auth skipped (dev only)", SERVICE_KEY_ENV)
        return True
    hdr = request.headers.get(SERVICE_KEY_HEADER, "").strip()
    if hdr and hdr == expected:
        return True
    auth = request.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer ") and auth[7:].strip() == expected:
        return True
    return False


def _ttl_sec() -> int:
    try:
        return max(60, int(os.environ.get(TTL_ENV, "900")))
    except ValueError:
        return 900


def _iso_z(ts: float) -> str:
    """UTC ISO-8601 with Z (second precision enough for MVP)."""
    return (
        datetime.datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"
    )


def _public_http_base(request: web.Request) -> str:
    """Browser-facing http(s) origin for links returned in JSON (proxy-safe)."""
    env_base = os.environ.get(PUBLIC_HTTP_ENV, "").strip().rstrip("/")
    if env_base:
        return env_base
    xf_proto = request.headers.get("X-Forwarded-Proto", "").strip().lower()
    scheme = xf_proto if xf_proto in ("http", "https") else request.scheme
    host = request.headers.get("Host", "localhost:8080")
    return f"{scheme}://{host}"


def _public_ws_base(request: web.Request) -> str:
    base = os.environ.get(PUBLIC_WS_ENV, "").strip().rstrip("/")
    if base:
        return base
    http_base = os.environ.get(PUBLIC_HTTP_ENV, "").strip().rstrip("/")
    if http_base:
        if http_base.startswith("https://"):
            return "wss://" + http_base[len("https://") :]
        if http_base.startswith("http://"):
            return "ws://" + http_base[len("http://") :]
    scheme = "wss" if request.scheme == "https" else "ws"
    host = request.headers.get("Host", "localhost:8080")
    return f"{scheme}://{host}"


def _cors_headers(request: web.Request) -> Dict[str, str]:
    origin = os.environ.get(CORS_ORIGIN_ENV, "").strip()
    if not origin:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, "
        + SERVICE_KEY_HEADER,
        "Access-Control-Max-Age": "86400",
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _handle_realtime_token_options(request: web.Request) -> web.Response:
    return web.Response(status=204, headers=_cors_headers(request))


async def handle_realtime_token(request: web.Request) -> web.Response:
    if request.method == "OPTIONS":
        return await _handle_realtime_token_options(request)
    cors = _cors_headers(request)
    if not _verify_service_request(request):
        return web.json_response({"error": "unauthorized"}, status=401, headers=cors)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid_json"}, status=400, headers=cors)
    if not isinstance(body, dict):
        return web.json_response({"error": "invalid_body"}, status=400, headers=cors)
    sid = body.get("sessionId") or body.get("session_id")
    if not sid or not isinstance(sid, str):
        return web.json_response({"error": "missing_sessionId"}, status=400, headers=cors)
    emp = body.get("employeeId") or body.get("employee_id")
    emp_s = str(emp) if emp is not None else None
    nx = body.get("nullxesSessionId") or body.get("nullxes_session_id")
    nx_s = str(nx) if nx is not None else None

    store: RealtimeTokenStore = request.app["realtime_token_store"]
    token, iat, exp = store.mint(
        str(sid),
        employee_id=emp_s,
        nullxes_session_id=nx_s,
        ttl_sec=_ttl_sec(),
    )
    ws_base = _public_ws_base(request)
    ws_url = f"{ws_base}/v1/ws?token={token}"
    payload = {
        "token": token,
        "websocketUrl": ws_url,
        "issuedAt": _iso_z(iat),
        "expiresAt": _iso_z(exp),
    }
    return web.json_response(payload, headers=cors)


async def _handle_chat_options(request: web.Request) -> web.Response:
    return web.Response(status=204, headers=_cors_headers(request))


async def handle_chat(request: web.Request) -> web.Response:
    if request.method == "OPTIONS":
        return await _handle_chat_options(request)
    cors = _cors_headers(request)
    if not _verify_service_request(request):
        return web.json_response({"error": "unauthorized"}, status=401, headers=cors)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid_json"}, status=400, headers=cors)
    if not isinstance(body, dict):
        return web.json_response({"error": "invalid_body"}, status=400, headers=cors)
    sid = body.get("sessionId") or body.get("session_id")
    if not sid or not isinstance(sid, str):
        return web.json_response({"error": "missing_sessionId"}, status=400, headers=cors)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return web.json_response({"error": "missing_messages"}, status=400, headers=cors)
    last = messages[-1]
    content = ""
    if isinstance(last, dict):
        content = str(last.get("content") or "")
    stream = bool(body.get("stream"))
    if stream:
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                **cors,
            },
        )
        await resp.prepare(request)
        chunk = json.dumps({"delta": _http_chat_assistant_text(content)[:200]}, ensure_ascii=False)
        await resp.write(f"data: {chunk}\n\n".encode("utf-8"))
        await resp.write_eof()
        return resp
    return web.json_response(
        {
            "message": {
                "id": f"chat_{_now_ms()}",
                "role": "assistant",
                "content": _http_chat_assistant_text(content),
            }
        },
        headers=cors,
    )


async def _handle_avatar_preview_options(request: web.Request) -> web.Response:
    return web.Response(status=204, headers=_cors_headers(request))


async def _handle_avatar_bootstrap_options(request: web.Request) -> web.Response:
    return web.Response(status=204, headers=_cors_headers(request))


def _bootstrap_preview_cooldown_sec() -> int:
    try:
        return max(0, int(os.environ.get(BOOTSTRAP_PREVIEW_COOLDOWN_ENV, "0")))
    except ValueError:
        return 0


def _bootstrap_preview_cache_key(session_id: str, employee_id: Optional[str]) -> str:
    return f"{session_id}\x1f{employee_id or ''}"


def _static_avatar_video_fields_or_error(
    request: web.Request, cors: Dict[str, str]
) -> tuple[Optional[Dict[str, Any]], Optional[web.Response]]:
    """
    Resolve videoPreviewUrl from configured static URL or same-origin asset. No audio assets.
    Returns (fields dict, None) or (None, error Response).
    """
    video_url = os.environ.get(AVATAR_PREVIEW_VIDEO_ENV, "").strip()
    asset_path = os.environ.get(AVATAR_PREVIEW_ASSET_ENV, "").strip()
    if not video_url and asset_path:
        if not os.path.isfile(asset_path):
            return None, web.json_response(
                {
                    "error": "preview_asset_missing",
                    "detail": f"{AVATAR_PREVIEW_ASSET_ENV} is not a readable file: {asset_path}",
                },
                status=503,
                headers=cors,
            )
        base = _public_http_base(request)
        video_url = base + AVATAR_PREVIEW_ASSET_URL_PATH
    if not video_url:
        return None, web.json_response(
            {
                "error": "preview_not_configured",
                "detail": (
                    f"Set {AVATAR_PREVIEW_VIDEO_ENV} to a full HTTPS mp4 URL, or set "
                    f"{AVATAR_PREVIEW_ASSET_ENV} to a local mp4 path and "
                    f"{PUBLIC_HTTP_ENV} (e.g. RunPod proxy https origin) for same-origin preview."
                ),
            },
            status=503,
            headers=cors,
        )

    profile = os.environ.get(AVATAR_PREVIEW_PROFILE_ENV, "gpt-realtime-arachne-v1-mvp").strip()
    if not profile:
        profile = "gpt-realtime-arachne-v1-mvp"

    return {
        "videoPreviewUrl": video_url,
        "pipelineMode": "static_preview",
        "arachneOutputProfile": profile,
    }, None


async def handle_avatar_preview_asset(request: web.Request) -> web.Response:
    """Public GET: stream local mp4 for same-origin videoPreviewUrl (no service key)."""
    path = os.environ.get(AVATAR_PREVIEW_ASSET_ENV, "").strip()
    cors = _cors_headers(request)
    if not path or not os.path.isfile(path):
        return web.Response(
            status=404,
            text="avatar preview asset not configured or missing\n",
            headers=cors,
        )
    return web.FileResponse(
        path,
        headers={
            "Content-Type": "video/mp4",
            "Cache-Control": "public, max-age=3600",
            **cors,
        },
    )


async def handle_avatar_preview(request: web.Request) -> web.Response:
    """
    Dashboard avatar preview (line B): returns a public mp4 URL without running infer.

    Real DiT / at2v can replace this later; body fields are accepted for forward compatibility.
    """
    if request.method == "OPTIONS":
        return await _handle_avatar_preview_options(request)
    cors = _cors_headers(request)
    if not _verify_service_request(request):
        return web.json_response({"error": "unauthorized"}, status=401, headers=cors)
    raw = await request.read()
    if not raw.strip():
        body: Dict[str, Any] = {}
    else:
        try:
            body = JSON_DECODER.decode(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return web.json_response({"error": "invalid_json"}, status=400, headers=cors)
    if not isinstance(body, dict):
        return web.json_response({"error": "invalid_body"}, status=400, headers=cors)

    preview_fields, err = _static_avatar_video_fields_or_error(request, cors)
    if err is not None:
        return err

    # Optional for future at2v/infer: employeeId, sessionId, imageUrl, speakText (snake_case ok).

    payload = {**preview_fields, "status": "ready"}
    return web.json_response(payload, headers=cors)


async def handle_avatar_bootstrap(request: web.Request) -> web.Response:
    """
    One server-to-server call: mint WebSocket token + static video preview URL.
    Audio is expected via GPT Realtime (or other path), not as upload/wav assets here.

    pipelineMode is static_preview unless you add a separate inference preview path.
    When generation is wired, use NULLXES_AVATAR_BOOTSTRAP_PREVIEW_COOLDOWN_SEC to avoid
    re-running it on every bootstrap; token is still minted each call.
    """
    if request.method == "OPTIONS":
        return await _handle_avatar_bootstrap_options(request)
    cors = _cors_headers(request)
    if not _verify_service_request(request):
        return web.json_response({"error": "unauthorized"}, status=401, headers=cors)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "invalid_json"}, status=400, headers=cors)
    if not isinstance(body, dict):
        return web.json_response({"error": "invalid_body"}, status=400, headers=cors)
    sid = body.get("sessionId") or body.get("session_id")
    if not sid or not isinstance(sid, str):
        return web.json_response({"error": "missing_sessionId"}, status=400, headers=cors)

    emp = body.get("employeeId") or body.get("employee_id")
    emp_s = str(emp) if emp is not None else None

    force_preview = bool(
        body.get("regeneratePreview")
        or body.get("forceAvatarRefresh")
        or body.get("regenerate_preview")
    )
    cd_sec = _bootstrap_preview_cooldown_sec()
    cache: Dict[str, tuple[float, Dict[str, Any]]] = request.app["avatar_bootstrap_preview_cache"]
    ckey = _bootstrap_preview_cache_key(str(sid), emp_s)
    preview_cached = False

    if cd_sec > 0 and not force_preview:
        now = time.time()
        hit = cache.get(ckey)
        if hit is not None and (now - hit[0]) < cd_sec:
            preview_fields = hit[1]
            preview_cached = True
        else:
            preview_fields, err = _static_avatar_video_fields_or_error(request, cors)
            if err is not None:
                return err
            cache[ckey] = (now, dict(preview_fields))
    else:
        if force_preview:
            cache.pop(ckey, None)
        preview_fields, err = _static_avatar_video_fields_or_error(request, cors)
        if err is not None:
            return err
        if cd_sec > 0:
            cache[ckey] = (time.time(), dict(preview_fields))
    nx = body.get("nullxesSessionId") or body.get("nullxes_session_id")
    nx_s = str(nx) if nx is not None else None

    store: RealtimeTokenStore = request.app["realtime_token_store"]
    token, iat, exp = store.mint(
        str(sid),
        employee_id=emp_s,
        nullxes_session_id=nx_s,
        ttl_sec=_ttl_sec(),
    )
    ws_base = _public_ws_base(request)
    ws_url = f"{ws_base}/v1/ws?token={token}"

    payload = {
        "sessionId": str(sid),
        "token": token,
        "websocketUrl": ws_url,
        "issuedAt": _iso_z(iat),
        "expiresAt": _iso_z(exp),
        **preview_fields,
        "avatarPreviewStatus": "ready",
        "audioTransport": "gpt_realtime",
        "avatarPreviewCached": preview_cached,
    }
    return web.json_response(payload, headers=cors)


async def handle_websocket(request: web.Request) -> web.StreamResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    store: RealtimeTokenStore = request.app["realtime_token_store"]
    q_token = request.query.get("token", "").strip()
    rec = store.peek(q_token) if q_token else None

    if rec is None:
        try:
            await _wait_auth_frame(ws, store)
        except (AuthError, json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
            logger.info("ws auth failed: %s", e)
            await _send_error(ws, "auth_failed")
            await ws.close(code=WS_CLOSE_AUTH, message="invalid or expired token")
            return ws
        rec = getattr(ws, "_realtime_rec", None)

    if rec is None:
        await _send_error(ws, "auth_failed")
        await ws.close(code=WS_CLOSE_AUTH, message="invalid or expired token")
        return ws

    await _send_json(ws, {"type": "session.connecting", "at": _now_ms()})
    await _send_json(ws, {"type": "session.connected", "at": _now_ms()})

    nx_sid = getattr(rec, "nullxes_session_id", None)
    if isinstance(nx_sid, str) and nx_sid.strip():
        asyncio.create_task(
            _pump_avatar_out_queue(ws, nx_sid.strip(), request.app),
            name=f"ws-pump-{nx_sid.strip()[:8]}",
        )

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await _handle_ws_text(ws, msg.data, rec, request.app)
            elif msg.type == web.WSMsgType.ERROR:
                logger.warning("ws error: %s", ws.exception())
                break
    finally:
        if not ws.closed:
            try:
                await _send_json(
                    ws,
                    {
                        "type": "session.disconnected",
                        "at": _now_ms(),
                        "reason": "client_close",
                    },
                )
            except Exception:
                pass
            await ws.close()
    return ws


class AuthError(Exception):
    pass


async def _wait_auth_frame(ws: web.WebSocketResponse, store: RealtimeTokenStore) -> None:
    deadline = time.monotonic() + WS_AUTH_TIMEOUT_SEC
    while time.monotonic() < deadline:
        msg = await ws.receive(timeout=max(0.1, deadline - time.monotonic()))
        if msg.type == web.WSMsgType.TEXT:
            data = JSON_DECODER.decode(msg.data)
            if not isinstance(data, dict):
                raise AuthError("not_object")
            if data.get("type") != "auth":
                raise AuthError("expected_auth")
            pv = data.get("protocolVersion")
            if pv is not None:
                try:
                    if int(pv) != PROTOCOL_VERSION:
                        raise AuthError("bad_protocol_version")
                except (TypeError, ValueError) as e:
                    raise AuthError("bad_protocol_version") from e
            t = (data.get("token") or "").strip()
            if not t:
                raise AuthError("missing_token")
            rec = store.peek(t)
            if not rec:
                raise AuthError("bad_token")
            ws._realtime_rec = rec  # type: ignore[attr-defined]
            return
        if msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
            raise AuthError("closed")
    raise AuthError("timeout")


async def _send_json(ws: web.WebSocketResponse, obj: Dict[str, Any]) -> None:
    await ws.send_str(json.dumps(obj, ensure_ascii=False))


async def _send_error(ws: web.WebSocketResponse, message: str) -> None:
    await _send_json(ws, {"type": "session.error", "at": _now_ms(), "message": message})


def _route_chat_via_session_worker() -> bool:
    v = os.environ.get(ROUTE_VIA_SESSION_WORKER_ENV, "1").strip().lower()
    return v in ("1", "true", "yes")


async def _pump_avatar_out_queue(
    ws: web.WebSocketResponse, nx_sid: str, app: web.Application
) -> None:
    """Drain SessionWorker.out_queue (and future producers) to this WebSocket."""
    while not ws.closed:
        qmap = app.get("avatar_frame_queues")
        if not isinstance(qmap, dict):
            await asyncio.sleep(0.25)
            continue
        q = qmap.get(nx_sid)
        if q is None:
            await asyncio.sleep(0.2)
            continue
        try:
            msg = await asyncio.wait_for(q.get(), timeout=120.0)
        except asyncio.TimeoutError:
            continue
        if msg is None:
            continue
        try:
            await _send_json(ws, msg)
        except (ConnectionResetError, ConnectionError):
            break


def _ws_avatar_stream_effective_mode() -> str:
    """
    off — no avatar chain after chat.
    stub — metadata-only avatar.stream.chunk (NUM_CHUNKS / CHUNK_MS); dev/tests only.
    video — JPEG base64 from NULLXES_AVATAR_PREVIEW_ASSET_PATH (local mp4).
    inference — GPU worker: NULLXES_AVATAR_INFERENCE_URL → NDJSON JPEG stream (no MP4).

    NULLXES_WS_AVATAR_STREAM_STUB=0 forces off (backward compatible).
    If MODE is unset: inference when INFERENCE_URL set; else video if preview asset file exists; else off.
    """
    stub_raw = os.environ.get(WS_AVATAR_STREAM_STUB_ENV, "1").strip().lower()
    if stub_raw in ("0", "false", "no", "off"):
        return "off"
    explicit = os.environ.get(WS_AVATAR_STREAM_MODE_ENV, "").strip().lower()
    if explicit in ("off", "stub", "video", "inference"):
        return explicit
    if inference_base_url():
        return "inference"
    asset = os.environ.get(AVATAR_PREVIEW_ASSET_ENV, "").strip()
    if asset and os.path.isfile(asset):
        return "video"
    return "off"


def _ws_avatar_stream_timing() -> tuple[int, float]:
    try:
        n = max(1, min(60, int(os.environ.get(WS_AVATAR_STREAM_NUM_CHUNKS_ENV, "5"))))
    except ValueError:
        n = 5
    try:
        ms = max(0, min(500, int(os.environ.get(WS_AVATAR_STREAM_CHUNK_MS_ENV, "40"))))
    except ValueError:
        ms = 40
    return n, ms / 1000.0


async def _video_avatar_ws_playback(
    ws: web.WebSocketResponse, frames_b64: list[str], fps: float
) -> None:
    """Emit speaking → one chunk per JPEG frame → idle."""
    delay_s = 1.0 / fps if fps > 0 else 1.0 / 30.0
    try:
        if ws.closed:
            return
        await _send_json(
            ws,
            {"type": "avatar.state.changed", "at": _now_ms(), "state": "speaking"},
        )
        for seq, data in enumerate(frames_b64, start=1):
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if ws.closed:
                return
            await _send_json(
                ws,
                {
                    "type": "avatar.stream.chunk",
                    "at": _now_ms(),
                    "kind": "video",
                    "seq": seq,
                    "encoding": "jpeg_base64",
                    "data": data,
                },
            )
        if ws.closed:
            return
        await _send_json(
            ws,
            {"type": "avatar.state.changed", "at": _now_ms(), "state": "idle"},
        )
    except Exception:
        logger.debug("ws avatar video playback stopped", exc_info=True)


async def _stub_avatar_ws_playback(ws: web.WebSocketResponse) -> None:
    """
    Metadata-only chunks (no payload). Used when MODE=stub or as fallback when video decode fails.
    Цепочка после chat.send — см. Documentation/D_SAAS/ARACHNE_X_FRONTEND_CONTRACT.md §5–6.
    """
    n_chunks, delay_s = _ws_avatar_stream_timing()
    try:
        if ws.closed:
            return
        await _send_json(
            ws,
            {"type": "avatar.state.changed", "at": _now_ms(), "state": "speaking"},
        )
        for seq in range(1, n_chunks + 1):
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if ws.closed:
                return
            await _send_json(
                ws,
                {
                    "type": "avatar.stream.chunk",
                    "at": _now_ms(),
                    "kind": "video",
                    "seq": seq,
                },
            )
        if ws.closed:
            return
        await _send_json(
            ws,
            {"type": "avatar.state.changed", "at": _now_ms(), "state": "idle"},
        )
    except Exception:
        logger.debug("ws avatar stub playback stopped", exc_info=True)


async def _inference_avatar_ws_playback(
    ws: web.WebSocketResponse,
    app: web.Application,
    prompt: str,
    session_id: str,
    *,
    task_override: str | None = None,
    img_b64: str | None = None,
    audio_b64: str | None = None,
    cont_b64: str | None = None,
    num_seg: int | None = None,
    ref_idx: int | None = None,
) -> None:
    del task_override, audio_b64, cont_b64, num_seg, ref_idx
    env_img = os.environ.get(AVATAR_INFERENCE_IMAGE_ENV, "").strip() or None
    use_img = img_b64 or env_img
    if not use_img:
        logger.warning("inference avatar: no imageBase64; set %s", AVATAR_INFERENCE_IMAGE_ENV)
        await _send_error(ws, "avatar_inference_missing_image")
        return
    http = app.get("avatar_http_session")
    if http is None or http.closed:
        await _send_error(ws, "avatar_http_unavailable")
        return
    use_prompt = (prompt or "").strip() or "Professional avatar speaking clearly, natural motion."
    pipeline_cfg = app.get("pipeline_cfg") or {}
    tts_cfg = dict(pipeline_cfg.get("tts") or {})
    try:
        audio_f32 = await asyncio.to_thread(synthesize_pcm_f32_16k, use_prompt, tts_cfg)
    except Exception as e:
        logger.warning("inference path TTS failed: %s", e)
        await _send_error(ws, "tts_failed")
        return
    if audio_f32.size == 0:
        await _send_error(ws, "tts_empty")
        return
    audio_b64_out = base64.b64encode(np.asarray(audio_f32, dtype=np.float32).tobytes()).decode(
        "ascii"
    )
    try:
        await _send_json(
            ws,
            {"type": "avatar.state.changed", "at": _now_ms(), "state": "speaking"},
        )
        seq = 0
        async for _seq, jpeg_b64 in stream_avatar_jpeg_frames(
            http,
            prompt=use_prompt,
            session_id=session_id or "session",
            image_base64=str(use_img),
            audio_float32_base64=audio_b64_out,
        ):
            if ws.closed:
                break
            seq += 1
            await _send_json(
                ws,
                {
                    "type": "avatar.stream.chunk",
                    "at": _now_ms(),
                    "kind": "video",
                    "seq": seq,
                    "encoding": "jpeg_base64",
                    "data": jpeg_b64,
                },
            )
        if not ws.closed:
            await _send_json(
                ws,
                {"type": "avatar.state.changed", "at": _now_ms(), "state": "idle"},
            )
    except Exception:
        logger.exception("avatar inference stream failed")
        await _send_error(ws, "avatar_inference_failed")
        if not ws.closed:
            await _send_json(
                ws,
                {"type": "avatar.state.changed", "at": _now_ms(), "state": "idle"},
            )


async def _run_avatar_ws_playback(
    ws: web.WebSocketResponse,
    mode: str,
    asset_path: str,
    *,
    app: web.Application,
    chat_text: str,
    session_id: str,
    inference_opts: Dict[str, Any] | None = None,
) -> None:
    if mode == "inference":
        inf = inference_opts or {}
        await _inference_avatar_ws_playback(
            ws,
            app,
            chat_text,
            session_id,
            task_override=(str(inf["task"]) if inf.get("task") else None),
            img_b64=(str(inf["imageBase64"]) if inf.get("imageBase64") else None),
            audio_b64=(str(inf["audioBase64"]) if inf.get("audioBase64") else None),
            cont_b64=(str(inf["continuationState"]) if inf.get("continuationState") else None),
            num_seg=(int(inf["numSegments"]) if inf.get("numSegments") is not None else None),
            ref_idx=(int(inf["refImgIndex"]) if inf.get("refImgIndex") is not None else None),
        )
        return
    if mode == "video":
        if asset_path and os.path.isfile(asset_path):
            frames, fps = load_jpeg_frames_from_mp4(asset_path)
            if frames:
                await _video_avatar_ws_playback(ws, frames, fps)
                return
            logger.info("avatar ws: video mode but zero frames from %s, using stub", asset_path)
        else:
            logger.debug("avatar ws: video mode but no readable asset path, using stub")
    if mode in ("stub", "video"):
        await _stub_avatar_ws_playback(ws)


async def _handle_ws_text(
    ws: web.WebSocketResponse, raw: str, rec: Any, app: web.Application
) -> None:
    try:
        data = JSON_DECODER.decode(raw)
    except json.JSONDecodeError:
        logger.debug("ws ignore non-json frame")
        return
    if not isinstance(data, dict):
        return
    typ = data.get("type")
    if typ == "auth":
        return
    if typ == "chat.send":
        text = str(data.get("text") or "")
        cid = str(data.get("id") or f"c_{_now_ms()}")
        nx = getattr(rec, "nullxes_session_id", None)
        nx_s = str(nx).strip() if nx else ""
        workers = app.get("workers") or {}
        w = workers.get(nx_s) if nx_s else None
        if _route_chat_via_session_worker() and w is not None and w.running:
            await w.enqueue_user_text(text)
            return
        await _send_json(
            ws,
            {
                "type": "chat.message.received",
                "at": _now_ms(),
                "message": {
                    "id": f"reply_{cid}",
                    "from": "assistant",
                    "text": _ws_assistant_reply_text(text),
                },
            },
        )
        m = _ws_avatar_stream_effective_mode()
        if m != "off":
            ap = os.environ.get(AVATAR_PREVIEW_ASSET_ENV, "").strip()
            sid = getattr(rec, "session_id", "") or ""
            inf_opts = data.get("inference") if isinstance(data.get("inference"), dict) else None
            asyncio.create_task(
                _run_avatar_ws_playback(
                    ws,
                    m,
                    ap,
                    app=app,
                    chat_text=text,
                    session_id=sid,
                    inference_opts=inf_opts,
                )
            )
        return
    if typ == "voice.pcm16":
        raw_b64 = data.get("data") or data.get("pcmBase64")
        if not isinstance(raw_b64, str) or not raw_b64.strip():
            return
        try:
            pcm = base64.b64decode(raw_b64)
        except (ValueError, TypeError):
            return
        nx = getattr(rec, "nullxes_session_id", None)
        nx_s = str(nx).strip() if nx else ""
        workers = app.get("workers") or {}
        w = workers.get(nx_s) if nx_s else None
        if w is not None and w.running:
            await w.feed_pcm16(pcm)
        return
    if typ == "session.disconnect":
        await ws.close()
        return
    if typ == "voice.mute":
        return
    logger.debug("ws unknown type=%s", typ)
