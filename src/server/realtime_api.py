"""Dashboard realtime: mint token, WebSocket, optional HTTP chat (line B MVP)."""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from aiohttp import web

from src.server.realtime_store import RealtimeTokenStore

logger = logging.getLogger(__name__)

JSON_DECODER = json.JSONDecoder()
SERVICE_KEY_ENV = "NULLXES_REALTIME_SERVICE_KEY"
SERVICE_KEY_HEADER = "X-NULLXES-Realtime-Service-Key"
TTL_ENV = "NULLXES_REALTIME_TOKEN_TTL_SEC"
PUBLIC_HTTP_ENV = "NULLXES_PUBLIC_HTTP_BASE"
PUBLIC_WS_ENV = "NULLXES_PUBLIC_WS_BASE"
CORS_ORIGIN_ENV = "NULLXES_CORS_ORIGIN"
WS_AUTH_TIMEOUT_SEC = 12.0
WS_CLOSE_AUTH = 4401
PROTOCOL_VERSION = 1


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
        chunk = json.dumps({"delta": f"[stub] {content[:200]}"}, ensure_ascii=False)
        await resp.write(f"data: {chunk}\n\n".encode("utf-8"))
        await resp.write_eof()
        return resp
    return web.json_response(
        {
            "message": {
                "id": f"chat_{_now_ms()}",
                "role": "assistant",
                "content": f"[stub] {content[:500]}",
            }
        },
        headers=cors,
    )


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

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await _handle_ws_text(ws, msg.data, rec)
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


async def _handle_ws_text(ws: web.WebSocketResponse, raw: str, _rec: Any) -> None:
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
        await _send_json(
            ws,
            {
                "type": "chat.message.received",
                "at": _now_ms(),
                "message": {
                    "id": f"reply_{cid}",
                    "from": "assistant",
                    "text": f"[stub] {text[:500]}",
                },
            },
        )
        return
    if typ == "session.disconnect":
        await ws.close()
        return
    if typ == "voice.mute":
        return
    logger.debug("ws unknown type=%s", typ)
