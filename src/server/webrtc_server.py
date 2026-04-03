"""aiohttp app: webhooks, sessions, media slots, pipeline worker registry."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from aiohttp import web

from src.server.media_layer import media_backend_from_env
from src.server.openapi_spec import SPEC
from src.server.realtime_api import (
    handle_avatar_preview,
    handle_avatar_preview_asset,
    handle_chat,
    handle_realtime_token,
    handle_websocket,
)
from src.server.realtime_store import RealtimeTokenStore
from src.server.session_manager import SessionManager, SessionRecord, SessionState
from src.server.session_worker import SessionWorker
from src.server.webhook_security import SIGNATURE_HEADER, TIMESTAMP_HEADER, verify_webhook

logger = logging.getLogger(__name__)

JSON_DECODER = json.JSONDecoder()
AUTO_START_ENV = "NULLXES_AUTO_START_WEBHOOK"


def _json_response(data: Dict[str, Any], status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _max_sessions() -> int:
    try:
        return max(1, int(os.environ.get("MAX_CONCURRENT_SESSIONS", "10")))
    except ValueError:
        return 10


def _auto_start_webhook() -> bool:
    return os.environ.get(AUTO_START_ENV, "1").lower() in ("1", "true", "yes")


def create_app(pipeline_cfg: Dict[str, Any] | None = None) -> web.Application:
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app["pipeline_cfg"] = dict(pipeline_cfg or {})
    max_slots = _max_sessions()
    app["session_manager"] = SessionManager(max_slots=max_slots)
    app["media_backend"] = media_backend_from_env(max_slots)
    app["webhook_secret"] = os.environ.get("NULLXES_WEBHOOK_SECRET", "").strip() or None
    app["workers"] = {}
    app["realtime_token_store"] = RealtimeTokenStore()

    app.router.add_get("/health", _handle_health)
    app.router.add_get("/v1/openapi.json", _handle_openapi)
    app.router.add_post("/v1/webhooks/session", _handle_webhook_session)
    app.router.add_post("/v1/sessions/{session_id}/start", _handle_session_start)
    app.router.add_post("/v1/sessions/{session_id}/stop", _handle_session_stop)
    app.router.add_get("/v1/sessions/{session_id}/status", _handle_session_status)
    app.router.add_patch("/v1/sessions/{session_id}/media", _handle_session_media_patch)
    app.router.add_get("/v1/media/slots", _handle_media_slots)

    app.router.add_post("/v1/realtime/token", handle_realtime_token)
    app.router.add_options("/v1/realtime/token", handle_realtime_token)
    app.router.add_post("/v1/chat", handle_chat)
    app.router.add_options("/v1/chat", handle_chat)
    app.router.add_get("/v1/avatar/preview/asset.mp4", handle_avatar_preview_asset)
    app.router.add_post("/v1/avatar/preview", handle_avatar_preview)
    app.router.add_options("/v1/avatar/preview", handle_avatar_preview)
    app.router.add_get("/v1/ws", handle_websocket)

    app.on_startup.append(_on_startup)
    return app


async def _on_startup(app: web.Application) -> None:
    backend = app["media_backend"]
    if hasattr(backend, "ensure_slots"):
        backend.ensure_slots()
    logger.info("NULLXES server startup: media slots ensured (backend=%s)", type(backend).__name__)


def _workers(app: web.Application) -> Dict[str, SessionWorker]:
    return app["workers"]


async def _handle_health(_request: web.Request) -> web.Response:
    return _json_response({"status": "ok"})


async def _handle_openapi(_request: web.Request) -> web.Response:
    return web.json_response(SPEC)


async def _handle_webhook_session(request: web.Request) -> web.Response:
    raw = await request.read()
    ts = request.headers.get(TIMESTAMP_HEADER)
    sig = request.headers.get(SIGNATURE_HEADER)
    secret = request.app["webhook_secret"]
    ok, reason = verify_webhook(secret, raw, ts, sig)
    if not ok:
        logger.warning("webhook verify failed: %s", reason)
        return _json_response({"error": "unauthorized", "detail": reason}, status=401)

    try:
        body = JSON_DECODER.decode(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return _json_response({"error": "invalid_json", "detail": str(e)}, status=400)

    if not isinstance(body, dict):
        return _json_response({"error": "invalid_body"}, status=400)

    event = body.get("event") or body.get("type")
    ext_sid = body.get("session_id") or body.get("external_session_id")
    if not event or not ext_sid:
        return _json_response({"error": "missing_event_or_session_id"}, status=400)

    sm: SessionManager = request.app["session_manager"]
    idem = request.headers.get("Idempotency-Key")
    rec, status = sm.upsert_from_webhook(
        external_session_id=str(ext_sid),
        event=str(event),
        correlation_id=body.get("correlation_id"),
        config=body.get("config") if isinstance(body.get("config"), dict) else {},
        callback_url=body.get("callback_url"),
        idempotency_key=idem,
    )
    if rec is None:
        return _json_response({"error": "capacity_full"}, status=507)

    if _auto_start_webhook() and str(event).lower() in (
        "interview.session.created",
        "session.start",
        "interview.created",
    ):
        ok_start, _ = sm.start(rec.nullxes_session_id)
        if ok_start:
            w = _ensure_worker(request.app, rec)
            await w.start()

    return _json_response(
        {
            "nullxes_session_id": rec.nullxes_session_id,
            "media_slot": rec.media_slot,
            "status": "accepted" if status == "created" else "existing",
        },
        status=202,
    )


def _ensure_worker(app: web.Application, rec: SessionRecord) -> SessionWorker:
    workers = _workers(app)
    w = workers.get(rec.nullxes_session_id)
    if w is None:
        w = SessionWorker(app, rec)
        workers[rec.nullxes_session_id] = w
    return w


async def _handle_session_start(request: web.Request) -> web.Response:
    sid = request.match_info["session_id"]
    sm: SessionManager = request.app["session_manager"]
    ok, msg = sm.start(sid)
    if not ok:
        if msg == "not_found":
            return _json_response({"error": msg}, status=404)
        return _json_response({"error": msg}, status=400)
    rec = sm.get(sid)
    if not rec:
        return _json_response({"error": "not_found"}, status=404)
    w = _ensure_worker(request.app, rec)
    await w.start()
    return _json_response({"ok": True, "state": rec.state.value})


async def _handle_session_stop(request: web.Request) -> web.Response:
    sid = request.match_info["session_id"]
    sm: SessionManager = request.app["session_manager"]
    workers = _workers(request.app)
    ok, msg = sm.stop(sid)
    if not ok:
        return _json_response({"error": msg}, status=404)
    w = workers.get(sid)
    if w and w.running:
        w.request_stop()
        await w.wait_done(timeout=float(os.environ.get("NULLXES_WORKER_STOP_TIMEOUT", "30")))
    elif msg == "cancelled_scheduled":
        workers.pop(sid, None)
    return _json_response({"ok": True, "detail": msg})


async def _handle_session_status(request: web.Request) -> web.Response:
    sid = request.match_info["session_id"]
    sm: SessionManager = request.app["session_manager"]
    rec = sm.get(sid)
    if not rec:
        return _json_response({"error": "not_found"}, status=404)
    payload = sm.to_status_dict(rec)
    w = _workers(request.app).get(sid)
    payload["worker_running"] = bool(w and w.running)
    return _json_response(payload)


async def _handle_session_media_patch(request: web.Request) -> web.Response:
    sid = request.match_info["session_id"]
    sm: SessionManager = request.app["session_manager"]
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _json_response({"error": "invalid_json"}, status=400)
    if not isinstance(body, dict):
        return _json_response({"error": "invalid_body"}, status=400)
    ok, msg = sm.patch_media(sid, body)
    if not ok:
        return _json_response({"error": msg}, status=404)
    rec = sm.get(sid)
    return _json_response({"ok": True, "media_binding": dict(rec.media_binding) if rec else {}})


async def _handle_media_slots(request: web.Request) -> web.Response:
    sm: SessionManager = request.app["session_manager"]
    backend = request.app["media_backend"]
    sessions = sm.slot_snapshot()
    media = backend.snapshot() if hasattr(backend, "snapshot") else []
    return _json_response({"sessions": sessions, "media": media})
