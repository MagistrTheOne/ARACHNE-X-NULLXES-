"""Tests for dashboard realtime token, WebSocket, optional chat (line B)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.server.avatar_ws_frames import clear_frame_cache
from src.server.webrtc_server import create_app


def _write_tiny_mp4(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32))
    for i in range(4):
        fr = np.zeros((32, 32, 3), dtype=np.uint8)
        fr[:, :] = (10 + i * 20, 50, 200)
        w.write(fr)
    w.release()


@pytest.mark.asyncio
async def test_realtime_token_mint_no_service_key():
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/realtime/token", json={"sessionId": "ui_sess_1"})
        assert r.status == 200
        data = await r.json()
        assert "token" in data and "websocketUrl" in data
        assert data["token"] in data["websocketUrl"]
        assert data["issuedAt"].endswith("Z")
        assert data["expiresAt"].endswith("Z")


@pytest.mark.asyncio
async def test_realtime_token_requires_service_key(monkeypatch):
    monkeypatch.setenv("NULLXES_REALTIME_SERVICE_KEY", "secret_svc")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/realtime/token", json={"sessionId": "x"})
        assert r.status == 401
        r2 = await client.post(
            "/v1/realtime/token",
            json={"sessionId": "x"},
            headers={"X-NULLXES-Realtime-Service-Key": "secret_svc"},
        )
        assert r2.status == 200


@pytest.mark.asyncio
async def test_websocket_query_token_frames(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/realtime/token", json={"sessionId": "ws1"})
        tok = (await r.json())["token"]
        ws = await client.ws_connect(f"/v1/ws?token={tok}")
        try:
            m1 = await ws.receive_json()
            m2 = await ws.receive_json()
            assert m1["type"] == "session.connecting"
            assert m2["type"] == "session.connected"
            await ws.send_str(
                json.dumps({"type": "chat.send", "id": "c1", "text": "hi"})
            )
            m3 = await ws.receive_json()
            assert m3["type"] == "chat.message.received"
            assert "hi" in m3["message"]["text"]
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_websocket_avatar_stub_after_chat(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv("NULLXES_WS_AVATAR_STREAM_MODE", "stub")
    monkeypatch.setenv("NULLXES_WS_AVATAR_STREAM_CHUNK_MS", "0")
    monkeypatch.setenv("NULLXES_WS_AVATAR_STREAM_NUM_CHUNKS", "3")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/realtime/token", json={"sessionId": "ws_av"})
        tok = (await r.json())["token"]
        ws = await client.ws_connect(f"/v1/ws?token={tok}")
        try:
            await ws.receive_json()
            await ws.receive_json()
            await ws.send_str(json.dumps({"type": "chat.send", "id": "a1", "text": "hi"}))
            assert (await ws.receive_json())["type"] == "chat.message.received"
            m4 = await ws.receive_json()
            assert m4["type"] == "avatar.state.changed" and m4["state"] == "speaking"
            for seq in (1, 2, 3):
                ch = await ws.receive_json()
                assert ch["type"] == "avatar.stream.chunk"
                assert ch["seq"] == seq
                assert ch["kind"] == "video"
            mid = await ws.receive_json()
            assert mid["type"] == "avatar.state.changed" and mid["state"] == "idle"
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_websocket_avatar_video_jpeg_after_chat(monkeypatch, tmp_path):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    clear_frame_cache()
    mp4 = tmp_path / "clip.mp4"
    _write_tiny_mp4(mp4)
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_ASSET_PATH", str(mp4))
    monkeypatch.setenv("NULLXES_WS_AVATAR_STREAM_MODE", "video")
    monkeypatch.setenv("NULLXES_WS_AVATAR_VIDEO_MAX_FRAMES", "10")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/realtime/token", json={"sessionId": "ws_av_vid"})
        tok = (await r.json())["token"]
        ws = await client.ws_connect(f"/v1/ws?token={tok}")
        try:
            await ws.receive_json()
            await ws.receive_json()
            await ws.send_str(json.dumps({"type": "chat.send", "id": "v1", "text": "hi"}))
            assert (await ws.receive_json())["type"] == "chat.message.received"
            m4 = await ws.receive_json()
            assert m4["type"] == "avatar.state.changed" and m4["state"] == "speaking"
            for seq in (1, 2, 3, 4):
                ch = await ws.receive_json()
                assert ch["type"] == "avatar.stream.chunk"
                assert ch["seq"] == seq
                assert ch["kind"] == "video"
                assert ch.get("encoding") == "jpeg_base64"
                assert isinstance(ch.get("data"), str) and len(ch["data"]) > 80
            mid = await ws.receive_json()
            assert mid["type"] == "avatar.state.changed" and mid["state"] == "idle"
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_websocket_avatar_stub_disabled(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv("NULLXES_WS_AVATAR_STREAM_STUB", "0")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/realtime/token", json={"sessionId": "ws_av2"})
        tok = (await r.json())["token"]
        ws = await client.ws_connect(f"/v1/ws?token={tok}")
        try:
            await ws.receive_json()
            await ws.receive_json()
            await ws.send_str(json.dumps({"type": "chat.send", "id": "x", "text": "yo"}))
            await ws.receive_json()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.receive_json(), timeout=0.2)
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_websocket_auth_frame(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/realtime/token", json={"sessionId": "ws2"})
        tok = (await r.json())["token"]
        ws = await client.ws_connect("/v1/ws")
        try:
            await ws.send_str(
                json.dumps(
                    {"type": "auth", "token": tok, "protocolVersion": 1}
                )
            )
            m1 = await ws.receive_json()
            m2 = await ws.receive_json()
            assert m1["type"] == "session.connecting"
            assert m2["type"] == "session.connected"
        finally:
            await ws.close()


@pytest.mark.asyncio
async def test_chat_json_stub(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post(
            "/v1/chat",
            json={
                "sessionId": "s",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )
        assert r.status == 200
        data = await r.json()
        assert data["message"]["role"] == "assistant"


def test_openapi_contains_realtime_paths():
    from src.server.openapi_spec import SPEC

    assert "/v1/realtime/token" in SPEC["paths"]
    assert "/v1/ws" in SPEC["paths"]
    assert "/v1/chat" in SPEC["paths"]
    assert "/v1/avatar/preview" in SPEC["paths"]
    assert "/v1/avatar/preview/asset.mp4" in SPEC["paths"]
    assert "/v1/avatar/bootstrap" in SPEC["paths"]


@pytest.mark.asyncio
async def test_avatar_preview_stub(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv(
        "NULLXES_AVATAR_PREVIEW_VIDEO_URL",
        "https://cdn.example.com/stub.mp4",
    )
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post(
            "/v1/avatar/preview",
            json={"employeeId": "1", "sessionId": "s"},
        )
        assert r.status == 200
        data = await r.json()
        assert data["videoPreviewUrl"] == "https://cdn.example.com/stub.mp4"
        assert data["status"] == "ready"
        assert data["pipelineMode"] == "at2v_stub"
        assert data["arachneOutputProfile"] == "gpt-realtime-arachne-v1-mvp"


@pytest.mark.asyncio
async def test_avatar_preview_not_configured(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.delenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", raising=False)
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/avatar/preview", json={})
        assert r.status == 503
        data = await r.json()
        assert data["error"] == "preview_not_configured"


@pytest.mark.asyncio
async def test_avatar_preview_requires_service_key(monkeypatch):
    monkeypatch.setenv("NULLXES_REALTIME_SERVICE_KEY", "svc_secret")
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", "https://x/mp4")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/avatar/preview", json={})
        assert r.status == 401
        r2 = await client.post(
            "/v1/avatar/preview",
            json={},
            headers={"X-NULLXES-Realtime-Service-Key": "svc_secret"},
        )
        assert r2.status == 200


@pytest.mark.asyncio
async def test_avatar_preview_same_origin_asset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.delenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", raising=False)
    mp4 = tmp_path / "demo.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_ASSET_PATH", str(mp4))
    monkeypatch.setenv(
        "NULLXES_PUBLIC_HTTP_BASE",
        "https://1qs8mciim8zovo-8080.proxy.runpod.net",
    )
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/avatar/preview", json={"employeeId": "1"})
        assert r.status == 200
        data = await r.json()
        assert (
            data["videoPreviewUrl"]
            == "https://1qs8mciim8zovo-8080.proxy.runpod.net/v1/avatar/preview/asset.mp4"
        )
        g = await client.get("/v1/avatar/preview/asset.mp4")
        assert g.status == 200
        assert "video/mp4" in g.headers.get("Content-Type", "")


@pytest.mark.asyncio
async def test_avatar_bootstrap_combined(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", "https://cdn.example.com/x.mp4")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post(
            "/v1/avatar/bootstrap",
            json={"sessionId": "sb1", "employeeId": "9"},
        )
        assert r.status == 200
        data = await r.json()
        assert data["sessionId"] == "sb1"
        assert data["audioTransport"] == "gpt_realtime"
        assert data["avatarPreviewStatus"] == "ready"
        assert data["avatarPreviewCached"] is False
        assert data["videoPreviewUrl"] == "https://cdn.example.com/x.mp4"
        assert "token" in data and "websocketUrl" in data
        assert data["token"] in data["websocketUrl"]


@pytest.mark.asyncio
async def test_avatar_bootstrap_preview_cooldown_cache(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv("NULLXES_AVATAR_BOOTSTRAP_PREVIEW_COOLDOWN_SEC", "600")
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", "https://cdn.example.com/c.mp4")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r1 = await client.post(
            "/v1/avatar/bootstrap",
            json={"sessionId": "cd1", "employeeId": "1"},
        )
        r2 = await client.post(
            "/v1/avatar/bootstrap",
            json={"sessionId": "cd1", "employeeId": "1"},
        )
        assert r1.status == 200 and r2.status == 200
        d1, d2 = await r1.json(), await r2.json()
        assert d1["avatarPreviewCached"] is False
        assert d2["avatarPreviewCached"] is True
        assert d1["videoPreviewUrl"] == d2["videoPreviewUrl"]
        assert d1["token"] != d2["token"]


@pytest.mark.asyncio
async def test_avatar_bootstrap_regenerate_preview_bypasses_cache(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv("NULLXES_AVATAR_BOOTSTRAP_PREVIEW_COOLDOWN_SEC", "600")
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", "https://cdn.example.com/z.mp4")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        await client.post("/v1/avatar/bootstrap", json={"sessionId": "cd2", "employeeId": "2"})
        r2 = await client.post(
            "/v1/avatar/bootstrap",
            json={"sessionId": "cd2", "employeeId": "2", "regeneratePreview": True},
        )
        assert r2.status == 200
        assert (await r2.json())["avatarPreviewCached"] is False


@pytest.mark.asyncio
async def test_avatar_bootstrap_missing_session(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", "https://x/mp4")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/avatar/bootstrap", json={})
        assert r.status == 400


@pytest.mark.asyncio
async def test_avatar_bootstrap_preview_503_no_mint(monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.delenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", raising=False)
    monkeypatch.delenv("NULLXES_AVATAR_PREVIEW_ASSET_PATH", raising=False)
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/avatar/bootstrap", json={"sessionId": "x"})
        assert r.status == 503


@pytest.mark.asyncio
async def test_avatar_preview_video_url_overrides_asset(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    mp4 = tmp_path / "demo.mp4"
    mp4.write_bytes(b"x")
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_ASSET_PATH", str(mp4))
    monkeypatch.setenv("NULLXES_AVATAR_PREVIEW_VIDEO_URL", "https://cdn.example.com/a.mp4")
    app = create_app()
    async with TestClient(TestServer(app)) as client:
        r = await client.post("/v1/avatar/preview", json={})
        data = await r.json()
        assert data["videoPreviewUrl"] == "https://cdn.example.com/a.mp4"
