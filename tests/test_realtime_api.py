"""Tests for dashboard realtime token, WebSocket, optional chat (line B)."""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.server.webrtc_server import create_app


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
