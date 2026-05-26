"""Unit tests for session → worker hash routing."""

from __future__ import annotations

import os

import pytest

from src.server.avatar_worker_router import route_worker_base_url, route_worker_index, worker_base_urls


def test_single_worker_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NULLXES_AVATAR_INFERENCE_URL", "http://gpu-a:9090")
    monkeypatch.delenv("NULLXES_AVATAR_WORKER_URLS", raising=False)
    assert worker_base_urls() == ["http://gpu-a:9090"]
    assert route_worker_base_url("sess-1") == "http://gpu-a:9090"


def test_hash_routing_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "NULLXES_AVATAR_WORKER_URLS",
        "http://gpu-a:9090,http://gpu-b:9090,http://gpu-c:9090",
    )
    idx1, url1 = route_worker_index("meeting-42")
    idx2, url2 = route_worker_index("meeting-42")
    assert idx1 == idx2
    assert url1 == url2
    assert url1 in worker_base_urls()
