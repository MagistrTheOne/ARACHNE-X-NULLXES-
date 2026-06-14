"""CPU contract tests for ARACHNE-X-ULTRA-V3 NIGHTCORE prod_guard."""

from __future__ import annotations

import os

import pytest


def test_prod_guard_skips_when_not_production(monkeypatch):
    monkeypatch.delenv("NULLXES_PRODUCTION", raising=False)
    from arachne_x.runtime.prod_guard import validate_production_boot

    validate_production_boot(role="any")


def test_prod_guard_fails_missing_worker_key(monkeypatch):
    monkeypatch.setenv("NULLXES_PRODUCTION", "1")
    for k in (
        "NULLXES_INFERENCE_SERVICE_KEY",
        "NULLXES_AVATAR_INFERENCE_SERVICE_KEY",
        "LONGCAT_INFERENCE_SERVICE_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    from arachne_x.runtime.prod_guard import validate_production_boot

    with pytest.raises(RuntimeError, match="INFERENCE_SERVICE_KEY"):
        validate_production_boot(role="worker")


def test_prod_guard_bans_legacy_streaming(monkeypatch):
    monkeypatch.setenv("NULLXES_PRODUCTION", "1")
    monkeypatch.setenv("NULLXES_INFERENCE_SERVICE_KEY", "k")
    monkeypatch.setenv("ARACHNE_LEGACY_STREAMING", "1")
    from arachne_x.runtime.prod_guard import validate_production_boot

    with pytest.raises(RuntimeError, match="ARACHNE_LEGACY_STREAMING"):
        validate_production_boot(role="worker")


def test_prod_guard_orchestrator_requires_realtime_key(monkeypatch):
    monkeypatch.setenv("NULLXES_PRODUCTION", "1")
    monkeypatch.delenv("NULLXES_REALTIME_SERVICE_KEY", raising=False)
    monkeypatch.setenv("NULLXES_AVATAR_INFERENCE_URL", "http://worker:8000")
    from arachne_x.runtime.prod_guard import validate_production_boot

    with pytest.raises(RuntimeError, match="NULLXES_REALTIME_SERVICE_KEY"):
        validate_production_boot(role="orchestrator")


def test_dev_stub_allowed_only_outside_production(monkeypatch):
    monkeypatch.delenv("NULLXES_PRODUCTION", raising=False)
    monkeypatch.setenv("NULLXES_ALLOW_DEV_STUB", "1")
    from arachne_x.runtime.prod_guard import dev_stub_allowed

    assert dev_stub_allowed() is True

    monkeypatch.setenv("NULLXES_PRODUCTION", "1")
    assert dev_stub_allowed() is False
