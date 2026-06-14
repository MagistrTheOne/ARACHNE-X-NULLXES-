"""
Production boot contract for ARACHNE-X-ULTRA-V3 NIGHTCORE.

When NULLXES_PRODUCTION=1, fail closed on missing secrets, dev stubs, and legacy paths.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

Role = Literal["worker", "orchestrator", "any"]

PRODUCTION_ENV = "NULLXES_PRODUCTION"

# Shared bans when production=1
_BANNED_WHEN_PROD: tuple[tuple[str, frozenset[str]], ...] = (
    ("ARACHNE_LEGACY_STREAMING", frozenset({"1", "true", "yes", "on"})),
    ("ALLOW_INFERENCE_DEV_MOCK", frozenset({"1", "true", "yes", "on"})),
    ("NULLXES_ALLOW_DEV_STUB", frozenset({"1", "true", "yes", "on"})),
    ("NULLXES_CHAT_ASSISTANT_FIXED_REPLY", frozenset()),  # any non-empty
    ("NULLXES_WS_CHAT_ASSISTANT_FIXED_REPLY", frozenset()),
)

_WORKER_REQUIRED: tuple[str, ...] = (
    "NULLXES_INFERENCE_SERVICE_KEY",
    "NULLXES_AVATAR_INFERENCE_SERVICE_KEY",
    "LONGCAT_INFERENCE_SERVICE_KEY",
)

_ORCHESTRATOR_REQUIRED: tuple[str, ...] = (
    "NULLXES_REALTIME_SERVICE_KEY",
    "NULLXES_AVATAR_INFERENCE_URL",
)


def is_production() -> bool:
    return os.environ.get(PRODUCTION_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _env_nonempty(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _first_set(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return None


def validate_production_boot(*, role: Role = "any") -> None:
    """
    Raise RuntimeError if production env contract is violated.

    role:
      worker — inference worker (FastAPI arachnex-worker)
      orchestrator — src/server webrtc stack
      any — all checks applicable to the caller
    """
    if not is_production():
        return

    errors: list[str] = []

    for env_name, banned_values in _BANNED_WHEN_PROD:
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        if banned_values:
            if raw.lower() in banned_values:
                errors.append(f"{env_name} must not be enabled in production (got {raw!r})")
        else:
            errors.append(f"{env_name} must be unset in production")

    if role in ("worker", "any"):
        if _first_set(*_WORKER_REQUIRED) is None:
            errors.append(
                "One of NULLXES_INFERENCE_SERVICE_KEY, "
                "NULLXES_AVATAR_INFERENCE_SERVICE_KEY, or LONGCAT_INFERENCE_SERVICE_KEY is required"
            )

    if role in ("orchestrator", "any"):
        if not _env_nonempty("NULLXES_REALTIME_SERVICE_KEY"):
            errors.append("NULLXES_REALTIME_SERVICE_KEY is required in production")
        if not _env_nonempty("NULLXES_AVATAR_INFERENCE_URL"):
            errors.append("NULLXES_AVATAR_INFERENCE_URL is required in production")

        ws_mode = os.environ.get("NULLXES_WS_AVATAR_STREAM_MODE", "").strip().lower()
        if ws_mode and ws_mode not in ("inference", "off"):
            errors.append(
                f"NULLXES_WS_AVATAR_STREAM_MODE must be inference or off in production (got {ws_mode!r})"
            )
        elif not ws_mode and not _env_truthy("NULLXES_ALLOW_DEV_STUB"):
            # Default effective mode should be inference when URL is set
            pass

    if errors:
        raise RuntimeError(
            "NULLXES production boot contract failed:\n  - " + "\n  - ".join(errors)
        )


def dev_stub_allowed() -> bool:
    """True when explicit dev stub flag is set and not in production."""
    if is_production():
        return False
    return _env_truthy("NULLXES_ALLOW_DEV_STUB")


def legacy_streaming_allowed() -> bool:
    if is_production():
        return False
    return _env_truthy("ARACHNE_LEGACY_STREAMING")
