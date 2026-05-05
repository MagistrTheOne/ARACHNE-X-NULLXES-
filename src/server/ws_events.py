"""Small helpers for websocket event payloads (runtime-only)."""

from __future__ import annotations

import time
from typing import Any, Optional


def ws_event_base(*, session_id: str, meeting_id: Optional[str] = None) -> dict[str, Any]:
    """
    Existing WS base contract:
    - includes monotonic-ish wall clock timestamp for UI ordering (ms)
    - includes sessionId, and optional meetingId
    """
    ev: dict[str, Any] = {"at": int(time.time() * 1000), "sessionId": str(session_id)}
    if meeting_id:
        ev["meetingId"] = str(meeting_id)
    return ev

