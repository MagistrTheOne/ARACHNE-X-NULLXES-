"""In-memory opaque realtime tokens for dashboard WebSocket (line B MVP)."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RealtimeTokenRecord:
    session_id: str
    employee_id: Optional[str]
    nullxes_session_id: Optional[str]
    expires_at: float


class RealtimeTokenStore:
    def __init__(self) -> None:
        self._by_token: dict[str, RealtimeTokenRecord] = {}

    def _purge_expired(self) -> None:
        now = time.time()
        dead = [t for t, r in self._by_token.items() if r.expires_at <= now]
        for t in dead:
            del self._by_token[t]

    def mint(
        self,
        session_id: str,
        *,
        employee_id: Optional[str] = None,
        nullxes_session_id: Optional[str] = None,
        ttl_sec: int = 900,
    ) -> tuple[str, float, float]:
        self._purge_expired()
        now = time.time()
        exp = now + max(60, ttl_sec)
        token = secrets.token_urlsafe(32)
        self._by_token[token] = RealtimeTokenRecord(
            session_id=session_id,
            employee_id=employee_id,
            nullxes_session_id=nullxes_session_id,
            expires_at=exp,
        )
        return token, now, exp

    def peek(self, token: str) -> Optional[RealtimeTokenRecord]:
        """Validate without consuming (e.g. logging)."""
        self._purge_expired()
        rec = self._by_token.get(token)
        if rec is None or rec.expires_at <= time.time():
            return None
        return rec
