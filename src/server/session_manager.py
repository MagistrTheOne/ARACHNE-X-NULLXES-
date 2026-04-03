"""Session FSM, slot allocation, idempotency (in-process MVP store)."""

from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SessionState(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class SessionRecord:
    nullxes_session_id: str
    external_session_id: str
    media_slot: int
    state: SessionState = SessionState.SCHEDULED
    event: str = ""
    correlation_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None
    media_binding: Dict[str, Any] = field(default_factory=dict)
    degraded: bool = False
    degraded_reason: Optional[str] = None
    media_errors: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    health: Dict[str, str] = field(
        default_factory=lambda: {"stt": "unknown", "llm": "unknown", "tts": "unknown", "avatar": "unknown"}
    )


class SessionManager:
    def __init__(self, max_slots: int = 10) -> None:
        self._max = max(1, max_slots)
        self._sessions: Dict[str, SessionRecord] = {}
        self._by_external: Dict[str, str] = {}
        self._slot_owner: Dict[int, str] = {}
        self._idempotency: Dict[str, str] = {}

    def _alloc_slot(self) -> Optional[int]:
        for i in range(self._max):
            if i not in self._slot_owner:
                return i
        return None

    def _touch(self, rec: SessionRecord) -> None:
        rec.updated_at = time.time()

    def touch_record(self, rec: SessionRecord) -> None:
        self._touch(rec)

    def get(self, nullxes_session_id: str) -> Optional[SessionRecord]:
        return self._sessions.get(nullxes_session_id)

    def upsert_from_webhook(
        self,
        *,
        external_session_id: str,
        event: str,
        correlation_id: Optional[str],
        config: Dict[str, Any],
        callback_url: Optional[str],
        idempotency_key: Optional[str],
    ) -> tuple[Optional[SessionRecord], str]:
        """
        Returns (record, status) where status is 'created' | 'existing' | 'capacity'.
        """
        if idempotency_key and idempotency_key in self._idempotency:
            sid = self._idempotency[idempotency_key]
            rec = self._sessions.get(sid)
            if rec:
                return rec, "existing"

        if external_session_id in self._by_external:
            sid = self._by_external[external_session_id]
            rec = self._sessions.get(sid)
            if rec:
                return rec, "existing"

        slot = self._alloc_slot()
        if slot is None:
            return None, "capacity"

        sid = f"nx_{uuid.uuid4().hex[:16]}"
        rec = SessionRecord(
            nullxes_session_id=sid,
            external_session_id=external_session_id,
            media_slot=slot,
            event=event,
            correlation_id=correlation_id,
            config=dict(config) if config else {},
            callback_url=callback_url,
        )
        self._sessions[sid] = rec
        self._by_external[external_session_id] = sid
        self._slot_owner[slot] = sid
        if idempotency_key:
            self._idempotency[idempotency_key] = sid
        return rec, "created"

    def start(self, nullxes_session_id: str) -> tuple[bool, str]:
        rec = self._sessions.get(nullxes_session_id)
        if not rec:
            return False, "not_found"
        if rec.state in (SessionState.STOPPED, SessionState.FAILED):
            return False, "terminal_state"
        if rec.state == SessionState.RUNNING:
            return True, "already_running"
        rec.state = SessionState.RUNNING
        self._touch(rec)
        return True, "ok"

    def stop(self, nullxes_session_id: str) -> tuple[bool, str]:
        rec = self._sessions.get(nullxes_session_id)
        if not rec:
            return False, "not_found"
        if rec.state == SessionState.STOPPED:
            return True, "already_stopped"
        if rec.state == SessionState.SCHEDULED:
            self.finalize_stop(nullxes_session_id)
            return True, "cancelled_scheduled"
        rec.state = SessionState.DRAINING
        self._touch(rec)
        return True, "draining"

    def finalize_stop(self, nullxes_session_id: str) -> None:
        rec = self._sessions.get(nullxes_session_id)
        if not rec:
            return
        slot = rec.media_slot
        if slot in self._slot_owner and self._slot_owner[slot] == nullxes_session_id:
            del self._slot_owner[slot]
        rec.state = SessionState.STOPPED
        self._touch(rec)

    def mark_failed(self, nullxes_session_id: str, reason: str) -> None:
        rec = self._sessions.get(nullxes_session_id)
        if not rec:
            return
        slot = rec.media_slot
        if slot in self._slot_owner and self._slot_owner[slot] == nullxes_session_id:
            del self._slot_owner[slot]
        rec.state = SessionState.FAILED
        rec.degraded_reason = reason
        self._touch(rec)

    def set_degraded(self, nullxes_session_id: str, reason: str) -> None:
        rec = self._sessions.get(nullxes_session_id)
        if not rec:
            return
        rec.degraded = True
        rec.degraded_reason = reason
        self._touch(rec)

    def patch_media(self, nullxes_session_id: str, binding: Dict[str, Any]) -> tuple[bool, str]:
        rec = self._sessions.get(nullxes_session_id)
        if not rec:
            return False, "not_found"
        rec.media_binding.update(binding)
        self._touch(rec)
        return True, "ok"

    def slot_snapshot(self) -> list[dict]:
        rows = []
        for i in range(self._max):
            owner = self._slot_owner.get(i)
            rows.append(
                {
                    "slot": i,
                    "occupied": owner is not None,
                    "nullxes_session_id": owner,
                    "default_sink_name": f"nx_slot_{i}",
                    "default_monitor_name": f"nx_slot_{i}.monitor",
                }
            )
        return rows

    def to_status_dict(self, rec: SessionRecord) -> dict:
        return {
            "nullxes_session_id": rec.nullxes_session_id,
            "external_session_id": rec.external_session_id,
            "state": rec.state.value,
            "media_slot": rec.media_slot,
            "degraded": rec.degraded,
            "degraded_reason": rec.degraded_reason,
            "media_errors": rec.media_errors,
            "media_binding": dict(rec.media_binding),
            "health": dict(rec.health),
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }
