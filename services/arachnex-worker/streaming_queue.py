"""
Explicit GPU streaming admission queue for POST /v1/realtime/avatar_frames.

Replaces implicit blocking on a global threading lock with:
  admit → wait (bounded) → active slot → release

Env:
  ARACHNE_STREAM_MAX_ACTIVE_JOBS (default 1)
  ARACHNE_STREAM_MAX_QUEUE (default 3)
  ARACHNE_STREAM_QUEUE_TIMEOUT_SEC (default 15)
  ARACHNE_STREAM_ESTIMATED_JOB_MS (default 8000) — wait estimate for retryAfterMs
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Optional


class WorkerLifecycle(str, Enum):
    starting = "starting"
    active = "active"
    draining = "draining"
    offline = "offline"


class StreamingQueueError(Exception):
    """Base for admission / wait failures."""


class StreamingQueueRejected(StreamingQueueError):
    def __init__(
        self,
        *,
        error: str,
        retry_after_ms: int,
        queue_depth: int = 0,
        estimated_wait_ms: int = 0,
    ) -> None:
        super().__init__(error)
        self.error = error
        self.retry_after_ms = retry_after_ms
        self.queue_depth = queue_depth
        self.estimated_wait_ms = estimated_wait_ms

    def to_detail(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "retryAfterMs": int(self.retry_after_ms),
            "queueDepth": int(self.queue_depth),
            "estimatedWaitMs": int(self.estimated_wait_ms),
        }


class StreamingQueueTimeout(StreamingQueueError):
    def __init__(self, *, queue_depth: int, waited_ms: int) -> None:
        super().__init__("queue_timeout")
        self.queue_depth = queue_depth
        self.waited_ms = waited_ms

    def to_detail(self) -> dict[str, Any]:
        est = _estimated_job_ms()
        return {
            "error": "queue_timeout",
            "retryAfterMs": est,
            "queueDepth": int(self.queue_depth),
            "estimatedWaitMs": est,
            "waitedMs": int(self.waited_ms),
        }


@dataclass
class StreamingAdmitTicket:
    ticket_id: str
    session_id: str
    admitted_at: float = field(default_factory=time.monotonic)
    queue_position: int = 0


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, str(default)))))
    except ValueError:
        return default


def _max_active_jobs() -> int:
    return _env_int("ARACHNE_STREAM_MAX_ACTIVE_JOBS", 1, 1, 4)


def _max_queue() -> int:
    return _env_int("ARACHNE_STREAM_MAX_QUEUE", 3, 0, 64)


def _queue_timeout_sec() -> float:
    return float(_env_int("ARACHNE_STREAM_QUEUE_TIMEOUT_SEC", 15, 1, 300))


def _estimated_job_ms() -> int:
    return _env_int("ARACHNE_STREAM_ESTIMATED_JOB_MS", 8000, 1000, 120_000)


class StreamingInferenceQueue:
    """Thread-safe admission queue for realtime NDJSON avatar streams."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._lifecycle = WorkerLifecycle.starting
        self._active_ticket: Optional[str] = None
        self._active_session_id: Optional[str] = None
        self._active_since: Optional[float] = None
        self._waiting: Deque[StreamingAdmitTicket] = deque()
        self._tickets: dict[str, StreamingAdmitTicket] = {}
        self._started_at = time.time()
        self._last_job_finished_at: Optional[float] = None
        self._reject_total = 0
        self._timeout_total = 0
        self._completed_total = 0
        self._total_wait_ms = 0.0
        self._wait_samples = 0

    def mark_ready(self) -> None:
        with self._cond:
            if self._lifecycle == WorkerLifecycle.starting:
                self._lifecycle = WorkerLifecycle.active
                self._cond.notify_all()

    def set_lifecycle(self, lifecycle: WorkerLifecycle) -> None:
        with self._cond:
            self._lifecycle = lifecycle
            self._cond.notify_all()

    def lifecycle(self) -> WorkerLifecycle:
        with self._cond:
            return self._lifecycle

    def try_admit(self, session_id: str) -> StreamingAdmitTicket:
        sid = str(session_id or "").strip() or "anonymous"
        with self._cond:
            if self._lifecycle == WorkerLifecycle.offline:
                self._reject_total += 1
                raise StreamingQueueRejected(
                    error="worker_offline",
                    retry_after_ms=_estimated_job_ms(),
                    queue_depth=self.queue_depth_locked(),
                )
            if self._lifecycle == WorkerLifecycle.draining:
                self._reject_total += 1
                raise StreamingQueueRejected(
                    error="worker_draining",
                    retry_after_ms=_estimated_job_ms(),
                    queue_depth=self.queue_depth_locked(),
                )

            depth = self.queue_depth_locked()
            max_total = _max_active_jobs() + _max_queue()
            if depth >= max_total:
                self._reject_total += 1
                est = _estimated_job_ms() * max(1, depth)
                raise StreamingQueueRejected(
                    error="worker_busy",
                    retry_after_ms=_estimated_job_ms(),
                    queue_depth=depth,
                    estimated_wait_ms=est,
                )

            ticket = StreamingAdmitTicket(
                ticket_id=str(uuid.uuid4()),
                session_id=sid,
                queue_position=len(self._waiting) + (1 if self._active_ticket else 0),
            )
            self._tickets[ticket.ticket_id] = ticket
            self._waiting.append(ticket)
            self._cond.notify_all()
            return ticket

    def wait_for_active(self, ticket_id: str) -> None:
        deadline = time.monotonic() + _queue_timeout_sec()
        wait_start = time.monotonic()
        with self._cond:
            ticket = self._tickets.get(ticket_id)
            if ticket is None:
                raise StreamingQueueRejected(
                    error="invalid_ticket",
                    retry_after_ms=_estimated_job_ms(),
                )

            while self._active_ticket != ticket_id:
                if self._lifecycle == WorkerLifecycle.offline:
                    self._cancel_ticket_locked(ticket_id)
                    raise StreamingQueueRejected(
                        error="worker_offline",
                        retry_after_ms=_estimated_job_ms(),
                        queue_depth=self.queue_depth_locked(),
                    )

                if ticket_id not in self._tickets:
                    raise StreamingQueueRejected(
                        error="ticket_cancelled",
                        retry_after_ms=_estimated_job_ms(),
                    )

                self._maybe_promote_locked()

                if self._active_ticket == ticket_id:
                    waited_ms = int((time.monotonic() - wait_start) * 1000)
                    self._total_wait_ms += waited_ms
                    self._wait_samples += 1
                    return

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timeout_total += 1
                    depth = self.queue_depth_locked()
                    waited_ms = int((time.monotonic() - wait_start) * 1000)
                    self._cancel_ticket_locked(ticket_id)
                    raise StreamingQueueTimeout(queue_depth=depth, waited_ms=waited_ms)

                self._cond.wait(timeout=min(remaining, 0.25))

    def release_active(self, ticket_id: str) -> None:
        with self._cond:
            if self._active_ticket != ticket_id:
                self._cancel_ticket_locked(ticket_id)
                return
            self._active_ticket = None
            self._active_session_id = None
            self._active_since = None
            self._last_job_finished_at = time.time()
            self._completed_total += 1
            self._tickets.pop(ticket_id, None)
            self._maybe_promote_locked()
            self._cond.notify_all()

    def cancel_ticket(self, ticket_id: str) -> None:
        with self._cond:
            self._cancel_ticket_locked(ticket_id)
            self._cond.notify_all()

    def _cancel_ticket_locked(self, ticket_id: str) -> None:
        self._tickets.pop(ticket_id, None)
        self._waiting = deque(t for t in self._waiting if t.ticket_id != ticket_id)
        if self._active_ticket == ticket_id:
            self._active_ticket = None
            self._active_session_id = None
            self._active_since = None
            self._maybe_promote_locked()

    def _maybe_promote_locked(self) -> None:
        if self._active_ticket is not None:
            return
        while self._waiting:
            front = self._waiting[0]
            if front.ticket_id not in self._tickets:
                self._waiting.popleft()
                continue
            self._waiting.popleft()
            self._active_ticket = front.ticket_id
            self._active_session_id = front.session_id
            self._active_since = time.monotonic()
            return

    def queue_depth_locked(self) -> int:
        active = 1 if self._active_ticket else 0
        return active + len(self._waiting)

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            depth = self.queue_depth_locked()
            avg_wait = (
                int(self._total_wait_ms / self._wait_samples) if self._wait_samples else 0
            )
            active_age_ms = None
            if self._active_since is not None:
                active_age_ms = int((time.monotonic() - self._active_since) * 1000)
            return {
                "lifecycle": self._lifecycle.value,
                "maxActiveJobs": _max_active_jobs(),
                "maxQueue": _max_queue(),
                "queueTimeoutSec": _queue_timeout_sec(),
                "queueDepth": depth,
                "waitingCount": len(self._waiting),
                "activeStreamingJobs": 1 if self._active_ticket else 0,
                "activeSessionId": self._active_session_id,
                "activeJobAgeMs": active_age_ms,
                "rejectTotal": self._reject_total,
                "timeoutTotal": self._timeout_total,
                "completedTotal": self._completed_total,
                "avgQueueWaitMs": avg_wait,
                "uptimeSec": int(time.time() - self._started_at),
                "lastJobFinishedAt": self._last_job_finished_at,
                "estimatedJobMs": _estimated_job_ms(),
            }


streaming_queue = StreamingInferenceQueue()
