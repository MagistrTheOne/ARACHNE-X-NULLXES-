"""Unit tests for GPU streaming admission queue (no CUDA)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKER_DIR = Path(__file__).resolve().parents[1] / "services" / "arachnex-worker"
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))

from streaming_queue import (  # noqa: E402
    StreamingInferenceQueue,
    StreamingQueueRejected,
    StreamingQueueTimeout,
    WorkerLifecycle,
)


@pytest.fixture(autouse=True)
def _queue_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARACHNE_STREAM_MAX_ACTIVE_JOBS", "1")
    monkeypatch.setenv("ARACHNE_STREAM_MAX_QUEUE", "2")
    monkeypatch.setenv("ARACHNE_STREAM_QUEUE_TIMEOUT_SEC", "1")
    monkeypatch.setenv("ARACHNE_STREAM_ESTIMATED_JOB_MS", "100")


def test_reject_when_queue_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARACHNE_STREAM_MAX_QUEUE", "1")
    q = StreamingInferenceQueue()
    q.mark_ready()
    t1 = q.try_admit("s1")
    t2 = q.try_admit("s2")
    with pytest.raises(StreamingQueueRejected) as exc:
        q.try_admit("s3")
    assert exc.value.error == "worker_busy"
    assert exc.value.retry_after_ms >= 1000
    q.cancel_ticket(t1.ticket_id)
    q.cancel_ticket(t2.ticket_id)


def test_wait_promote_and_release() -> None:
    q = StreamingInferenceQueue()
    q.mark_ready()
    t1 = q.try_admit("session-a")
    q.wait_for_active(t1.ticket_id)
    snap = q.snapshot()
    assert snap["activeStreamingJobs"] == 1
    assert snap["activeSessionId"] == "session-a"
    q.release_active(t1.ticket_id)
    assert q.snapshot()["activeStreamingJobs"] == 0


def test_queue_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARACHNE_STREAM_QUEUE_TIMEOUT_SEC", "1")
    q = StreamingInferenceQueue()
    q.mark_ready()
    t1 = q.try_admit("hold")
    q.wait_for_active(t1.ticket_id)

    t2 = q.try_admit("waiter")
    with pytest.raises(StreamingQueueTimeout):
        q.wait_for_active(t2.ticket_id)

    q.release_active(t1.ticket_id)


def test_draining_rejects_new_admits() -> None:
    q = StreamingInferenceQueue()
    q.mark_ready()
    q.set_lifecycle(WorkerLifecycle.draining)
    with pytest.raises(StreamingQueueRejected) as exc:
        q.try_admit("x")
    assert exc.value.error == "worker_draining"
