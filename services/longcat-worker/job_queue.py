"""
Single-GPU serial job queue (one H200 pod): asyncio queue + background worker.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

JobExecutor = Callable[[dict[str, Any]], bytes]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    body_dict: dict[str, Any]
    error: Optional[str] = None
    result_path: Optional[str] = None
    created: float = field(default_factory=time.time)


def _max_queue() -> int:
    try:
        return max(1, min(256, int(os.environ.get("INFERENCE_MAX_QUEUE", "32"))))
    except ValueError:
        return 32


class InferenceJobQueue:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._executor: Optional[JobExecutor] = None
        self._worker_task: Optional[asyncio.Task[None]] = None

    def set_executor(self, fn: JobExecutor) -> None:
        self._executor = fn

    def start_worker(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._run_worker(), name="inference-job-worker")

    async def stop_worker(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def _run_worker(self) -> None:
        import tempfile

        while True:
            job_id = await self._queue.get()
            rec = self._jobs.get(job_id)
            if rec is None or self._executor is None:
                self._queue.task_done()
                continue
            rec.status = JobStatus.running
            try:
                data = await asyncio.to_thread(self._executor, rec.body_dict)
                fd, path = tempfile.mkstemp(suffix=".mp4", prefix="nx_job_")
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                rec.result_path = path
                rec.status = JobStatus.done
            except Exception as e:
                rec.status = JobStatus.failed
                rec.error = str(e)[:8000]
            finally:
                self._queue.task_done()

    def enqueue(self, body_dict: dict[str, Any]) -> str:
        if self._queue.qsize() >= _max_queue():
            raise RuntimeError("queue_full")
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = JobRecord(
            job_id=job_id,
            status=JobStatus.queued,
            body_dict=body_dict,
        )
        self._queue.put_nowait(job_id)
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def pop_result_path(self, job_id: str) -> Optional[str]:
        rec = self._jobs.get(job_id)
        if rec is None or rec.status != JobStatus.done or not rec.result_path:
            return None
        path = rec.result_path
        rec.result_path = None
        return path

    def delete_job(self, job_id: str) -> None:
        rec = self._jobs.pop(job_id, None)
        if rec and rec.result_path and os.path.isfile(rec.result_path):
            try:
                os.unlink(rec.result_path)
            except OSError:
                pass


job_queue = InferenceJobQueue()
