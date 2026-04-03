"""Per-session async pipeline: STT → LLM → TTS chunks → avatar (stub) with degraded path."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.server.session_manager import SessionManager, SessionRecord, SessionState

if TYPE_CHECKING:
    from aiohttp.web import Application

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").lower().strip()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return default


class SessionWorker:
    """
    MVP: simulates streaming stages with asyncio queues and sleeps.
    Replace stubs with real VAD/STT/LLM/TTS/ARACHNE bindings behind the same queues.
    """

    def __init__(self, app: "Application", record: SessionRecord) -> None:
        self._app = app
        self._rec = record
        self._task: Optional[asyncio.Task[None]] = None
        self._cancel = asyncio.Event()
        self._stt_queue: asyncio.Queue[str] = asyncio.Queue()
        self._llm_token_queue: asyncio.Queue[str] = asyncio.Queue()
        self._tts_chunk_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._pipeline_cfg: Dict[str, Any] = dict(app.get("pipeline_cfg") or {})

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._cancel.clear()
        self._task = asyncio.create_task(self._run(), name=f"worker-{self._rec.nullxes_session_id}")

    def request_stop(self) -> None:
        self._cancel.set()

    async def wait_done(self, timeout: float = 30.0) -> None:
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("worker %s stop timeout", self._rec.nullxes_session_id)
                self._task.cancel()

    async def _run(self) -> None:
        sm: SessionManager = self._app["session_manager"]
        try:
            while not self._cancel.is_set():
                rec = sm.get(self._rec.nullxes_session_id)
                if not rec or rec.state in (SessionState.STOPPED, SessionState.FAILED):
                    break
                await self._one_streaming_cycle(sm, rec)
                await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            logger.info("worker cancelled %s", self._rec.nullxes_session_id)
            raise
        except Exception as e:
            logger.exception("worker failed %s: %s", self._rec.nullxes_session_id, e)
            sm.mark_failed(self._rec.nullxes_session_id, str(e))
            return
        finally:
            rec = sm.get(self._rec.nullxes_session_id)
            if rec and rec.state in (SessionState.DRAINING, SessionState.RUNNING):
                sm.finalize_stop(self._rec.nullxes_session_id)

    async def _one_streaming_cycle(self, sm: SessionManager, rec: SessionRecord) -> None:
        # --- Simulated partial STT ---
        await asyncio.sleep(0.02)
        await self._stt_queue.put("[stub partial stt]")
        rec.health["stt"] = "streaming"
        sm.touch_record(rec)

        # --- LLM token stream ---
        phrase = "Здравствуйте, я готов продолжить интервью."
        for w in phrase.split():
            if self._cancel.is_set():
                return
            await self._llm_token_queue.put(w + " ")
            await asyncio.sleep(0.02)
        rec.health["llm"] = "streaming"
        sm.touch_record(rec)

        # --- TTS chunks → optional avatar micro-turn ---
        tts_chunks: List[bytes] = [b"\x00\x00\x00\x40" * 256, b"\x00\x00\x00\x80" * 256]
        for chunk in tts_chunks:
            if self._cancel.is_set():
                return
            await self._tts_chunk_queue.put(chunk)
            rec.health["tts"] = "streaming"

            if rec.degraded:
                await asyncio.sleep(0.01)
            else:
                try:
                    await self._avatar_micro_turn(chunk)
                except Exception as e:
                    logger.warning("avatar micro-turn failed, degrading: %s", e)
                    sm.set_degraded(rec.nullxes_session_id, f"avatar:{e}")

            rec = sm.get(self._rec.nullxes_session_id) or rec

        if _env_flag("NULLXES_SIMULATE_MEDIA_ERRORS"):
            rec.media_errors += 1
            sm.touch_record(rec)
            if rec.media_errors >= 3 and not rec.degraded:
                sm.set_degraded(rec.nullxes_session_id, "media_channel:threshold")

        rec.health["stt"] = "ok"
        rec.health["llm"] = "ok"
        rec.health["tts"] = "ok"
        rec.health["avatar"] = "degraded" if rec.degraded else "ok"
        sm.touch_record(rec)

    async def _avatar_micro_turn(self, _pcm_chunk: bytes) -> None:
        """Placeholder for ``generate_streaming_ai2v`` step; keep latency model."""
        await asyncio.sleep(0.04)
        av = self._pipeline_cfg.get("avatar") or {}
        if av.get("fail_once"):
            self._pipeline_cfg.setdefault("avatar", {})["fail_once"] = False
            raise RuntimeError("simulated_avatar_failure")
