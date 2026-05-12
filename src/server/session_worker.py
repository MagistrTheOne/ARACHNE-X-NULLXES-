"""Per-session realtime pipeline: PCM → VAD → ASR → LLM → TTS → GPU JPEG stream."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from src.server.realtime_avatar_loop import (
    run_realtime_avatar_turn,
    stream_avatar_frames_from_audio,
    transcribe_utterance,
)
from src.server.session_manager import SessionManager, SessionRecord, SessionState
from src.server.session_state import RealtimeSessionState
from src.server.vad_rms import RMSUtteranceDetector, RMSVADConfig
from src.server.ws_events import ws_event_base

if TYPE_CHECKING:
    from aiohttp.web import Application

logger = logging.getLogger(__name__)

WorkItem = Union[tuple[str, str], tuple[str, np.ndarray]]


def _pcm16le_to_f32(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros((0,), dtype=np.float32)
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.clip(x / 32768.0, -1.0, 1.0)


class SessionWorker:
    def __init__(self, app: "Application", record: SessionRecord) -> None:
        self._app = app
        self._rec = record
        self._cancel = asyncio.Event()
        self._turn_cancel: Optional[asyncio.Event] = None
        self._turn_task: Optional[asyncio.Task[None]] = None
        self._main_task: Optional[asyncio.Task[None]] = None
        self._pcm_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        self._work_queue: asyncio.Queue[WorkItem] = asyncio.Queue(maxsize=64)
        self._pipeline_cfg: Dict[str, Any] = dict(app.get("pipeline_cfg") or {})
        self._state = RealtimeSessionState()
        self._vad_cfg = RMSVADConfig(
            sample_rate=int(self._pipeline_cfg.get("vad", {}).get("sample_rate_hz", 16000)),
        )
        self._vad = RMSUtteranceDetector(self._vad_cfg)
        self._process_lock = asyncio.Lock()
        max_frames = int(os.environ.get("NULLXES_AVATAR_FRAME_QUEUE_MAX", "48"))
        self.out_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max_frames)
        self._duplex_mode: bool = False
        self._video_audio_source: str = "tts"
        self._avatar_inference_engine: str = "arachne"
        self._mic_asr_transcript_only: bool = True
        self._meeting_id: Optional[str] = None
        self._apply_session_config(record)

    def _apply_session_config(self, record: SessionRecord) -> None:
        cfg = dict(record.config or {})
        cfg.update(dict(record.media_binding or {}))
        self._state.avatar_prompt = str(cfg.get("avatar_prompt") or cfg.get("avatarPrompt") or "")
        img = cfg.get("avatar_image_base64") or cfg.get("avatarImageBase64")
        if img:
            self._state.avatar_image_base64 = str(img)
        res = cfg.get("resolution") or cfg.get("avatar_resolution")
        if res:
            self._state.resolution = str(res)
        iid = cfg.get("identity_id") or cfg.get("identityId")
        if iid is not None:
            try:
                self._state.identity_id = int(iid)
            except (TypeError, ValueError):
                self._state.identity_id = None

        dm = cfg.get("duplex_mode")
        if dm is None:
            dm = cfg.get("duplexMode")
        self._duplex_mode = bool(dm) if dm is not None else False

        vas = cfg.get("video_audio_source") or cfg.get("videoAudioSource") or "tts"
        self._video_audio_source = str(vas).lower()

        eng = cfg.get("avatar_inference_engine") or cfg.get("avatarInferenceEngine") or "arachne"
        self._avatar_inference_engine = str(eng).lower()

        mat = cfg.get("mic_asr_transcript_only")
        if mat is None:
            mat = cfg.get("micAsrTranscriptOnly")
        self._mic_asr_transcript_only = bool(mat) if mat is not None else True

        mid = cfg.get("meeting_id") or cfg.get("meetingId")
        if mid:
            self._meeting_id = str(mid).strip()
        elif self._rec.external_session_id:
            self._meeting_id = str(self._rec.external_session_id).strip()
        else:
            self._meeting_id = None

    @property
    def running(self) -> bool:
        return self._main_task is not None and not self._main_task.done()

    @property
    def nullxes_session_id(self) -> str:
        return self._rec.nullxes_session_id

    def _register_out_queue(self) -> None:
        d = self._app.setdefault("avatar_frame_queues", {})
        d[self._rec.nullxes_session_id] = self.out_queue

    def _unregister_out_queue(self) -> None:
        d = self._app.get("avatar_frame_queues")
        if isinstance(d, dict):
            d.pop(self._rec.nullxes_session_id, None)

    async def start(self) -> None:
        if self.running:
            return
        self._cancel.clear()
        self._register_out_queue()
        self._main_task = asyncio.create_task(
            self._run(), name=f"worker-{self._rec.nullxes_session_id}"
        )

    def request_stop(self) -> None:
        self._cancel.set()
        if self._turn_cancel:
            self._turn_cancel.set()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        if self._mic_cancel:
            self._mic_cancel.set()
        if self._mic_task and not self._mic_task.done():
            self._mic_task.cancel()

    async def wait_done(self, timeout: float = 30.0) -> None:
        if self._main_task:
            try:
                await asyncio.wait_for(self._main_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("worker %s stop timeout", self._rec.nullxes_session_id)
                self._main_task.cancel()
            except asyncio.CancelledError:
                pass

    async def feed_pcm16(self, pcm16le: bytes) -> None:
        if self._cancel.is_set():
            return
        try:
            self._pcm_queue.put_nowait(pcm16le)
        except asyncio.QueueFull:
            try:
                _ = self._pcm_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._pcm_queue.put_nowait(pcm16le)
            except asyncio.QueueFull:
                logger.warning("pcm queue saturated for %s", self._rec.nullxes_session_id)

    async def enqueue_user_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        await self._work_queue.put(("text", text))

    async def _cancel_mic_render(self) -> None:
        if self._mic_cancel:
            self._mic_cancel.set()
        if self._mic_task and not self._mic_task.done():
            self._mic_task.cancel()
            try:
                await self._mic_task
            except asyncio.CancelledError:
                pass

    async def _cancel_assistant_turn(self) -> None:
        if self._turn_cancel:
            self._turn_cancel.set()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass

    async def _emit_mic_transcript(self, utt: np.ndarray) -> None:
        asr_cfg = dict(self._pipeline_cfg.get("asr") or {})
        try:
            text = await transcribe_utterance(np.asarray(utt, dtype=np.float32).copy(), asr_cfg)
            if not text.strip():
                return
            await self.out_queue.put(
                {
                    **ws_event_base(session_id=self._rec.nullxes_session_id, meeting_id=self._meeting_id),
                    "type": "voice.transcript",
                    "text": text.strip(),
                    "source": "mic",
                }
            )
        except Exception as e:
            logger.debug("mic transcript ASR skipped: %s", e)

    async def _run(self) -> None:
        sm: SessionManager = self._app["session_manager"]
        pcm_task = asyncio.create_task(self._pcm_vad_loop(), name=f"pcm-vad-{self._rec.nullxes_session_id}")
        try:
            while not self._cancel.is_set():
                rec = sm.get(self._rec.nullxes_session_id)
                if not rec or rec.state in (SessionState.STOPPED, SessionState.FAILED):
                    break
                try:
                    kind, payload = await asyncio.wait_for(
                        self._work_queue.get(), timeout=0.25
                    )
                except asyncio.TimeoutError:
                    continue
                if kind == "text":
                    await self._start_turn(str(payload), from_asr=False)
                elif kind == "voice":
                    await self._handle_voice_utterance(np.asarray(payload, dtype=np.float32))
                rec = sm.get(self._rec.nullxes_session_id) or rec
                sm.touch_record(rec)
        except asyncio.CancelledError:
            logger.info("worker cancelled %s", self._rec.nullxes_session_id)
            raise
        except Exception as e:
            logger.exception("worker failed %s: %s", self._rec.nullxes_session_id, e)
            sm.mark_failed(self._rec.nullxes_session_id, str(e))
        finally:
            pcm_task.cancel()
            try:
                await pcm_task
            except asyncio.CancelledError:
                pass
            self._unregister_out_queue()
            rec = sm.get(self._rec.nullxes_session_id)
            if rec and rec.state in (SessionState.DRAINING, SessionState.RUNNING):
                sm.finalize_stop(self._rec.nullxes_session_id)

    async def _pcm_vad_loop(self) -> None:
        while not self._cancel.is_set():
            try:
                chunk = await asyncio.wait_for(self._pcm_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            f32 = _pcm16le_to_f32(chunk)
            if f32.size:
                rms = float(np.sqrt(np.mean(np.square(f32), dtype=np.float64)))
                self._state.update_emotion_from_rms(rms)
            utt = self._vad.push_pcm_f32_mono(f32)
            if utt is not None and utt.size > 0:
                self._state.push_audio_context(utt)
                try:
                    self._work_queue.put_nowait(("voice", utt))
                except asyncio.QueueFull:
                    logger.warning("work queue full; dropping utterance")

    async def _handle_voice_utterance(self, utt: np.ndarray) -> None:
        if self._duplex_mode and (self._video_audio_source in ("mic", "auto")):
            await self._cancel_assistant_turn()
            await self._cancel_mic_render()

            if self._mic_asr_transcript_only:
                asyncio.create_task(
                    self._emit_mic_transcript(np.asarray(utt, dtype=np.float32).copy()),
                    name=f"mic-asr-{self._rec.nullxes_session_id}",
                )

            sm: SessionManager = self._app["session_manager"]
            sid = self._rec.nullxes_session_id

            self._mic_cancel = asyncio.Event()
            mic_cancel = self._mic_cancel

            async def _mic_render() -> None:
                try:
                    rec_now = sm.get(sid)
                    if rec_now:
                        rec_now.health["stt"] = "duplex_mic"
                        sm.touch_record(rec_now)
                    await self.out_queue.put(
                        {
                            **ws_event_base(session_id=sid, meeting_id=self._meeting_id),
                            "type": "speaker.changed",
                            "speaker": "candidate",
                        }
                    )
                    await stream_avatar_frames_from_audio(
                        app=self._app,
                        nullxes_session_id=sid,
                        state=self._state,
                        audio_f32=np.asarray(utt, dtype=np.float32),
                        cancel=mic_cancel,
                        frame_queue=self.out_queue,
                        prompt=str(self._state.avatar_prompt or ""),
                        engine=self._avatar_inference_engine,
                    )
                finally:
                    try:
                        await self.out_queue.put(
                            {
                                **ws_event_base(session_id=sid, meeting_id=self._meeting_id),
                                "type": "speaker.changed",
                                "speaker": "none",
                            }
                        )
                    except asyncio.QueueFull:
                        pass
                    rec_done = sm.get(sid)
                    if rec_done:
                        rec_done.health["avatar"] = "ok"
                        sm.touch_record(rec_done)

            self._mic_task = asyncio.create_task(_mic_render(), name=f"mic-{sid}")
            return

        asr_cfg = dict(self._pipeline_cfg.get("asr") or {})
        text = ""
        try:
            text = await transcribe_utterance(utt, asr_cfg)
        except Exception as e:
            logger.warning("ASR failed: %s", e)
            self._state.last_error_stage = "asr"
            self._state.text_only_mode = True
            try:
                self.out_queue.put_nowait(
                    {
                        **ws_event_base(session_id=self._rec.nullxes_session_id, meeting_id=self._meeting_id),
                        "type": "session.notice",
                        "message": "asr_unavailable_use_chat_text",
                    }
                )
            except asyncio.QueueFull:
                pass
            return
        if not text.strip():
            return
        await self._start_turn(text.strip(), from_asr=True)

    async def _start_turn(self, user_text: str, from_asr: bool) -> None:
        await self._cancel_mic_render()
        if self._turn_task and not self._turn_task.done():
            if self._turn_cancel:
                self._turn_cancel.set()
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass
        self._turn_cancel = asyncio.Event()
        cancel = self._turn_cancel
        sm: SessionManager = self._app["session_manager"]
        sid = self._rec.nullxes_session_id

        async def _one() -> None:
            try:
                rec_now = sm.get(sid)
                if rec_now:
                    rec_now.health["stt"] = "ok" if from_asr else "skipped"
                    rec_now.health["llm"] = "running"
                    sm.touch_record(rec_now)
                await run_realtime_avatar_turn(
                    app=self._app,
                    nullxes_session_id=sid,
                    state=self._state,
                    pipeline_cfg=self._pipeline_cfg,
                    user_text=user_text,
                    cancel=cancel,
                    frame_queue=self.out_queue,
                    avatar_inference_engine=self._avatar_inference_engine,
                )
            finally:
                rec_done = sm.get(sid)
                if rec_done:
                    rec_done.health["llm"] = "ok"
                    rec_done.health["tts"] = "ok"
                    rec_done.health["avatar"] = "ok"
                    sm.touch_record(rec_done)

        self._turn_task = asyncio.create_task(_one(), name=f"turn-{self._rec.nullxes_session_id}")

    def refresh_config(self, record: SessionRecord) -> None:
        self._apply_session_config(record)
