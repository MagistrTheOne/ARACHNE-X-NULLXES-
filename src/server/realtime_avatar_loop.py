"""Single entry: run one conversational turn through LLM → TTS → GPU frame stream."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from src.server.asr_whisper import transcribe_f32_mono_16k
from src.server.avatar_stream_client import stream_avatar_frames
from src.server.llm_runner import generate_reply_sync
from src.server.session_state import RealtimeSessionState
from src.server.tts_runner import synthesize_pcm_f32_16k
from src.server.ws_events import ws_event_base

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)

# Omit optional "engine" JSON field for NULLXES core path (and legacy client aliases).
_CORE_AVATAR_ENGINES = frozenset({"", "arachne", "nullxes", "longcat", "core"})


def _emotion_hint(vec: np.ndarray) -> str:
    if vec.size == 0:
        return ""
    return ",".join(f"{float(v):.3f}" for v in vec[:8])


async def stream_avatar_frames_from_audio(
    *,
    app: "web.Application",
    nullxes_session_id: str,
    state: RealtimeSessionState,
    audio_f32: np.ndarray,
    cancel: asyncio.Event,
    frame_queue: "asyncio.Queue[dict[str, Any]]",
    prompt: str,
    engine: Optional[str] = None,
) -> None:
    """
    Stream JPEG chunks from inference worker for arbitrary float32 mono 16 kHz audio (TTS or mic utterance).
    Emits avatar.state.changed and avatar.stream.chunk messages only (no speaker.changed).
    """
    if audio_f32.size == 0:
        return

    prompt_txt, img_b64, resolution, _ = state.snapshot_avatar()
    use_prompt = (prompt or prompt_txt).strip() or "neutral speech"
    if not img_b64 or not str(img_b64).strip():
        logger.error("No avatar_image_base64 in session state; cannot render")
        await frame_queue.put(
            {
                **ws_event_base(session_id=nullxes_session_id),
                "type": "session.error",
                "message": "avatar_image_missing",
            }
        )
        return

    # Single audio format contract to inference: PCM16 mono 16kHz base64.
    pcm16 = np.clip(np.asarray(audio_f32, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    audio_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
    eng = (engine or "arachne").strip().lower()

    http = app.get("avatar_http_session")
    base_ev = ws_event_base(session_id=nullxes_session_id)
    try:
        await frame_queue.put(
            {
                **base_ev,
                "type": "avatar.state.changed",
                "state": "speaking",
            }
        )
        seq = 0
        async for _seq, frame in stream_avatar_frames(
            http,
            prompt=use_prompt[:8000],
            session_id=nullxes_session_id,
            image_base64=str(img_b64),
            audio_pcm16_base64=audio_b64,
            resolution=str(resolution or "480p"),
            engine=eng if eng not in _CORE_AVATAR_ENGINES else None,
        ):
            if cancel.is_set():
                break
            seq += 1
            await frame_queue.put(
                {
                    **base_ev,
                    "type": "avatar.chunk",
                    "kind": "video",
                    "seq": seq,
                    "encoding": str(frame.get("encoding") or "rgb24_base64"),
                    "data": frame.get("data"),
                    "width": frame.get("width"),
                    "height": frame.get("height"),
                    "tsMs": frame.get("tsMs"),
                }
            )
        await frame_queue.put(
            {
                **base_ev,
                "type": "avatar.state.changed",
                "state": "idle",
            }
        )
    except Exception as e:
        logger.exception("GPU avatar stream failed: %s", e)
        await frame_queue.put(
            {
                **base_ev,
                "type": "session.error",
                "message": "avatar_inference_failed",
                "detail": str(e)[:800],
            }
        )
        await frame_queue.put(
            {
                **base_ev,
                "type": "avatar.state.changed",
                "state": "idle",
            }
        )


async def run_realtime_avatar_turn(
    *,
    app: "web.Application",
    nullxes_session_id: str,
    state: RealtimeSessionState,
    pipeline_cfg: Dict[str, Any],
    user_text: str,
    cancel: asyncio.Event,
    frame_queue: "asyncio.Queue[dict[str, Any]]",
    avatar_inference_engine: Optional[str] = None,
) -> None:
    """
    user_text: transcript or chat input.
    Puts dict messages on frame_queue for WebSocket: {type, ...}.
    """
    llm_cfg = dict(pipeline_cfg.get("llm") or {})
    tts_cfg = dict(pipeline_cfg.get("tts") or {})
    state.append_message("user", user_text)
    emotion_hint = _emotion_hint(state.emotion_vector)
    system = str(
        llm_cfg.get("system_prompt")
        or "You are a helpful assistant. Reply concisely in the same language as the user."
    )

    reply_text = ""
    backoff = 0.5
    max_retries = int(os.environ.get("NULLXES_LLM_MAX_RETRIES", "3"))
    for attempt in range(max_retries):
        if cancel.is_set():
            return
        try:
            reply_text = await asyncio.to_thread(
                generate_reply_sync,
                state.messages_for_llm(),
                llm_cfg,
                system_prompt=system,
                emotion_hint=emotion_hint,
            )
            state.llm_retry_count = 0
            break
        except Exception as e:
            logger.warning("LLM attempt %s failed: %s", attempt + 1, e)
            state.llm_retry_count = attempt + 1
            if attempt + 1 >= max_retries:
                reply_text = str(
                    llm_cfg.get("failure_reply")
                    or "I am having trouble responding right now. Please try again."
                )
            else:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 8.0)

    if cancel.is_set():
        return

    state.append_message("assistant", reply_text)
    base_ev0 = ws_event_base(session_id=nullxes_session_id)
    await frame_queue.put(
        {
            **base_ev0,
            "type": "chat.message.received",
            "message": {
                "id": f"reply_{nullxes_session_id}_{int(time.time() * 1000)}",
                "from": "assistant",
                "text": reply_text,
            },
        }
    )

    try:
        audio_f32 = await asyncio.to_thread(synthesize_pcm_f32_16k, reply_text, tts_cfg)
    except Exception as e:
        logger.warning("TTS failed, text-only stream: %s", e)
        await frame_queue.put(
            {
                **base_ev0,
                "type": "avatar.mode",
                "mode": "text_only",
                "reason": str(e)[:500],
            }
        )
        return

    if audio_f32.size == 0:
        return

    base_ev = ws_event_base(session_id=nullxes_session_id)
    await frame_queue.put(
        {
            **base_ev,
            "type": "speaker.changed",
            "speaker": "assistant",
        }
    )

    prompt_txt, _img, _res, _ = state.snapshot_avatar()
    render_prompt = (prompt_txt or reply_text[:500]).strip() or reply_text[:500]

    try:
        await stream_avatar_frames_from_audio(
            app=app,
            nullxes_session_id=nullxes_session_id,
            state=state,
            audio_f32=audio_f32,
            cancel=cancel,
            frame_queue=frame_queue,
            prompt=render_prompt,
            engine=avatar_inference_engine,
        )
    finally:
        await frame_queue.put(
            {
                **base_ev,
                "type": "speaker.changed",
                "speaker": "none",
            }
        )


async def transcribe_utterance(samples: np.ndarray, asr_cfg: Dict[str, Any]) -> str:
    return await asyncio.to_thread(transcribe_f32_mono_16k, samples, asr_cfg)


realtime_avatar_loop = run_realtime_avatar_turn
