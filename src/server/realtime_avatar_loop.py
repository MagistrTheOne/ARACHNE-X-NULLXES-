"""Single entry: run one conversational turn through LLM → TTS → GPU frame stream."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from src.server.asr_whisper import transcribe_f32_mono_16k
from src.server.avatar_stream_client import stream_avatar_jpeg_frames
from src.server.llm_runner import generate_reply_sync
from src.server.session_state import RealtimeSessionState
from src.server.tts_runner import synthesize_pcm_f32_16k

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)


def _emotion_hint(vec: np.ndarray) -> str:
    if vec.size == 0:
        return ""
    return ",".join(f"{float(v):.3f}" for v in vec[:8])


async def run_realtime_avatar_turn(
    *,
    app: "web.Application",
    nullxes_session_id: str,
    state: RealtimeSessionState,
    pipeline_cfg: Dict[str, Any],
    user_text: str,
    cancel: asyncio.Event,
    frame_queue: "asyncio.Queue[dict[str, Any]]",
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
    await frame_queue.put(
        {
            "type": "chat.message.received",
            "at": int(time.time() * 1000),
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
                "type": "avatar.mode",
                "at": int(time.time() * 1000),
                "mode": "text_only",
                "reason": str(e)[:500],
            }
        )
        return

    if audio_f32.size == 0:
        return

    prompt, img_b64, resolution, _identity_id = state.snapshot_avatar()
    if not img_b64 or not str(img_b64).strip():
        logger.error("No avatar_image_base64 in session state; cannot render")
        await frame_queue.put(
            {
                "type": "session.error",
                "at": int(time.time() * 1000),
                "message": "avatar_image_missing",
            }
        )
        return

    raw_bytes = np.asarray(audio_f32, dtype=np.float32).tobytes()
    audio_b64 = base64.b64encode(raw_bytes).decode("ascii")

    http = app.get("avatar_http_session")
    try:
        await frame_queue.put(
            {
                "type": "avatar.state.changed",
                "at": int(time.time() * 1000),
                "state": "speaking",
            }
        )
        seq = 0
        async for _seq, jpeg_b64 in stream_avatar_jpeg_frames(
            http,
            prompt=prompt or reply_text[:500],
            session_id=nullxes_session_id,
            image_base64=str(img_b64),
            audio_float32_base64=audio_b64,
            resolution=str(resolution or "480p"),
        ):
            if cancel.is_set():
                break
            seq += 1
            await frame_queue.put(
                {
                    "type": "avatar.stream.chunk",
                    "at": int(time.time() * 1000),
                    "kind": "video",
                    "seq": seq,
                    "encoding": "jpeg_base64",
                    "data": jpeg_b64,
                }
            )
        await frame_queue.put(
            {
                "type": "avatar.state.changed",
                "at": int(time.time() * 1000),
                "state": "idle",
            }
        )
    except Exception as e:
        logger.exception("GPU avatar stream failed: %s", e)
        await frame_queue.put(
            {
                "type": "session.error",
                "at": int(time.time() * 1000),
                "message": "avatar_inference_failed",
                "detail": str(e)[:800],
            }
        )
        await frame_queue.put(
            {
                "type": "avatar.state.changed",
                "at": int(time.time() * 1000),
                "state": "idle",
            }
        )


async def transcribe_utterance(samples: np.ndarray, asr_cfg: Dict[str, Any]) -> str:
    return await asyncio.to_thread(transcribe_f32_mono_16k, samples, asr_cfg)


realtime_avatar_loop = run_realtime_avatar_turn
