import asyncio
import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from src.asr.whisper_stream import WhisperStreamASR
from src.avatar.arachne_adapter import ArachneSession
from src.llm.openai_adapter import StreamingLLM
from src.pipeline.audio_buffer import AudioRingBuffer
from src.pipeline.vad import SileroVAD, SpeechState, VADResult
from src.tts.cosyvoice_stream import CosyVoiceStream


class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class PipelineEvent:
    state: SessionState
    transcript: Optional[str] = None


class RealtimePipeline:
    """Single production realtime pipeline for talking avatars."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_buffer = AudioRingBuffer(
            max_seconds=float(config.get("audio_buffer_seconds", 30.0)),
            sample_rate=int(config.get("sample_rate", 16000)),
        )
        self.vad = SileroVAD(**config["vad"])
        self.asr = WhisperStreamASR(**config["asr"])
        self.llm = StreamingLLM(**config["llm"])
        self.tts = CosyVoiceStream(**config["tts"])
        self.session = ArachneSession(**config["avatar"])
        self.session_state = SessionState.IDLE
        self._utterance_lock = asyncio.Lock()
        self._processing_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self.session.start()

    async def stop(self) -> None:
        if self._processing_task is not None:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task
        await self.session.stop()

    async def on_audio_chunk(self, pcm_16k: bytes | np.ndarray, timestamp: Optional[float] = None) -> Optional[PipelineEvent]:
        ts = float(timestamp) if timestamp is not None else asyncio.get_running_loop().time()
        pcm = self._normalize_pcm(pcm_16k)
        await self.audio_buffer.push(pcm, ts)
        self.session_state = SessionState.LISTENING

        vad_result = self.vad.process_chunk(pcm, ts)
        if vad_result.state != SpeechState.ENDED or vad_result.utterance_audio is None:
            return PipelineEvent(state=self.session_state)

        if self._processing_task is not None and not self._processing_task.done():
            self.llm.cancel()
            await self.session.interrupt()
            self.session_state = SessionState.INTERRUPTED

        self._processing_task = asyncio.create_task(self._handle_utterance(vad_result))
        return PipelineEvent(state=SessionState.THINKING)

    async def _handle_utterance(self, vad_result: VADResult) -> None:
        async with self._utterance_lock:
            self.session_state = SessionState.THINKING
            asr_result = await self.asr.transcribe_utterance(
                vad_result.utterance_audio,
                language=self.config.get("language", "ru"),
            )
            if not asr_result.text.strip():
                self.session_state = SessionState.IDLE
                return

            text_stream = self.llm.stream_sentence_chunks(asr_result.text)
            self.session_state = SessionState.SPEAKING
            async for pcm_chunk in self.tts.synthesize_segments(text_stream):
                await self.session.push_audio_chunk(pcm_chunk)
            self.session_state = SessionState.IDLE

    @staticmethod
    def _normalize_pcm(pcm_16k: bytes | np.ndarray) -> np.ndarray:
        if isinstance(pcm_16k, bytes):
            pcm = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(pcm)
