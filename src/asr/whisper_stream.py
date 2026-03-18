import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ASRResult:
    text: str
    language: Optional[str]
    duration_ms: float
    latency_ms: float


class WhisperStreamASR:
    """Utterance-final ASR adapter over faster-whisper."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
        num_workers: int = 2,
    ):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )
        self.beam_size = int(beam_size)

    async def transcribe_utterance(self, audio: np.ndarray, language: Optional[str] = "ru") -> ASRResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ASRResult(text="", language=language, duration_ms=0.0, latency_ms=0.0)

        started = time.perf_counter()

        loop = asyncio.get_running_loop()

        def _transcribe_sync():
            return self.model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=False,
                word_timestamps=False,
            )

        segments, info = await loop.run_in_executor(None, _transcribe_sync)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        latency_ms = (time.perf_counter() - started) * 1000.0
        duration_ms = audio.size * 1000.0 / 16000.0
        detected_language = getattr(info, "language", language)
        return ASRResult(text=text, language=detected_language, duration_ms=duration_ms, latency_ms=latency_ms)
