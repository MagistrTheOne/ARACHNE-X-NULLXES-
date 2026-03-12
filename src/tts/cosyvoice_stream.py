import asyncio
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soxr

from src.llm.openai_adapter import SentenceChunk


class CosyVoiceStream:
    """Sentence-chunked streaming adapter around CosyVoice AutoModel."""

    TARGET_SR = 16000

    def __init__(
        self,
        model_dir: str,
        reference_audio: str,
        reference_text: str,
        fp16: bool = False,
        min_chunk_chars: int = 8,
    ):
        from cosyvoice.cli.cosyvoice import AutoModel

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"CosyVoice model dir not found: {model_dir}")
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        if not reference_text.strip():
            raise ValueError("`reference_text` must not be empty.")

        self.model = AutoModel(model_dir=model_dir, fp16=fp16)
        self.reference_audio = reference_audio
        self.reference_text = reference_text.strip()
        self.min_chunk_chars = int(min_chunk_chars)
        self.source_sr = int(getattr(self.model, "sample_rate", self.TARGET_SR))

    async def synthesize_segments(
        self,
        text_stream: AsyncGenerator[SentenceChunk, None],
    ) -> AsyncGenerator[np.ndarray, None]:
        async for chunk in text_stream:
            text = chunk.text.strip()
            if len(text) < self.min_chunk_chars and not chunk.is_final:
                continue
            loop = asyncio.get_running_loop()
            pcm_chunks = await loop.run_in_executor(None, self._synth_sync, text)
            for pcm in pcm_chunks:
                yield pcm

    def _synth_sync(self, text: str) -> list[np.ndarray]:
        chunks: list[np.ndarray] = []
        for payload in self.model.inference_zero_shot(
            text,
            self.reference_text,
            self.reference_audio,
        ):
            speech = payload["tts_speech"]
            pcm = speech.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            if pcm.size:
                if self.source_sr != self.TARGET_SR:
                    pcm = soxr.resample(pcm, self.source_sr, self.TARGET_SR).astype(np.float32, copy=False)
                chunks.append(np.ascontiguousarray(pcm))
        return chunks
