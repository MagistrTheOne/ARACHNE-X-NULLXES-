import asyncio
from pathlib import Path
from typing import AsyncGenerator

import numpy as np

from src.llm.openai_adapter import SentenceChunk


class CosyVoiceStream:
    """Sentence-chunked streaming adapter around CosyVoice2."""

    TARGET_SR = 16000

    def __init__(
        self,
        model_dir: str,
        reference_audio: str,
        load_jit: bool = True,
        load_trt: bool = False,
        min_chunk_chars: int = 8,
    ):
        from cosyvoice.cli.cosyvoice import CosyVoice2

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"CosyVoice model dir not found: {model_dir}")
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        self.model = CosyVoice2(model_dir, load_jit=load_jit, load_trt=load_trt)
        self.reference_audio = reference_audio
        self.min_chunk_chars = int(min_chunk_chars)

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
        for payload in self.model.inference_zero_shot(text, self.reference_audio, stream=True):
            speech = payload["tts_speech"]
            pcm = speech.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            if pcm.size:
                chunks.append(np.ascontiguousarray(pcm))
        return chunks
