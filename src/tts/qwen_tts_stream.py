import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soxr

from src.llm.openai_adapter import SentenceChunk


class QwenTTSStream:
    """Streaming adapter around Qwen3-TTS models."""

    TARGET_SR = 16000

    def __init__(
        self,
        model_name_or_path: str,
        mode: str = "custom_voice",
        language: str = "Russian",
        speaker: Optional[str] = None,
        instruct: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        min_chunk_chars: int = 8,
    ):
        from qwen_tts import Qwen3TTSModel
        import torch

        model_path = Path(model_name_or_path)
        if model_path.exists():
            resolved_model = str(model_path)
        else:
            resolved_model = model_name_or_path

        dtype_value = getattr(torch, dtype)
        self.model = Qwen3TTSModel.from_pretrained(
            resolved_model,
            device_map=device,
            dtype=dtype_value,
            attn_implementation=attn_implementation,
        )
        self.mode = mode
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.x_vector_only_mode = bool(x_vector_only_mode)
        self.min_chunk_chars = int(min_chunk_chars)

        if self.mode == "custom_voice" and not self.speaker:
            raise ValueError("`speaker` is required for Qwen custom_voice mode.")
        if self.mode == "voice_clone":
            if not self.ref_audio:
                raise ValueError("`ref_audio` is required for Qwen voice_clone mode.")
            if not self.x_vector_only_mode and not (self.ref_text and self.ref_text.strip()):
                raise ValueError("`ref_text` is required for Qwen voice_clone mode unless x_vector_only_mode=True.")
            if Path(self.ref_audio).exists() is False and not str(self.ref_audio).startswith(("http://", "https://")):
                raise FileNotFoundError(f"Reference audio not found: {self.ref_audio}")

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
        if self.mode == "custom_voice":
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=self.language,
                speaker=self.speaker,
                instruct=self.instruct or None,
            )
        elif self.mode == "voice_design":
            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=self.language,
                instruct=self.instruct,
            )
        elif self.mode == "voice_clone":
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                x_vector_only_mode=self.x_vector_only_mode,
            )
        else:
            raise ValueError(f"Unsupported Qwen TTS mode: {self.mode}")

        chunks: list[np.ndarray] = []
        for wav in wavs:
            pcm = np.asarray(wav, dtype=np.float32).reshape(-1)
            if not pcm.size:
                continue
            if int(sr) != self.TARGET_SR:
                pcm = soxr.resample(pcm, int(sr), self.TARGET_SR).astype(np.float32, copy=False)
            chunks.append(np.ascontiguousarray(pcm))
        return chunks
