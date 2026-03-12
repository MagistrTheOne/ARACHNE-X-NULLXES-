from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

from openai import AsyncOpenAI


@dataclass(frozen=True)
class SentenceChunk:
    text: str
    is_final: bool


class StreamingLLM:
    """Conversational streaming wrapper over OpenAI Responses API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 300,
        min_chunk_chars: int = 24,
    ):
        self.client = AsyncOpenAI(base_url=base_url)
        self.model = model
        self.max_tokens = int(max_tokens)
        self.min_chunk_chars = int(min_chunk_chars)
        self._history: List[dict] = []
        self._cancelled = False
        self._system_prompt = system_prompt
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

    def cancel(self) -> None:
        self._cancelled = True

    def reset_history(self) -> None:
        self._history = []
        if self._system_prompt:
            self._history.append({"role": "system", "content": self._system_prompt})

    async def stream_response(self, user_text: str) -> AsyncGenerator[str, None]:
        self._cancelled = False
        self._history.append({"role": "user", "content": user_text})
        response_text: list[str] = []

        stream = await self.client.responses.create(
            model=self.model,
            input=self._history,
            stream=True,
            max_output_tokens=self.max_tokens,
        )

        async for event in stream:
            if self._cancelled:
                break
            if event.type == "response.output_text.delta":
                delta = event.delta or ""
                if delta:
                    response_text.append(delta)
                    yield delta

        full_text = "".join(response_text).strip()
        if full_text and not self._cancelled:
            self._history.append({"role": "assistant", "content": full_text})

    async def stream_sentence_chunks(self, user_text: str) -> AsyncGenerator[SentenceChunk, None]:
        buffer = ""
        punct = {".", "!", "?", ";", ":", ",", "。", "！", "？"}
        async for delta in self.stream_response(user_text):
            buffer += delta
            tail = buffer[-4:]
            if len(buffer.strip()) < self.min_chunk_chars:
                continue
            if any(ch in punct for ch in tail) or buffer.endswith("\n"):
                chunk = buffer.strip()
                if chunk:
                    yield SentenceChunk(text=chunk, is_final=False)
                    buffer = ""
        if buffer.strip():
            yield SentenceChunk(text=buffer.strip(), is_final=True)
