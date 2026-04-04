"""Thread-safe runtime state for realtime avatar sessions."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class RealtimeSessionState:
    """
    Mutable state shared across VAD/ASR/LLM/TTS/GPU stages for one nullxes_session_id.
    """

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    conversation: List[dict[str, Any]] = field(default_factory=list)
    """OpenAI-style messages: {role, content, ts}."""

    emotion_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )
    """Running summary (8 dims): energy, valence proxy, speech rate, etc."""

    speaker_identity_embedding: Optional[np.ndarray] = None
    """Optional float32 embedding for voice/avatar continuity (filled when ASR provides diarization)."""

    audio_context_cache: List[np.ndarray] = field(default_factory=list)
    """Recent float32 mono 16 kHz snippets (post-VAD) for LLM context; capped."""

    avatar_prompt: str = ""
    avatar_image_base64: Optional[str] = None
    resolution: str = "480p"
    identity_id: Optional[int] = None

    llm_retry_count: int = 0
    last_error_stage: Optional[str] = None
    text_only_mode: bool = False
    """Set when ASR fails: user must use chat.send text only."""

    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        with self._lock:
            self.updated_at = time.time()

    def append_message(self, role: str, content: str) -> None:
        with self._lock:
            self.conversation.append(
                {"role": role, "content": content, "ts": time.time()}
            )
            if len(self.conversation) > 48:
                self.conversation = self.conversation[-48:]
            self.touch()

    def messages_for_llm(self) -> List[dict[str, str]]:
        with self._lock:
            return [{"role": m["role"], "content": str(m["content"])} for m in self.conversation]

    def update_emotion_from_rms(self, rms: float) -> None:
        with self._lock:
            e = self.emotion_vector
            e[0] = float(0.85 * e[0] + 0.15 * min(1.0, rms * 10.0))
            self.emotion_vector = e
            self.touch()

    def push_audio_context(self, pcm_f32: np.ndarray, max_chunks: int = 8) -> None:
        with self._lock:
            if pcm_f32.size == 0:
                return
            self.audio_context_cache.append(np.asarray(pcm_f32, dtype=np.float32).copy())
            if len(self.audio_context_cache) > max_chunks:
                self.audio_context_cache = self.audio_context_cache[-max_chunks:]
            self.touch()

    def set_identity_embedding(self, emb: Optional[np.ndarray]) -> None:
        with self._lock:
            if emb is None:
                self.speaker_identity_embedding = None
            else:
                self.speaker_identity_embedding = np.asarray(emb, dtype=np.float32).copy()
            self.touch()

    def snapshot_avatar(self) -> tuple[str, Optional[str], str, Optional[int]]:
        with self._lock:
            return (
                str(self.avatar_prompt),
                self.avatar_image_base64,
                str(self.resolution),
                self.identity_id,
            )
