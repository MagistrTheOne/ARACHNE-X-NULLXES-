from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CompilerBackend = Literal["off"]
AvatarInferMode = Literal[
    "ai2v",
    "at2v",
    "streaming_ai2v",
    "t2v",
    "i2v",
    "vc",
    "avc",
    "audio_i2v",
    "imagine_i2v",
]


@dataclass(frozen=True)
class AvatarTurnPlan:
    positive_prompt: str
    negative_prompt: str
    compiler_backend: CompilerBackend
    compiler_latency_ms: float
    emotion_id: Optional[int] = None
    source_user_text: str = ""
