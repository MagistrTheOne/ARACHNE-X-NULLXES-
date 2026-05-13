"""Session-scoped memory for digital actor (V2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionEmotionState:
    """Rolling emotion summary for a session (filled by emotion2vec adapter later)."""

    label: Optional[str] = None
    confidence: float = 0.0
    last_update_utc: Optional[str] = None


@dataclass
class SessionMemory:
    """
    Persistent session store — complements identity_bank on disk.

    Serialize/deserialize for employee_packs / worker session affinity.
    """

    session_id: str
    transcript_turns: List[str] = field(default_factory=list)
    emotion: SessionEmotionState = field(default_factory=SessionEmotionState)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.metadata["last_touch_utc"] = datetime.now(timezone.utc).isoformat()

    def append_transcript(self, text: str, *, max_turns: int = 32) -> None:
        t = (text or "").strip()
        if not t:
            return
        self.transcript_turns.append(t)
        if len(self.transcript_turns) > max_turns:
            self.transcript_turns = self.transcript_turns[-max_turns:]
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "transcript_turns": list(self.transcript_turns),
            "emotion": {
                "label": self.emotion.label,
                "confidence": self.emotion.confidence,
                "last_update_utc": self.emotion.last_update_utc,
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionMemory":
        emotion_payload = payload.get("emotion") or {}
        return cls(
            session_id=str(payload.get("session_id") or "default"),
            transcript_turns=[str(x) for x in payload.get("transcript_turns", [])],
            emotion=SessionEmotionState(
                label=emotion_payload.get("label"),
                confidence=float(emotion_payload.get("confidence") or 0.0),
                last_update_utc=emotion_payload.get("last_update_utc"),
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path, *, session_id: str) -> "SessionMemory":
        p = Path(path)
        if not p.exists():
            return cls(session_id=session_id)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
