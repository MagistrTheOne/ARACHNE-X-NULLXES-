"""Session-scoped memory for digital actor (V2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
