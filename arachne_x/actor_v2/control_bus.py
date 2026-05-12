"""Typed control signals aggregated before pipeline / runtime calls (V2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ControlBus:
    """
    Unified behavioral control layer for NIGHT FURY V2.

    Maps to existing pipeline kwargs where applicable (emotion_id, emotion_intensity, etc.).
    """

    emotion: Optional[str] = None
    speech_rate: float = 1.0
    gaze: Optional[Dict[str, float]] = None
    intensity: float = 1.0
    head_motion: float = 0.0
    blink_state: float = 0.0
    speaking_state: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_pipeline_emotion_kwargs(self) -> Dict[str, Any]:
        """Subset compatible with avatar pipeline emotion_* parameters (best-effort)."""
        out: Dict[str, Any] = {}
        if self.emotion is not None:
            out["emotion_hint"] = self.emotion
        out["emotion_intensity"] = float(self.intensity)
        return out
