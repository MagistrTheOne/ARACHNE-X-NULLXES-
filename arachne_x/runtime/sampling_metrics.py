"""
Runtime sampling metrics (TTFF, DiT forwards, chunk stats) for operational tuning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RuntimeSamplingMetrics:
    runtime_profile: Optional[str] = None
    dit_forwards: int = 0
    cfg_passes_per_step: int = 3
    chunk_count: int = 0
    frames_total: int = 0
    frames_per_chunk: list[int] = field(default_factory=list)
    kv_cache_hits: int = 0
    cross_chunk_kv_frames: int = 0
    denoise_wall_sec: float = 0.0
    ttff_sec: Optional[float] = None
    silence_ratio: Optional[float] = None
    audio_guidance_scale_effective: Optional[float] = None
    identity_cosine_per_chunk: list[float] = field(default_factory=list)
    identity_drift_min: Optional[float] = None
    corrective_actions: list[str] = field(default_factory=list)
    _t0: Optional[float] = None
    _first_emit: bool = False

    def mark_start(self) -> None:
        self._t0 = time.perf_counter()

    def record_dit_forward(self, n: int = 1) -> None:
        self.dit_forwards += int(n)

    def add_denoise_elapsed(self, sec: float) -> None:
        self.denoise_wall_sec += float(sec)

    def mark_first_frame_emit(self) -> None:
        if self._first_emit:
            return
        self._first_emit = True
        if self._t0 is not None:
            self.ttff_sec = time.perf_counter() - self._t0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime_profile": self.runtime_profile,
            "dit_forwards": self.dit_forwards,
            "cfg_passes_per_step": self.cfg_passes_per_step,
            "chunk_count": self.chunk_count,
            "frames_total": self.frames_total,
            "frames_per_chunk": list(self.frames_per_chunk),
            "kv_cache_hits": self.kv_cache_hits,
            "cross_chunk_kv_frames": self.cross_chunk_kv_frames,
            "silence_ratio": self.silence_ratio,
            "audio_guidance_scale_effective": self.audio_guidance_scale_effective,
            "identity_cosine_per_chunk": list(self.identity_cosine_per_chunk),
            "identity_drift_min": self.identity_drift_min,
            "corrective_actions": list(self.corrective_actions),
            "denoise_wall_sec": round(self.denoise_wall_sec, 4),
            "ttff_sec": round(self.ttff_sec, 4) if self.ttff_sec is not None else None,
        }
