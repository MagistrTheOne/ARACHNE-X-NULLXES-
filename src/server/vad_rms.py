"""Frame-level energy VAD with silence-based endpointing (real signal processing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RMSVADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    silence_ms_end: int = 450
    min_speech_ms: int = 200
    rms_threshold: float = 0.012
    max_buffer_sec: float = 25.0


class RMSUtteranceDetector:
    """
    Accumulates PCM float32 mono samples; when trailing silence exceeds threshold,
    returns one utterance slice.
    """

    def __init__(self, cfg: Optional[RMSVADConfig] = None) -> None:
        self.cfg = cfg or RMSVADConfig()
        self._frame_samples = max(1, int(self.cfg.sample_rate * self.cfg.frame_ms / 1000))
        self._silence_frames_needed = max(
            1, int(self.cfg.silence_ms_end / self.cfg.frame_ms)
        )
        self._min_speech_samples = max(
            self._frame_samples,
            int(self.cfg.sample_rate * self.cfg.min_speech_ms / 1000),
        )
        self._max_samples = int(self.cfg.sample_rate * self.cfg.max_buffer_sec)
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros((0,), dtype=np.float32)
        self._speech_active = False
        self._silence_frames = 0
        self._utterance_start = 0

    def _frame_rms(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))

    def push_pcm_f32_mono(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        if chunk.size == 0:
            return None
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, x])
        if self._pending.size > self._max_samples:
            self._pending = self._pending[-self._max_samples :]
            if self._speech_active:
                self._utterance_start = max(0, self._utterance_start - (x.size))

        utterance: Optional[np.ndarray] = None
        offset = 0
        while offset + self._frame_samples <= self._pending.size:
            frame = self._pending[offset : offset + self._frame_samples]
            offset += self._frame_samples
            rms = self._frame_rms(frame)
            is_voice = rms >= self.cfg.rms_threshold
            if not self._speech_active:
                if is_voice:
                    self._speech_active = True
                    self._silence_frames = 0
                    self._utterance_start = offset - self._frame_samples
            else:
                if is_voice:
                    self._silence_frames = 0
                else:
                    self._silence_frames += 1
                    if self._silence_frames >= self._silence_frames_needed:
                        end = offset - self._silence_frames * self._frame_samples
                        if end > self._utterance_start:
                            span = end - self._utterance_start
                            if span >= self._min_speech_samples:
                                utterance = self._pending[self._utterance_start : end].copy()
                        self._pending = self._pending[end:].copy()
                        offset = 0
                        self._speech_active = False
                        self._silence_frames = 0
                        self._utterance_start = 0
                        break

        if offset > 0:
            self._pending = self._pending[offset:].copy()
            if self._speech_active:
                self._utterance_start = max(0, self._utterance_start - offset)

        return utterance
