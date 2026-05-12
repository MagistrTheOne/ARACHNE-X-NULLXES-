"""Inter-frame stabilization policy (V2) — lightweight hooks for E-TEMP alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class TemporalGovernor:
    """
    Temporal stabilization between frames (anti-jitter, parameter smoothing).

    Does not replace diffusion steps; operates on decoded frames or control scalars.
    """

    smooth_alpha: float = 0.35
    _prev_control: Optional[np.ndarray] = None

    def smooth_scalar(self, value: float) -> float:
        x = float(value)
        if self._prev_control is None:
            self._prev_control = np.array([x], dtype=np.float64)
            return x
        prev = float(self._prev_control[0])
        y = prev + self.smooth_alpha * (x - prev)
        self._prev_control = np.array([y], dtype=np.float64)
        return y

    def reset(self) -> None:
        self._prev_control = None

    def postprocess_frame_rgb(self, frame_hwc: np.ndarray) -> np.ndarray:
        """Placeholder pass-through; replace with optical-flow / flicker policy when wired."""
        return np.asarray(frame_hwc)
