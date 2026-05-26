"""
Per-chunk identity drift monitor (face ROI cosine vs anchor frame).

No new foundation weights — lightweight numpy embedding for runtime policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def _face_roi(frame: np.ndarray, margin: float = 0.12) -> np.ndarray:
    """Center crop approximating face ROI on [H,W,C] uint8 or float."""
    f = np.asarray(frame)
    if f.dtype != np.float32 and f.max() > 1.5:
        f = f.astype(np.float32) / 255.0
    h, w = f.shape[0], f.shape[1]
    mh, mw = int(h * margin), int(w * margin)
    roi = f[mh : h - mh, mw : w - mw]
    if roi.size == 0:
        roi = f
    # downscale for stable cosine
    from PIL import Image

    im = Image.fromarray((np.clip(roi, 0, 1) * 255).astype(np.uint8))
    im = im.resize((64, 64), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def frame_embedding(frame: np.ndarray) -> np.ndarray:
    roi = _face_roi(frame)
    vec = roi.reshape(-1).astype(np.float32)
    n = np.linalg.norm(vec) + 1e-8
    return vec / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


@dataclass
class IdentityDriftMonitor:
    warn_threshold: float = 0.88
    critical_threshold: float = 0.82
    anchor_emb: Optional[np.ndarray] = None
    per_chunk_cosine: List[float] = field(default_factory=list)
    corrective_actions: List[str] = field(default_factory=list)

    def set_anchor_from_frame(self, frame: np.ndarray) -> None:
        self.anchor_emb = frame_embedding(frame)

    def score_chunk_tail(self, chunk_video: np.ndarray, *, tail_frames: int = 4) -> float:
        if self.anchor_emb is None or chunk_video is None or int(chunk_video.shape[0]) < 1:
            return 1.0
        tail = int(min(tail_frames, chunk_video.shape[0]))
        emb = frame_embedding(chunk_video[-tail])
        cos = cosine_similarity(self.anchor_emb, emb)
        self.per_chunk_cosine.append(round(cos, 4))
        return cos

    def policy_for_next_chunk(self, cosine: float) -> Dict[str, Any]:
        """
        Deterministic corrective hints for the next chunk (no ML).
        """
        out: Dict[str, Any] = {
            "cosine": round(float(cosine), 4),
            "refresh_identity_tokens": False,
            "audio_guidance_scale_multiplier": 1.0,
        }
        if cosine < self.critical_threshold:
            out["refresh_identity_tokens"] = True
            out["audio_guidance_scale_multiplier"] = 0.82
            self.corrective_actions.append(f"critical_cosine={cosine:.3f}")
        elif cosine < self.warn_threshold:
            out["refresh_identity_tokens"] = True
            out["audio_guidance_scale_multiplier"] = 0.92
            self.corrective_actions.append(f"warn_cosine={cosine:.3f}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_cosine_per_chunk": list(self.per_chunk_cosine),
            "identity_drift_min": round(min(self.per_chunk_cosine), 4) if self.per_chunk_cosine else None,
            "identity_drift_max": round(max(self.per_chunk_cosine), 4) if self.per_chunk_cosine else None,
            "corrective_actions": list(self.corrective_actions),
        }
