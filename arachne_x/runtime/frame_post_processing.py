"""
Frame post-processing chain for avatar / video output.

Doctrine: resolution policy != restoration policy.

The runtime emits a *canonical* frame (avatar = 720p). Restoration / upscale to
1080p+ is an explicit, ordered post-processing chain that runs *after* generation
and knows nothing about the generator, the diffusion schedule, or the bucket.

Operational properties:
- explicit lifecycle: stages are registered once, the chain is built once per stream.
- runtime metrics per stage: frames, ms, bypass count, last in/out HxW.
- graceful degradation: a per-frame budget bypasses trailing stages instead of
  blocking the realtime stream; a failing stage is bypassed, never fatal.
- no shipped weights, no stub backends: only real, dependency-light stages
  (``passthrough``, ``lanczos``) are built in. Heavy restorers
  (``realesrgan_trt`` / ``seedvr2`` / ``flashvsr``) attach via ``REGISTRY.register``
  from their own module *when their backend is actually present* — they are never
  faked here.

The seam is opt-in. With no configuration the chain is empty and ``process`` is a
no-op passthrough, so the hot path pays nothing.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
@dataclass
class StageMetrics:
    name: str
    frames: int = 0
    total_ms: float = 0.0
    bypassed: int = 0
    failed: int = 0
    last_in_hw: Optional[tuple[int, int]] = None
    last_out_hw: Optional[tuple[int, int]] = None

    def observe(self, dt_ms: float, in_hw: tuple[int, int], out_hw: tuple[int, int]) -> None:
        self.frames += 1
        self.total_ms += float(dt_ms)
        self.last_in_hw = in_hw
        self.last_out_hw = out_hw

    @property
    def avg_ms(self) -> float:
        return (self.total_ms / self.frames) if self.frames else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "frames": self.frames,
            "avg_ms": round(self.avg_ms, 3),
            "bypassed": self.bypassed,
            "failed": self.failed,
            "last_in_hw": self.last_in_hw,
            "last_out_hw": self.last_out_hw,
        }


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
class FrameProcessorStage:
    """Base stage. The default implementation is a real passthrough (not a stub).

    Stages operate on a single uint8 HWC RGB frame. Batch-oriented restorers may
    override :meth:`process_batch` for temporal coherence; the default maps
    per-frame.
    """

    name: str = "passthrough"

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame

    def process_batch(self, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        return [self.process(f) for f in frames]

    def close(self) -> None:  # lifecycle hook for stages owning GPU/engine handles
        return None


class LanczosUpscaleStage(FrameProcessorStage):
    """Dependency-light spatial upscale via PIL Lanczos, preserving aspect ratio.

    This is the safe baseline restorer: deterministic, no weights, no GPU. It is
    not a quality SR model — it exists so the chain is always functional and a
    higher-tier restorer can be slotted in without touching call sites.
    """

    name = "lanczos"

    def __init__(self, target_short_edge: int = 1080) -> None:
        if int(target_short_edge) < 16:
            raise ValueError(f"target_short_edge must be a pixel size >= 16, got {target_short_edge!r}")
        # Resolution tier follows "p" semantics: 720p/1080p == short edge in pixels.
        # 720p (short edge 720) -> 1080p (short edge 1080) is a 1.5x upscale.
        self.target_short_edge = int(target_short_edge)
        from PIL import Image  # lazy import; PIL is already a runtime dep

        self._Image = Image
        self._resample = Image.Resampling.LANCZOS

    def _target_hw(self, h: int, w: int) -> tuple[int, int]:
        short_edge = min(h, w)
        if short_edge >= self.target_short_edge:
            return h, w  # upscale-only; never downscale below the canonical frame
        scale = self.target_short_edge / float(short_edge)
        # keep even dimensions (encoder-friendly)
        th = int(round(h * scale / 2.0)) * 2
        tw = int(round(w * scale / 2.0)) * 2
        return max(th, 2), max(tw, 2)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = int(frame.shape[0]), int(frame.shape[1])
        th, tw = self._target_hw(h, w)
        if (th, tw) == (h, w):
            return frame
        img = self._Image.fromarray(frame)
        img = img.resize((tw, th), self._resample)
        return np.asarray(img)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
ProcessorFactory = Callable[..., FrameProcessorStage]


class ProcessorRegistry:
    """Name -> stage factory. Extensible at import time by backend modules."""

    def __init__(self) -> None:
        self._factories: Dict[str, ProcessorFactory] = {}

    def register(self, name: str, factory: ProcessorFactory, *, override: bool = False) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("processor name must be non-empty")
        if key in self._factories and not override:
            raise ValueError(f"processor {key!r} already registered (pass override=True to replace)")
        self._factories[key] = factory

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._factories

    def create(self, name: str, **kwargs) -> FrameProcessorStage:
        key = name.strip().lower()
        try:
            factory = self._factories[key]
        except KeyError:
            raise KeyError(
                f"unknown frame processor {name!r}; available={self.available()}"
            ) from None
        return factory(**kwargs)

    def available(self) -> List[str]:
        return sorted(self._factories.keys())


REGISTRY = ProcessorRegistry()
REGISTRY.register("none", lambda **_: FrameProcessorStage())
REGISTRY.register("passthrough", lambda **_: FrameProcessorStage())
REGISTRY.register(
    "lanczos",
    lambda *, target: LanczosUpscaleStage(target_short_edge=int(target) if target is not None else 1080),
)


# --------------------------------------------------------------------------- #
# chain
# --------------------------------------------------------------------------- #
class FrameProcessorChain:
    """Ordered stages with a per-frame budget and runtime metrics."""

    def __init__(
        self,
        stages: Sequence[FrameProcessorStage],
        *,
        budget_ms: Optional[float] = None,
    ) -> None:
        self.stages: List[FrameProcessorStage] = [s for s in stages if s.name not in ("none", "passthrough")]
        self.budget_ms = float(budget_ms) if budget_ms is not None else None
        self.metrics: Dict[str, StageMetrics] = {s.name: StageMetrics(s.name) for s in self.stages}
        self.degraded_frames = 0

    def __bool__(self) -> bool:
        return bool(self.stages)

    def describe(self) -> str:
        names = "->".join(s.name for s in self.stages) or "none"
        budget = f" budget_ms={self.budget_ms}" if self.budget_ms is not None else ""
        return f"{names}{budget}"

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.stages:
            return frame
        t_frame = time.perf_counter()
        out = frame
        degraded = False
        for stage in self.stages:
            if self.budget_ms is not None and not degraded:
                spent_ms = (time.perf_counter() - t_frame) * 1000.0
                if spent_ms >= self.budget_ms:
                    degraded = True
            if degraded:
                self.metrics[stage.name].bypassed += 1
                continue
            in_hw = (int(out.shape[0]), int(out.shape[1]))
            t0 = time.perf_counter()
            try:
                out = stage.process(out)
            except Exception:
                self.metrics[stage.name].failed += 1
                logger.warning("frame stage %s failed; bypassing (frame kept)", stage.name, exc_info=True)
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics[stage.name].observe(dt_ms, in_hw, (int(out.shape[0]), int(out.shape[1])))
        if degraded:
            self.degraded_frames += 1
        return out

    def process_batch(self, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        # Offline path: no per-frame budget, allow batch-aware stages.
        if not self.stages:
            return list(frames)
        out = list(frames)
        for stage in self.stages:
            t0 = time.perf_counter()
            try:
                out = list(stage.process_batch(out))
            except Exception:
                self.metrics[stage.name].failed += 1
                logger.warning("frame stage %s batch failed; bypassing", stage.name, exc_info=True)
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            m = self.metrics[stage.name]
            if out:
                m.observe(dt_ms / max(len(out), 1), (int(out[0].shape[0]), int(out[0].shape[1])), (int(out[0].shape[0]), int(out[0].shape[1])))
        return out

    def metrics_dict(self) -> Dict[str, object]:
        return {
            "chain": self.describe(),
            "degraded_frames": self.degraded_frames,
            "stages": [m.to_dict() for m in self.metrics.values()],
        }

    def close(self) -> None:
        for stage in self.stages:
            try:
                stage.close()
            except Exception:
                logger.debug("stage %s close() failed", stage.name, exc_info=True)


# --------------------------------------------------------------------------- #
# spec parsing
# --------------------------------------------------------------------------- #
def _parse_stage_token(token: str) -> tuple[str, Optional[int]]:
    """``"lanczos:1080"`` -> ``("lanczos", 1080)``; ``"face_restore"`` -> ``("face_restore", None)``."""
    name, _, arg = token.strip().partition(":")
    name = name.strip().lower()
    target: Optional[int] = None
    arg = arg.strip()
    if arg:
        try:
            target = int(arg)
        except ValueError:
            raise ValueError(f"frame postfx target must be an integer pixel size, got {arg!r}") from None
    return name, target


def build_chain_from_spec(spec: Optional[str], *, budget_ms: Optional[float] = None) -> FrameProcessorChain:
    """Build a chain from a spec string.

    Spec grammar: comma-separated ``name[:target]`` tokens, e.g.
    ``"lanczos:1080"`` or ``"face_restore,realesrgan_trt:1080"``. Empty / ``"none"``
    yields an empty (no-op) chain.
    """
    spec = (spec or "").strip()
    if not spec or spec.lower() == "none":
        return FrameProcessorChain([], budget_ms=budget_ms)

    stages: List[FrameProcessorStage] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        name, target = _parse_stage_token(token)
        if name in ("none", "passthrough"):
            continue
        stages.append(REGISTRY.create(name, target=target))
    return FrameProcessorChain(stages, budget_ms=budget_ms)


def resolve_chain_from_env(*, default_spec: str = "") -> FrameProcessorChain:
    """Resolve a chain from ``NULLXES_FRAME_POSTFX`` / ``ARACHNE_FRAME_POSTFX``.

    Budget from ``NULLXES_FRAME_POSTFX_BUDGET_MS`` (per-frame, realtime degrade).
    Defaults to an empty no-op chain so the hot path is free unless configured.
    """
    spec = ""
    for key in ("NULLXES_FRAME_POSTFX", "ARACHNE_FRAME_POSTFX"):
        v = (os.environ.get(key) or "").strip()
        if v:
            spec = v
            break
    if not spec:
        spec = default_spec

    budget_ms: Optional[float] = None
    for key in ("NULLXES_FRAME_POSTFX_BUDGET_MS", "ARACHNE_FRAME_POSTFX_BUDGET_MS"):
        v = (os.environ.get(key) or "").strip()
        if v:
            try:
                budget_ms = float(v)
            except ValueError:
                logger.warning("ignoring non-numeric %s=%r", key, v)
            break

    try:
        return build_chain_from_spec(spec, budget_ms=budget_ms)
    except (KeyError, ValueError) as exc:
        logger.warning("invalid frame postfx spec %r (%s); running no-op chain", spec, exc)
        return FrameProcessorChain([], budget_ms=budget_ms)
