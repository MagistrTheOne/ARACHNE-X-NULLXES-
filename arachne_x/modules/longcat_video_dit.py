"""
Legacy import path for the video DiT transformer.

Canonical implementation lives in :mod:`arachne_x.modules.arachne_video_dit`.
Import from there in new code; this module re-exports the same symbols for
checkpoint ABI compatibility (class names unchanged).
"""

from __future__ import annotations

from .arachne_video_dit import *  # noqa: F403
