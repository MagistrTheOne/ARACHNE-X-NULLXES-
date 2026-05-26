"""
Infer-time attention backend configuration (BSA parity control).

NULLXES policy — BSA is inference-only.
Do NOT enable during LoRA training.
Sparse attention during training causes train/infer distribution drift and
temporal artifacts (flicker, identity bleed, manifold collapse on long videos).
See ARCHITECTURE.md → "Runtime" and "Identity LoRA" sections.

Training code MUST call ``dit.disable_bsa()`` before forward; ``configure_infer_bsa``
is a runtime-only helper and is never imported from ``scripts/train_lora_avatar.py``.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def infer_bsa_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("ARACHNE_INFER_ENABLE_BSA", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def configure_infer_bsa(dit: Any, *, enabled: Optional[bool] = None) -> bool:
    """
    Enable or disable block-sparse attention on DiT for inference.

    Default ON (``ARACHNE_INFER_ENABLE_BSA=1``). Set to ``0`` for train/infer
    parity debugging (dense flash-attn only).
    """
    use_bsa = infer_bsa_enabled(enabled)
    if use_bsa and hasattr(dit, "enable_bsa"):
        dit.enable_bsa()
    elif hasattr(dit, "disable_bsa"):
        dit.disable_bsa()
    return use_bsa
