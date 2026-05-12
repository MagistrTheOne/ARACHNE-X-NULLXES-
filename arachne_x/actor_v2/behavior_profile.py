"""Per-employee behavioral defaults (V2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BehaviorProfile:
    """Loaded from employee_packs ``behavior_profile.json`` (future)."""

    display_name: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)
