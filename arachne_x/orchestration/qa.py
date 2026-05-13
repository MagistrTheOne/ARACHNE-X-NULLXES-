from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def check_artifact(path: Optional[str], *, min_bytes: int = 1) -> Dict[str, Any]:
    if not path:
        return {"ok": False, "reason": "missing_path"}
    p = Path(path)
    if not p.exists():
        return {"ok": False, "path": str(p), "reason": "not_found"}
    size = p.stat().st_size
    if size < min_bytes:
        return {"ok": False, "path": str(p), "size_bytes": size, "reason": "too_small"}
    return {"ok": True, "path": str(p), "size_bytes": size}


def check_turn_artifacts(*, tts_wav: Optional[str], video_mp4: Optional[str]) -> Dict[str, Any]:
    return {
        "tts_wav": check_artifact(tts_wav, min_bytes=1024) if tts_wav else {"ok": True, "skipped": True},
        "video_mp4": check_artifact(video_mp4, min_bytes=1024) if video_mp4 else {"ok": True, "skipped": True},
    }
