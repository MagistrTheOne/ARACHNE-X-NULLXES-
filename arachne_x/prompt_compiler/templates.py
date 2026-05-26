"""Avatar-specific prompt defaults (UMT5 input strings, not embeddings)."""

from __future__ import annotations

import re
from typing import Optional

DEFAULT_AVATAR_NEGATIVE = (
    "camera zoom, camera pan, camera tilt, dolly, handheld shake, "
    "blur, motion blur, distorted face, extra limbs, duplicate face, "
    "low quality, watermark, text overlay"
)

AVATAR_POSITIVE_SUFFIX_EN = (
    " The person speaks clearly to camera with accurate lip sync. "
    "Static camera, fixed framing, no camera movement."
)

AVATAR_POSITIVE_SUFFIX_ZH = (
    " 人物面向镜头清晰说话，口型与语音同步。固定机位，镜头不移动，无推拉摇移。"
)

_AVATAR_MODES = frozenset({"ai2v", "at2v", "streaming_ai2v", "avc"})
_IMAGINE_MODES = frozenset({"imagine_i2v", "audio_i2v"})

IMAGINE_POSITIVE_SUFFIX_EN = (
    " The subject speaks naturally to camera with audio-driven motion, subtle head movement, "
    "natural blinking, and stable identity. Static camera, fixed framing."
)

IMAGINE_POSITIVE_SUFFIX_ZH = (
    " 人物面向镜头自然说话，动作与语音节奏一致，轻微头部运动，自然眨眼，身份稳定。固定机位。"
)


def is_chinese_text(text: str) -> bool:
    valid = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text or "")
    if not valid:
        return False
    chinese = [c for c in valid if "\u4e00" <= c <= "\u9fff"]
    return len(chinese) / len(valid) > 0.25


def merge_avatar_defaults(
    positive: str,
    negative: str,
    *,
    mode: str,
    locale: str = "auto",
) -> tuple[str, str]:
    """Apply avatar lipsync / static-camera hints without an LLM."""
    pos = (positive or "").strip()
    neg = (negative or "").strip()

    if mode in _IMAGINE_MODES:
        use_zh = locale == "zh" or (locale == "auto" and is_chinese_text(pos))
        suffix = IMAGINE_POSITIVE_SUFFIX_ZH if use_zh else IMAGINE_POSITIVE_SUFFIX_EN
        if suffix.strip() not in pos:
            pos = (pos + suffix).strip()
        if not neg:
            neg = DEFAULT_AVATAR_NEGATIVE
        elif DEFAULT_AVATAR_NEGATIVE not in neg:
            neg = f"{neg}, {DEFAULT_AVATAR_NEGATIVE}"
        return pos, neg

    if mode not in _AVATAR_MODES:
        if not neg:
            neg = DEFAULT_AVATAR_NEGATIVE
        return pos, neg

    use_zh = locale == "zh" or (locale == "auto" and is_chinese_text(pos))
    suffix = AVATAR_POSITIVE_SUFFIX_ZH if use_zh else AVATAR_POSITIVE_SUFFIX_EN
    if suffix.strip() not in pos:
        pos = (pos + suffix).strip()

    if not neg:
        neg = DEFAULT_AVATAR_NEGATIVE
    elif DEFAULT_AVATAR_NEGATIVE not in neg:
        neg = f"{neg}, {DEFAULT_AVATAR_NEGATIVE}"

    return pos, neg


def truncate_for_log(text: str, max_len: int = 120) -> str:
    t = (text or "").replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."
