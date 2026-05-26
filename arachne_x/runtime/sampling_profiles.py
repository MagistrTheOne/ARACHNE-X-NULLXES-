"""
Operational vs cinematic sampling profiles (ARACHNE-X Sampling OS).

Explicit CLI flags override profile defaults (see ``apply_sampling_profile``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

RuntimeProfileName = Literal["operational", "cinematic"]


@dataclass(frozen=True)
class SamplingProfile:
    num_inference_steps: int
    use_distill: bool
    num_frames_mode: str
    num_frames_cap: Optional[int]
    text_guidance_scale: float
    audio_guidance_scale: float
    resolution: str
    chunk_frames: int
    chunk_overlap: int
    use_chunked_denoise: bool


PROFILES: dict[str, SamplingProfile] = {
    "operational": SamplingProfile(
        num_inference_steps=12,
        use_distill=True,
        num_frames_mode="sync",
        num_frames_cap=65,
        text_guidance_scale=4.0,
        audio_guidance_scale=5.0,
        resolution="720p",
        chunk_frames=33,
        chunk_overlap=8,
        use_chunked_denoise=True,
    ),
    "cinematic": SamplingProfile(
        num_inference_steps=35,
        use_distill=False,
        num_frames_mode="explicit",
        num_frames_cap=None,
        text_guidance_scale=4.0,
        audio_guidance_scale=5.5,
        resolution="720p",
        chunk_frames=97,
        chunk_overlap=0,
        use_chunked_denoise=False,
    ),
}


def get_profile(name: str) -> SamplingProfile:
    key = (name or "cinematic").strip().lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown runtime_profile {name!r}; use operational or cinematic.")
    return PROFILES[key]


def resolve_runtime_profile_from_env() -> Optional[str]:
    import os

    for env_key in ("ARACHNE_RUNTIME_PROFILE", "NULLXES_RUNTIME_PROFILE"):
        v = (os.environ.get(env_key) or "").strip().lower()
        if v:
            return v
    return None


def apply_sampling_profile(args, argv: Optional[list[str]] = None) -> Optional[SamplingProfile]:
    """
    Merge profile into ``args``. CLI argv wins when the flag appears explicitly.
    """
    import sys

    name = getattr(args, "runtime_profile", None) or resolve_runtime_profile_from_env()
    if not name:
        return None
    profile = get_profile(str(name))
    if argv is None:
        argv = sys.argv[1:]
    argv_joined = " ".join(argv)

    def _set(attr: str, value) -> None:
        flag = f"--{attr.replace('_', '-')}"
        if flag not in argv_joined:
            setattr(args, attr, value)

    _set("num_inference_steps", profile.num_inference_steps)
    _set("use_distill", profile.use_distill)
    _set("num_frames_mode", profile.num_frames_mode)
    _set("resolution", profile.resolution)
    _set("text_guidance_scale", profile.text_guidance_scale)
    _set("audio_guidance_scale", profile.audio_guidance_scale)
    _set("chunk_frames", profile.chunk_frames)
    _set("chunk_overlap", profile.chunk_overlap)
    _set("use_chunked_denoise", profile.use_chunked_denoise)

    args._sampling_profile_name = str(name).lower()
    args._sampling_profile_num_frames_cap = profile.num_frames_cap
    return profile


def resolve_use_distill(args) -> bool:
    explicit = getattr(args, "use_distill", None)
    if explicit is not None and bool(explicit):
        return True
    steps = int(getattr(args, "num_inference_steps", 50))
    return bool(getattr(args, "use_distill", False)) or steps <= 16


def cap_num_frames_for_profile(args, chosen: int) -> int:
    cap = getattr(args, "_sampling_profile_num_frames_cap", None)
    if cap is not None and int(chosen) > int(cap):
        return int(cap)
    return int(chosen)
