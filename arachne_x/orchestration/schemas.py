from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


VALID_SAFETY_MODES = {"prod", "redteam"}
VALID_PLAN_MODES = {"t2v"}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if value is None:
        return default
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class TurnInput:
    text: Optional[str] = None
    audio_path: Optional[str] = None
    character: str = "megan"
    output_dir: str = "output/turn"
    safety_mode: str = "prod"
    video_profile: str = "fast_distill_9x16"
    enable_tts: bool = True
    enable_video: bool = True
    job_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TtsPlan:
    enabled: bool = True
    speaker: str = "Serena"
    language: str = "English"
    instruct: str = "Speak in a warm, confident, elegant female corporate voice with calm executive presence."


@dataclass
class VideoPlan:
    enabled: bool = True
    mode: str = "t2v"
    profile: str = "fast_distill_9x16"
    positive_prompt: str = ""
    negative_prompt: str = ""
    seed: int = 778


@dataclass
class SafetyPlan:
    mode: str = "prod"
    allowed: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class ActionPlan:
    character: str = "megan"
    normalized_user_text: str = ""
    reply_text: str = ""
    video: VideoPlan = field(default_factory=VideoPlan)
    tts: TtsPlan = field(default_factory=TtsPlan)
    safety: SafetyPlan = field(default_factory=SafetyPlan)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionPlan":
        if not isinstance(data, dict):
            raise TypeError("ActionPlan payload must be a JSON object")
        video = data.get("video") or {}
        tts = data.get("tts") or {}
        safety = data.get("safety") or {}
        if not isinstance(video, dict):
            video = {}
        if not isinstance(tts, dict):
            tts = {}
        if not isinstance(safety, dict):
            safety = {}
        return cls(
            character=str(data.get("character") or "megan"),
            normalized_user_text=str(data.get("normalized_user_text") or ""),
            reply_text=str(data.get("reply_text") or ""),
            video=VideoPlan(
                enabled=_coerce_bool(video.get("enabled"), True),
                mode=str(video.get("mode") or "t2v"),
                profile=str(video.get("profile") or "fast_distill_9x16"),
                positive_prompt=str(video.get("positive_prompt") or ""),
                negative_prompt=str(video.get("negative_prompt") or ""),
                seed=_coerce_int(video.get("seed"), 778),
            ),
            tts=TtsPlan(
                enabled=_coerce_bool(tts.get("enabled"), True),
                speaker=str(tts.get("speaker") or "Serena"),
                language=str(tts.get("language") or "English"),
                instruct=str(
                    tts.get("instruct")
                    or "Speak in a warm, confident, elegant female corporate voice with calm executive presence."
                ),
            ),
            safety=SafetyPlan(
                mode=str(safety.get("mode") or "prod"),
                allowed=_coerce_bool(safety.get("allowed"), True),
                notes=[str(x) for x in safety.get("notes", [])],
            ),
        )

    def validate(self, *, allowed_video_profiles: Optional[List[str]] = None) -> List[str]:
        errors: List[str] = []
        if not self.character.strip():
            errors.append("character is required")
        if self.safety.mode not in VALID_SAFETY_MODES:
            errors.append(f"safety.mode must be one of {sorted(VALID_SAFETY_MODES)}")
        if self.video.enabled:
            if self.video.mode not in VALID_PLAN_MODES:
                errors.append(f"video.mode must be one of {sorted(VALID_PLAN_MODES)}")
            if allowed_video_profiles is not None and self.video.profile not in allowed_video_profiles:
                errors.append(f"video.profile is not configured: {self.video.profile}")
            if not self.video.positive_prompt.strip():
                errors.append("video.positive_prompt is required when video is enabled")
            if not (0 <= self.video.seed <= 2**32 - 1):
                errors.append("video.seed must be in uint32 range")
        if self.tts.enabled:
            if not self.reply_text.strip():
                errors.append("reply_text is required when tts is enabled")
            if not self.tts.speaker.strip():
                errors.append("tts.speaker is required when tts is enabled")
            if self.tts.language.strip().lower() != "english":
                errors.append("tts.language must be English for the current Qwen3-TTS preset")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactPaths:
    input_audio: Optional[str] = None
    asr_text: Optional[str] = None
    action_plan: Optional[str] = None
    reply_text: Optional[str] = None
    tts_wav: Optional[str] = None
    video_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    video_mp4: Optional[str] = None


@dataclass
class TurnManifest:
    project: str = "NULLXES Project FURIA: ARACHNE-X EIDOLON"
    date: str = ""
    status: str = "pending"
    character: str = "megan"
    job_id: Optional[str] = None
    session_id: Optional[str] = None
    input: Dict[str, Any] = field(default_factory=dict)
    action_plan: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    validation_notes: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    qa: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
