from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
        video = data.get("video") or {}
        tts = data.get("tts") or {}
        safety = data.get("safety") or {}
        return cls(
            character=str(data.get("character") or "megan"),
            normalized_user_text=str(data.get("normalized_user_text") or ""),
            reply_text=str(data.get("reply_text") or ""),
            video=VideoPlan(
                enabled=bool(video.get("enabled", True)),
                mode=str(video.get("mode") or "t2v"),
                profile=str(video.get("profile") or "fast_distill_9x16"),
                positive_prompt=str(video.get("positive_prompt") or ""),
                negative_prompt=str(video.get("negative_prompt") or ""),
                seed=int(video.get("seed", 778)),
            ),
            tts=TtsPlan(
                enabled=bool(tts.get("enabled", True)),
                speaker=str(tts.get("speaker") or "Serena"),
                language=str(tts.get("language") or "English"),
                instruct=str(
                    tts.get("instruct")
                    or "Speak in a warm, confident, elegant female corporate voice with calm executive presence."
                ),
            ),
            safety=SafetyPlan(
                mode=str(safety.get("mode") or "prod"),
                allowed=bool(safety.get("allowed", True)),
                notes=[str(x) for x in safety.get("notes", [])],
            ),
        )

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
    character: str = "megan"
    input: Dict[str, Any] = field(default_factory=dict)
    action_plan: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
