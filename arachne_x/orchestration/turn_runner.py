from __future__ import annotations

import datetime as _dt
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters.asr import run_asr
from .adapters.planner import run_planner
from .adapters.tts import run_tts
from .adapters.video import run_video
from .policy import apply_policy
from .presets import get_character_preset
from .schemas import ActionPlan, TurnInput, TurnManifest
from .subprocess_utils import default_python, write_json


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True)
        return out.strip()
    except Exception:
        return None


def _gpu_info(python_bin: str, repo_root: Path) -> Dict[str, Any]:
    code = (
        "import json, torch; "
        "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    try:
        out = subprocess.check_output([python_bin, "-c", code], cwd=str(repo_root), text=True)
        import json

        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {}


def _detect_stage3_attn(stage3_python: str, repo_root: Path) -> str:
    code = "import importlib.util; print('flash_attention_2' if importlib.util.find_spec('flash_attn') else 'sdpa')"
    try:
        out = subprocess.check_output([stage3_python, "-c", code], cwd=str(repo_root), text=True)
        return out.strip().splitlines()[-1] or "sdpa"
    except Exception:
        return "sdpa"


def _ensure_prompts(plan: ActionPlan, character: str) -> None:
    preset = get_character_preset(character)
    if not plan.video.positive_prompt.strip():
        plan.video.positive_prompt = preset["positive_prompt"]
    if not plan.video.negative_prompt.strip():
        plan.video.negative_prompt = preset["negative_prompt"]
    if not plan.reply_text.strip():
        plan.reply_text = (
            "Hello. I am Megan from NULLXES, and I am ready to coordinate the next phase with calm precision."
        )
    if not plan.tts.speaker.strip():
        plan.tts.speaker = preset["speaker"]
    if not plan.tts.instruct.strip():
        plan.tts.instruct = preset["tts_instruct"]


def run_turn(
    turn: TurnInput,
    *,
    repo_root: str | Path = ".",
    video_python: Optional[str] = None,
    stage3_python: Optional[str] = None,
    whisper_model: str = "/workspace/ARACHNE-X/weights/openai-whisper-large-v3-turbo",
    llm_model: str = "/workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507",
    tts_model: str = "/workspace/ARACHNE-X/weights/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    video_checkpoint: str = "/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO",
    attn_implementation: str = "auto",
) -> TurnManifest:
    root = Path(repo_root).resolve()
    video_py = video_python or default_python(root, ".venv")
    stage3_py = stage3_python or default_python(root, ".venv_stage3")
    attn = _detect_stage3_attn(stage3_py, root) if attn_implementation == "auto" else attn_implementation

    out_dir = Path(turn.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    input_audio = None
    user_text = turn.text or ""
    timings: Dict[str, float] = {}

    if turn.audio_path:
        src = Path(turn.audio_path)
        input_audio = str(out_dir / f"input{src.suffix or '.wav'}")
        shutil.copyfile(src, input_audio)
        asr_result, asr_elapsed = run_asr(
            python_bin=stage3_py,
            work_dir=out_dir,
            audio_path=input_audio,
            model_path=whisper_model,
        )
        timings["asr_sec"] = round(asr_elapsed, 3)
        user_text = str(asr_result.get("text") or "")

    plan_dict, planner_elapsed = run_planner(
        python_bin=stage3_py,
        work_dir=out_dir,
        model_path=llm_model,
        character=turn.character,
        user_text=user_text,
        video_profile=turn.video_profile,
        enable_tts=turn.enable_tts,
        enable_video=turn.enable_video,
        safety_mode=turn.safety_mode,
        attn_implementation=attn,
    )
    timings["planner_sec"] = round(planner_elapsed, 3)

    plan = ActionPlan.from_dict(plan_dict)
    plan.tts.enabled = plan.tts.enabled and turn.enable_tts
    plan.video.enabled = plan.video.enabled and turn.enable_video
    _ensure_prompts(plan, turn.character)
    plan = apply_policy(plan, turn.safety_mode)

    action_plan_path = out_dir / "action_plan.json"
    write_json(action_plan_path, plan.to_dict())
    (out_dir / "reply.txt").write_text(plan.reply_text, encoding="utf-8")
    (out_dir / "video_prompt.txt").write_text(plan.video.positive_prompt, encoding="utf-8")
    (out_dir / "negative_prompt.txt").write_text(plan.video.negative_prompt, encoding="utf-8")

    tts_path = None
    if plan.tts.enabled and plan.safety.allowed:
        tts_result, tts_elapsed = run_tts(
            python_bin=stage3_py,
            work_dir=out_dir,
            model_path=tts_model,
            text=plan.reply_text,
            speaker=plan.tts.speaker,
            language=plan.tts.language,
            instruct=plan.tts.instruct,
            attn_implementation=attn,
        )
        timings["tts_sec"] = round(tts_elapsed, 3)
        tts_path = str(tts_result.get("wav_path") or "")

    video_path = None
    if plan.video.enabled and plan.safety.allowed:
        video_result, video_elapsed = run_video(
            python_bin=video_py,
            work_dir=out_dir,
            checkpoint_dir=video_checkpoint,
            prompt=plan.video.positive_prompt,
            negative_prompt=plan.video.negative_prompt,
            profile_name=plan.video.profile,
            seed=plan.video.seed,
        )
        timings["video_sec"] = round(video_elapsed, 3)
        video_path = str(video_result.get("video_path") or "")

    now = _dt.datetime.now().isoformat(timespec="seconds")
    manifest = TurnManifest(
        date=now,
        character=turn.character,
        input={"text": turn.text, "audio_path": input_audio},
        action_plan=plan.to_dict(),
        artifacts={
            "asr_text": str(out_dir / "asr.txt") if turn.audio_path else None,
            "action_plan": str(action_plan_path),
            "reply_text": str(out_dir / "reply.txt"),
            "tts_wav": tts_path,
            "video_prompt": str(out_dir / "video_prompt.txt"),
            "negative_prompt": str(out_dir / "negative_prompt.txt"),
            "video_mp4": video_path,
        },
        timings=timings,
        models={
            "asr": whisper_model,
            "llm": llm_model,
            "tts": tts_model,
            "video": video_checkpoint,
        },
        runtime={
            "repo_root": str(root),
            "git_commit": _git_commit(root),
            "video_python": video_py,
            "stage3_python": stage3_py,
            "attn_implementation": attn,
            "gpu": _gpu_info(video_py, root),
        },
        safety=plan.safety.__dict__,
    )
    write_json(out_dir / "manifest.json", manifest.to_dict())
    return manifest
