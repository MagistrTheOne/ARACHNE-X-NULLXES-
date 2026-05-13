from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from arachne_x.actor_v2.session_memory import SessionMemory

from .adapters.asr import run_asr
from .adapters.planner import run_planner
from .adapters.tts import run_tts
from .adapters.video import run_video
from .policy import apply_policy
from .presets import VIDEO_PROFILES, get_character_preset
from .qa import check_turn_artifacts
from .schemas import ActionPlan, TurnInput, TurnManifest
from .subprocess_utils import SubprocessRunError, default_python, read_json, write_json


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


def _error_payload(stage: str, exc: BaseException) -> Dict[str, Any]:
    if isinstance(exc, SubprocessRunError):
        return exc.to_dict()
    return {"stage": stage, "message": str(exc), "type": exc.__class__.__name__}


def _session_path(root: Path, session_store_dir: Optional[str], session_id: str) -> Path:
    base = Path(session_store_dir) if session_store_dir else root / "output" / "sessions"
    if not base.is_absolute():
        base = root / base
    return base / f"{session_id}.json"


def _load_session(root: Path, session_store_dir: Optional[str], session_id: Optional[str]) -> Optional[SessionMemory]:
    if not session_id:
        return None
    return SessionMemory.load(_session_path(root, session_store_dir, session_id), session_id=session_id)


def _save_session(
    root: Path,
    session_store_dir: Optional[str],
    session: Optional[SessionMemory],
    *,
    user_text: str,
    reply_text: str,
    manifest_path: Path,
    job_id: Optional[str],
) -> None:
    if session is None:
        return
    session.append_transcript(f"user: {user_text}")
    session.append_transcript(f"assistant: {reply_text}")
    session.metadata["last_manifest"] = str(manifest_path)
    if job_id:
        session.metadata["last_job_id"] = job_id
    session.save(_session_path(root, session_store_dir, session.session_id))


def _write_plan_files(out_dir: Path, plan: ActionPlan) -> Path:
    action_plan_path = out_dir / "action_plan.json"
    write_json(action_plan_path, plan.to_dict())
    (out_dir / "reply.txt").write_text(plan.reply_text, encoding="utf-8")
    (out_dir / "video_prompt.txt").write_text(plan.video.positive_prompt, encoding="utf-8")
    (out_dir / "negative_prompt.txt").write_text(plan.video.negative_prompt, encoding="utf-8")
    return action_plan_path


def _write_manifest(out_dir: Path, manifest: TurnManifest) -> TurnManifest:
    write_json(out_dir / "manifest.json", manifest.to_dict())
    return manifest


def _build_manifest(
    *,
    out_dir: Path,
    root: Path,
    turn: TurnInput,
    status: str,
    plan: Optional[ActionPlan],
    input_audio: Optional[str],
    timings: Dict[str, float],
    models: Dict[str, str],
    video_py: str,
    stage3_py: str,
    attn: str,
    action_plan_path: Optional[Path],
    tts_path: Optional[str],
    video_path: Optional[str],
    validation_notes: List[str],
    errors: List[Dict[str, Any]],
    lifecycle: Dict[str, Any],
    qa: Optional[Dict[str, Any]] = None,
) -> TurnManifest:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    return TurnManifest(
        date=now,
        status=status,
        character=turn.character,
        job_id=turn.job_id,
        session_id=turn.session_id,
        input={"text": turn.text, "audio_path": input_audio},
        action_plan=plan.to_dict() if plan else {},
        artifacts={
            "asr_text": str(out_dir / "asr.txt") if input_audio else None,
            "asr_raw_text": str(out_dir / "asr_raw.txt") if input_audio else None,
            "action_plan": str(action_plan_path) if action_plan_path else None,
            "reply_text": str(out_dir / "reply.txt") if plan else None,
            "tts_wav": tts_path,
            "video_prompt": str(out_dir / "video_prompt.txt") if plan else None,
            "negative_prompt": str(out_dir / "negative_prompt.txt") if plan else None,
            "video_mp4": video_path,
        },
        timings=timings,
        models=models,
        runtime={
            "repo_root": str(root),
            "git_commit": _git_commit(root),
            "video_python": video_py,
            "stage3_python": stage3_py,
            "attn_implementation": attn,
            "gpu": _gpu_info(video_py, root),
        },
        safety=plan.safety.__dict__ if plan else {},
        validation_notes=validation_notes,
        errors=errors,
        lifecycle=lifecycle,
        qa=qa or {},
    )


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
    planner_lora_path: Optional[str] = None,
    stage: str = "execute",
    approved_action_plan_path: Optional[str] = None,
    approved_by: Optional[str] = None,
    approved_at: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    retries: int = 0,
    session_store_dir: Optional[str] = None,
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
    errors: List[Dict[str, Any]] = []
    validation_notes: List[str] = []
    action_plan_path: Optional[Path] = None
    tts_path = None
    video_path = None
    qa: Dict[str, Any] = {}
    stage = (stage or "execute").strip().lower()
    lifecycle: Dict[str, Any] = {"stage": stage}
    models = {
        "asr": whisper_model,
        "llm": llm_model,
        "tts": tts_model,
        "video": video_checkpoint,
    }
    session = _load_session(root, session_store_dir, turn.session_id)
    session_context = session.transcript_turns[-8:] if session else []

    if stage not in {"execute", "plan_only", "execute_plan"}:
        errors.append({"stage": "input", "message": f"unknown stage: {stage}"})
        manifest = _build_manifest(
            out_dir=out_dir,
            root=root,
            turn=turn,
            status="failed",
            plan=None,
            input_audio=input_audio,
            timings=timings,
            models=models,
            video_py=video_py,
            stage3_py=stage3_py,
            attn=attn,
            action_plan_path=None,
            tts_path=None,
            video_path=None,
            validation_notes=[],
            errors=errors,
            lifecycle=lifecycle,
        )
        return _write_manifest(out_dir, manifest)

    try:
        if stage == "execute_plan":
            plan_path = Path(approved_action_plan_path or out_dir / "action_plan.json")
            if not plan_path.is_absolute():
                plan_path = root / plan_path
            plan = ActionPlan.from_dict(read_json(plan_path))
            action_plan_path = plan_path
            lifecycle.update(
                {
                    "approved_action_plan": str(plan_path),
                    "approved_by": approved_by,
                    "approved_at": approved_at,
                }
            )
        else:
            if turn.audio_path:
                src = Path(turn.audio_path)
                input_audio = str(out_dir / f"input{src.suffix or '.wav'}")
                shutil.copyfile(src, input_audio)
                asr_result, asr_elapsed = run_asr(
                    python_bin=stage3_py,
                    work_dir=out_dir,
                    audio_path=input_audio,
                    model_path=whisper_model,
                    timeout_sec=timeout_sec,
                    retries=retries,
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
                session_context=session_context,
                planner_lora_path=planner_lora_path,
                timeout_sec=timeout_sec,
                retries=retries,
            )
            timings["planner_sec"] = round(planner_elapsed, 3)
            plan = ActionPlan.from_dict(plan_dict)
    except Exception as exc:
        errors.append(_error_payload("plan", exc))
        manifest = _build_manifest(
            out_dir=out_dir,
            root=root,
            turn=turn,
            status="failed",
            plan=None,
            input_audio=input_audio,
            timings=timings,
            models=models,
            video_py=video_py,
            stage3_py=stage3_py,
            attn=attn,
            action_plan_path=None,
            tts_path=None,
            video_path=None,
            validation_notes=[],
            errors=errors,
            lifecycle=lifecycle,
        )
        return _write_manifest(out_dir, manifest)

    plan.tts.enabled = plan.tts.enabled and turn.enable_tts
    plan.video.enabled = plan.video.enabled and turn.enable_video
    _ensure_prompts(plan, turn.character)
    plan = apply_policy(plan, turn.safety_mode)
    validation_notes = plan.validate(allowed_video_profiles=sorted(VIDEO_PROFILES))
    action_plan_path = _write_plan_files(out_dir, plan)

    if validation_notes:
        manifest = _build_manifest(
            out_dir=out_dir,
            root=root,
            turn=turn,
            status="failed",
            plan=plan,
            input_audio=input_audio,
            timings=timings,
            models=models,
            video_py=video_py,
            stage3_py=stage3_py,
            attn=attn,
            action_plan_path=action_plan_path,
            tts_path=None,
            video_path=None,
            validation_notes=validation_notes,
            errors=errors,
            lifecycle=lifecycle,
        )
        return _write_manifest(out_dir, manifest)

    if stage == "plan_only":
        lifecycle["requires_approval"] = True
        manifest = _build_manifest(
            out_dir=out_dir,
            root=root,
            turn=turn,
            status="pending_approval",
            plan=plan,
            input_audio=input_audio,
            timings=timings,
            models=models,
            video_py=video_py,
            stage3_py=stage3_py,
            attn=attn,
            action_plan_path=action_plan_path,
            tts_path=None,
            video_path=None,
            validation_notes=validation_notes,
            errors=errors,
            lifecycle=lifecycle,
        )
        return _write_manifest(out_dir, manifest)

    if plan.safety.allowed:
        if plan.tts.enabled:
            try:
                tts_result, tts_elapsed = run_tts(
                    python_bin=stage3_py,
                    work_dir=out_dir,
                    model_path=tts_model,
                    text=plan.reply_text,
                    speaker=plan.tts.speaker,
                    language=plan.tts.language,
                    instruct=plan.tts.instruct,
                    attn_implementation=attn,
                    timeout_sec=timeout_sec,
                    retries=retries,
                )
                timings["tts_sec"] = round(tts_elapsed, 3)
                tts_path = str(tts_result.get("wav_path") or "")
            except Exception as exc:
                errors.append(_error_payload("tts", exc))

        if plan.video.enabled:
            try:
                preset = get_character_preset(turn.character)
                video_result, video_elapsed = run_video(
                    python_bin=video_py,
                    work_dir=out_dir,
                    checkpoint_dir=video_checkpoint,
                    prompt=plan.video.positive_prompt,
                    negative_prompt=plan.video.negative_prompt,
                    profile_name=plan.video.profile,
                    seed=plan.video.seed,
                    identity_loras=list(preset.get("identity_loras") or []),
                    timeout_sec=timeout_sec,
                    retries=retries,
                )
                timings["video_sec"] = round(video_elapsed, 3)
                video_path = str(video_result.get("video_path") or "")
            except Exception as exc:
                errors.append(_error_payload("video", exc))

    qa = check_turn_artifacts(tts_wav=tts_path, video_mp4=video_path)
    if not plan.safety.allowed:
        status = "blocked"
    elif errors:
        status = "partial"
    else:
        status = "completed"

    manifest = _build_manifest(
        out_dir=out_dir,
        root=root,
        turn=turn,
        status=status,
        plan=plan,
        input_audio=input_audio,
        timings=timings,
        models=models,
        video_py=video_py,
        stage3_py=stage3_py,
        attn=attn,
        action_plan_path=action_plan_path,
        tts_path=tts_path,
        video_path=video_path,
        validation_notes=validation_notes,
        errors=errors,
        lifecycle=lifecycle,
        qa=qa,
    )
    manifest_path = out_dir / "manifest.json"
    _save_session(
        root,
        session_store_dir,
        session,
        user_text=user_text,
        reply_text=plan.reply_text,
        manifest_path=manifest_path,
        job_id=turn.job_id,
    )
    return _write_manifest(out_dir, manifest)
