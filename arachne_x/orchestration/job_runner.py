from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schemas import TurnInput, TurnManifest
from .subprocess_utils import default_python, read_json, write_json
from .turn_runner import run_turn


DEFAULT_WEIGHTS = {
    "whisper_model": "/workspace/ARACHNE-X/weights/openai-whisper-large-v3-turbo",
    "llm_model": "/workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507",
    "tts_model": "/workspace/ARACHNE-X/weights/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "video_checkpoint": "/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO",
}


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_dir(root: Path, job_id: str, character: str) -> str:
    safe_job = (job_id or _stamp()).strip().replace("/", "_").replace("\\", "_")
    safe_character = (character or "megan").strip().lower().replace(" ", "_")
    return str(root / "output" / "jobs" / f"{safe_character}_{safe_job}")


def healthcheck(
    *,
    repo_root: str | Path,
    video_python: Optional[str] = None,
    stage3_python: Optional[str] = None,
    weights: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    paths = {
        "repo_root": str(root),
        "video_python": video_python or default_python(root, ".venv"),
        "stage3_python": stage3_python or default_python(root, ".venv_stage3"),
        **(weights or DEFAULT_WEIGHTS),
    }
    checks = {name: Path(path).exists() for name, path in paths.items()}
    return {"ok": all(checks.values()), "paths": paths, "checks": checks}


def turn_from_job(job: Dict[str, Any], *, repo_root: str | Path) -> TurnInput:
    root = Path(repo_root).resolve()
    job_id = str(job.get("job_id") or job.get("id") or _stamp())
    character = str(job.get("character") or "megan")
    output_dir = str(job.get("output_dir") or job.get("out") or _default_output_dir(root, job_id, character))
    return TurnInput(
        text=job.get("text"),
        audio_path=job.get("audio_path") or job.get("audio"),
        character=character,
        output_dir=output_dir,
        safety_mode=str(job.get("safety_mode") or job.get("safety") or "prod"),
        video_profile=str(job.get("video_profile") or "fast_distill_9x16"),
        enable_tts=bool(job.get("enable_tts", not bool(job.get("no_tts", False)))),
        enable_video=bool(job.get("enable_video", not bool(job.get("no_video", False)))),
        job_id=job_id,
        session_id=job.get("session_id"),
    )


def run_job(
    job: Dict[str, Any],
    *,
    repo_root: str | Path,
    video_python: Optional[str] = None,
    stage3_python: Optional[str] = None,
) -> TurnManifest:
    turn = turn_from_job(job, repo_root=repo_root)
    weights = {**DEFAULT_WEIGHTS, **dict(job.get("weights") or {})}
    return run_turn(
        turn,
        repo_root=repo_root,
        video_python=video_python or job.get("video_python"),
        stage3_python=stage3_python or job.get("stage3_python"),
        whisper_model=str(job.get("whisper_model") or weights["whisper_model"]),
        llm_model=str(job.get("llm_model") or weights["llm_model"]),
        tts_model=str(job.get("tts_model") or weights["tts_model"]),
        video_checkpoint=str(job.get("video_checkpoint") or weights["video_checkpoint"]),
        attn_implementation=str(job.get("attn") or "auto"),
        stage=str(job.get("stage") or "execute"),
        approved_action_plan_path=job.get("approved_action_plan") or job.get("approved_action_plan_path"),
        approved_by=job.get("approved_by"),
        approved_at=job.get("approved_at"),
        timeout_sec=job.get("timeout_sec"),
        retries=int(job.get("retries") or 0),
        session_store_dir=job.get("session_store") or job.get("session_store_dir"),
    )


def run_job_file(
    job_path: str | Path,
    *,
    repo_root: str | Path,
    video_python: Optional[str] = None,
    stage3_python: Optional[str] = None,
) -> TurnManifest:
    path = Path(job_path)
    running_path = path.with_suffix(path.suffix + ".running")
    done_path = path.with_suffix(path.suffix + ".done")
    failed_path = path.with_suffix(path.suffix + ".failed")
    if done_path.exists():
        return TurnManifest(status="skipped", lifecycle={"reason": "job_already_done", "job_path": str(path)})
    running_path.write_text(dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    try:
        manifest = run_job(
            read_json(path),
            repo_root=repo_root,
            video_python=video_python,
            stage3_python=stage3_python,
        )
        marker = done_path if manifest.status not in {"failed"} else failed_path
        write_json(marker, manifest.to_dict())
        return manifest
    finally:
        if running_path.exists():
            running_path.unlink()


def pending_jobs(jobs_dir: str | Path) -> Iterable[Path]:
    root = Path(jobs_dir)
    root.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".result.json"):
            continue
        if path.with_suffix(path.suffix + ".done").exists():
            continue
        if path.with_suffix(path.suffix + ".running").exists():
            continue
        yield path


def watch_jobs(
    *,
    jobs_dir: str | Path,
    repo_root: str | Path,
    poll_sec: float = 5.0,
    once: bool = False,
    video_python: Optional[str] = None,
    stage3_python: Optional[str] = None,
) -> None:
    while True:
        for path in pending_jobs(jobs_dir):
            run_job_file(path, repo_root=repo_root, video_python=video_python, stage3_python=stage3_python)
        if once:
            return
        time.sleep(max(1.0, poll_sec))
