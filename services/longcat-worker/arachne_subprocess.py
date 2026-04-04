"""ARACHNE-X ULTRA inference via torchrun (run_t2v.py / run_at2v.py)."""
from __future__ import annotations
import json
import os
import subprocess
from typing import Any

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default

def run_arachne_inference(job: dict[str, Any]) -> bytes:
    repo = os.environ.get("ARACHNE_VIDEO_REPO", "").strip()
    ckpt = os.environ.get("ARACHNE_CHECKPOINT_DIR", "").strip()
    if not repo or not os.path.isdir(repo):
        raise RuntimeError("ARACHNE_VIDEO_REPO must be set to the ARACHNE video repository root")
    if not ckpt or not os.path.isdir(ckpt):
        raise RuntimeError("ARACHNE_CHECKPOINT_DIR must point to downloaded ARACHNE weights")
    out_mp4 = job.get("output_mp4")
    if not out_mp4:
        raise RuntimeError("job missing output_mp4")
    out_mp4 = os.path.abspath(out_mp4)
    work_dir = os.path.dirname(out_mp4)
    task = str(job.get("task") or "text-to-video")
    script_t2v = os.environ.get("ARACHNE_SCRIPT_T2V", "run_t2v.py").strip() or "run_t2v.py"
    script_at2v = os.environ.get("ARACHNE_SCRIPT_AT2V", "run_at2v.py").strip() or "run_at2v.py"
    image_mode = os.environ.get("ARACHNE_IMAGE_TO_VIDEO_SCRIPT", "t2v").strip().lower()
    extra_cli: list[str] = []
    if task == "text-to-video":
        script_name = script_t2v
    elif task == "image-to-video":
        script_name = script_at2v if image_mode == "at2v" else script_t2v
    elif task in ("audio-text-to-video", "audio-image-to-video", "video-continuation"):
        script_name = script_at2v
        ns = job.get("num_segments")
        ri = job.get("ref_img_index")
        if ns is not None:
            extra_cli.extend(["--num_segments", str(int(ns))])
        if ri is not None:
            extra_cli.extend(["--ref_img_index", str(int(ri))])
    else:
        raise RuntimeError(f"unsupported ARACHNE task: {task}")
    script_path = os.path.join(repo, script_name)
    if not os.path.isfile(script_path):
        raise RuntimeError(f"ARACHNE script not found: {script_path}")
    out_key = os.environ.get("ARACHNE_INPUT_JSON_OUTPUT_KEY", "output_video").strip() or "output_video"
    doc: dict[str, Any] = {"prompt": job.get("prompt") or ""}
    doc[out_key] = out_mp4
    if job.get("negative_prompt"):
        doc["negative_prompt"] = job["negative_prompt"]
    if job.get("image_path"):
        doc["image_path"] = job["image_path"]
    if job.get("video_path"):
        doc["video_path"] = job["video_path"]
    if job.get("audio_path"):
        doc["audio_path"] = job["audio_path"]
    overlay = job.get("input_json_overlay")
    if isinstance(overlay, dict):
        doc.update(overlay)
    input_json_path = os.path.join(work_dir, "_nx_arachne_input.json")
    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    nproc = max(1, _int_env("ARACHNE_NPROC", 1))
    timeout = max(60, _int_env("ARACHNE_SUBPROCESS_TIMEOUT_SEC", 7200))
    torchrun = os.environ.get("ARACHNE_TORCHRUN", "torchrun").strip() or "torchrun"
    cmd = [torchrun, "--standalone", f"--nproc_per_node={nproc}", script_path, "--checkpoint_dir", ckpt, "--input_json", input_json_path, *extra_cli]
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo if not prev else repo + os.pathsep + prev
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, timeout=timeout)
    try:
        os.unlink(input_json_path)
    except OSError:
        pass
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ARACHNE torchrun failed (exit {proc.returncode})\n{err[-12000:]}\n--- stdout ---\n{out[-4000:]}")
    if not os.path.isfile(out_mp4):
        raise RuntimeError(f"Inference finished but output file is missing: {out_mp4}")
    with open(out_mp4, "rb") as f:
        return f.read()
