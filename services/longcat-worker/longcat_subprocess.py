"""
Run longcat_generate_once.py via torchrun against a cloned LongCat-Video repo.

Required env:
  LONGCAT_VIDEO_REPO   — root of https://github.com/meituan-longcat/LongCat-Video (contains longcat_video/)
  LONGCAT_CHECKPOINT_DIR — e.g. ./weights/LongCat-Video

Optional:
  LONGCAT_NPROC (default 1), LONGCAT_CONTEXT_PARALLEL_SIZE (default same as NPROC),
  LONGCAT_ENABLE_COMPILE (0/1), LONGCAT_TORCHRUN (default torchrun),
  LONGCAT_SUBPROCESS_TIMEOUT_SEC (default 7200)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
GENERATE_SCRIPT = THIS_DIR / "longcat_generate_once.py"


def run_longcat_inference(job: dict) -> bytes:
    """
    Execute GPU inference. ``job`` must include absolute ``output_mp4`` and task-specific paths.
    Caller owns the directory containing ``output_mp4`` and should remove it after this returns.
    """
    repo = os.environ.get("LONGCAT_VIDEO_REPO", "").strip()
    ckpt = os.environ.get("LONGCAT_CHECKPOINT_DIR", "").strip()
    if not repo or not os.path.isdir(repo):
        raise RuntimeError("LONGCAT_VIDEO_REPO must be set to the LongCat-Video repository root")
    if not ckpt or not os.path.isdir(ckpt):
        raise RuntimeError("LONGCAT_CHECKPOINT_DIR must point to downloaded model weights")

    out_mp4 = job.get("output_mp4")
    if not out_mp4:
        raise RuntimeError("job missing output_mp4")
    out_mp4 = os.path.abspath(out_mp4)
    work_dir = os.path.dirname(out_mp4)
    job_path = os.path.join(work_dir, "_nx_longcat_job.json")

    payload = dict(job)
    payload["output_mp4"] = out_mp4
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    nproc = int(os.environ.get("LONGCAT_NPROC", "1"))
    cpp = int(os.environ.get("LONGCAT_CONTEXT_PARALLEL_SIZE", str(nproc)))
    timeout = int(os.environ.get("LONGCAT_SUBPROCESS_TIMEOUT_SEC", "7200"))
    torchrun = os.environ.get("LONGCAT_TORCHRUN", "torchrun")
    compile_ok = os.environ.get("LONGCAT_ENABLE_COMPILE", "").lower() in ("1", "true", "yes")

    cmd = [
        torchrun,
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(GENERATE_SCRIPT),
        "--checkpoint_dir",
        ckpt,
        "--job_json",
        job_path,
        "--context_parallel_size",
        str(cpp),
    ]
    if compile_ok:
        cmd.append("--enable_compile")

    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo if not prev else repo + os.pathsep + prev

    proc = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        capture_output=True,
        timeout=timeout,
    )
    try:
        os.unlink(job_path)
    except OSError:
        pass

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LongCat torchrun failed (exit {proc.returncode})\n"
            f"{err[-12000:]}\n--- stdout ---\n{out[-4000:]}"
        )

    if not os.path.isfile(out_mp4):
        raise RuntimeError(f"Inference finished but output file is missing: {out_mp4}")

    with open(out_mp4, "rb") as f:
        return f.read()
