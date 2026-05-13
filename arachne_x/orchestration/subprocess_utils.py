from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class SubprocessRunError(RuntimeError):
    def __init__(
        self,
        *,
        name: str,
        message: str,
        elapsed_sec: float,
        returncode: Optional[int] = None,
        stdout_path: Optional[str] = None,
        stderr_path: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.elapsed_sec = elapsed_sec
        self.returncode = returncode
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "message": str(self),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "returncode": self.returncode,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_python_script(
    *,
    python_bin: str,
    script_text: str,
    config: Dict[str, Any],
    work_dir: str | Path,
    name: str,
    timeout_sec: Optional[float] = None,
    retries: int = 0,
    retry_delay_sec: float = 2.0,
) -> Tuple[Dict[str, Any], float]:
    work = Path(work_dir)
    scripts = work / "_scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    script_path = scripts / f"{name}.py"
    config_path = scripts / f"{name}.json"
    result_path = scripts / f"{name}.result.json"

    script_path.write_text(textwrap.dedent(script_text).strip() + "\n", encoding="utf-8")
    write_json(config_path, {**config, "result_path": str(result_path)})

    env = os.environ.copy()
    repo = str(Path.cwd())
    env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    stdout_path = scripts / f"{name}.stdout.txt"
    stderr_path = scripts / f"{name}.stderr.txt"
    attempts = max(1, int(retries) + 1)
    started = time.perf_counter()
    last_error: Optional[SubprocessRunError] = None
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            completed = subprocess.run(
                [python_bin, str(script_path), str(config_path)],
                cwd=repo,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            elapsed = time.perf_counter() - t0
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode == 0:
                result = read_json(result_path)
                return result, time.perf_counter() - started
            last_error = SubprocessRunError(
                name=name,
                message=f"{name} subprocess failed on attempt {attempt}/{attempts}",
                elapsed_sec=elapsed,
                returncode=completed.returncode,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - t0
            stdout_path.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
            stderr_path.write_text((exc.stderr or "") if isinstance(exc.stderr, str) else "", encoding="utf-8")
            last_error = SubprocessRunError(
                name=name,
                message=f"{name} subprocess timed out after {timeout_sec} seconds on attempt {attempt}/{attempts}",
                elapsed_sec=elapsed,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        if attempt < attempts:
            time.sleep(max(0.0, retry_delay_sec))
    if last_error is not None:
        raise last_error
    elapsed = time.perf_counter() - started
    raise SubprocessRunError(name=name, message=f"{name} subprocess failed", elapsed_sec=elapsed)


def default_python(repo_root: str | Path, venv_name: str) -> str:
    root = Path(repo_root)
    posix_candidate = root / venv_name / "bin" / "python"
    win_candidate = root / venv_name / "Scripts" / "python.exe"
    if posix_candidate.exists():
        return str(posix_candidate)
    if win_candidate.exists():
        return str(win_candidate)
    return str(Path(venv_name) / "bin" / "python")
