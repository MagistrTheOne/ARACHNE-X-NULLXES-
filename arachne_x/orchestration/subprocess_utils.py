from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Tuple


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

    t0 = time.perf_counter()
    subprocess.run([python_bin, str(script_path), str(config_path)], cwd=repo, env=env, check=True)
    elapsed = time.perf_counter() - t0
    result = read_json(result_path)
    return result, elapsed


def default_python(repo_root: str | Path, venv_name: str) -> str:
    root = Path(repo_root)
    posix_candidate = root / venv_name / "bin" / "python"
    win_candidate = root / venv_name / "Scripts" / "python.exe"
    if posix_candidate.exists():
        return str(posix_candidate)
    if win_candidate.exists():
        return str(win_candidate)
    return str(Path(venv_name) / "bin" / "python")
