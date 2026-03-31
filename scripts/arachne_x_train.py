#!/usr/bin/env python3
"""
ARACHNE-X pod training launcher — один запуск на машине с GPU.

Читает переменные окружения (удобно в RunPod / Docker), а аргументы CLI имеют приоритет.

Обязательно:
  ARACHNE_CHECKPOINT_DIR — корень весов (локальный путь или org/repo при --allow-hub-download)
  ARACHNE_DATASET_DIR    — папка с .pt/.npz под LatentDataset

Часто нужно:
  ARACHNE_TRAIN_MODE     — base | avatar (default: base)
  ARACHNE_OUTPUT_DIR     — куда писать чекпоинты (default: ./outputs_train)

Опционально:
  ARACHNE_CONFIG         — JSON/YAML конфиг H200TrainingConfig
  ARACHNE_BATCH_SIZE, ARACHNE_LR, ARACHNE_MAX_STEPS, ARACHNE_SAVE_EVERY
  ARACHNE_ALLOW_HUB_DOWNLOAD=1 — скачать веса с Hub
  ARACHNE_WEIGHTS_CACHE_DIR
  ARACHNE_MERGE_INTO    — после обучения скопировать output/final/* в $DIR/dit или .../avatar_single
  ARACHNE_REPO_ROOT     — корень репо (default: родитель scripts/)

Пример (bash на поде):

  export PYTHONPATH=/workspace/ARACHNE-X
  cd /workspace/ARACHNE-X
  export ARACHNE_CHECKPOINT_DIR=/workspace/weights/full
  export ARACHNE_DATASET_DIR=/workspace/data/latents
  export ARACHNE_TRAIN_MODE=avatar
  export ARACHNE_MAX_STEPS=10000
  export ARACHNE_MERGE_INTO=/workspace/weights/production
  python scripts/arachne_x_train.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int | None = None) -> int | None:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return int(v)


def _env_float(key: str, default: float | None = None) -> float | None:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return float(v)


def _repo_root() -> Path:
    raw = os.environ.get("ARACHNE_REPO_ROOT")
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parents[1]


def _install_final_into(*, final_dir: Path, merge_into: Path, mode: str) -> None:
    sub = "dit" if mode == "base" else "avatar_single"
    dest = merge_into / sub
    dest.mkdir(parents=True, exist_ok=True)
    if not final_dir.is_dir():
        raise FileNotFoundError(f"Training final dir missing: {final_dir}")
    for item in final_dir.iterdir():
        target = dest / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
    print(f"[arachne_x_train] merged {final_dir} -> {dest}")


def main() -> None:
    root = _repo_root()
    train_py = root / "scripts" / "train.py"
    if not train_py.is_file():
        sys.stderr.write(f"Not found: {train_py}\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Launch scripts/train.py using env defaults for pod runs.",
    )
    parser.add_argument("--checkpoint-dir", default=os.environ.get("ARACHNE_CHECKPOINT_DIR"))
    parser.add_argument("--dataset-dir", default=os.environ.get("ARACHNE_DATASET_DIR"))
    parser.add_argument("--output-dir", default=os.environ.get("ARACHNE_OUTPUT_DIR", "./outputs_train"))
    parser.add_argument("--mode", choices=["base", "avatar"], default=os.environ.get("ARACHNE_TRAIN_MODE", "base"))
    parser.add_argument("--config", default=os.environ.get("ARACHNE_CONFIG"))
    parser.add_argument("--batch-size", type=int, default=_env_int("ARACHNE_BATCH_SIZE", 1))
    parser.add_argument("--lr", type=float, default=_env_float("ARACHNE_LR", 1e-4))
    parser.add_argument("--max-steps", type=int, default=_env_int("ARACHNE_MAX_STEPS", 1000))
    parser.add_argument("--save-every", type=int, default=_env_int("ARACHNE_SAVE_EVERY", 500))
    parser.add_argument(
        "--allow-hub-download",
        action="store_true",
        default=_env_bool("ARACHNE_ALLOW_HUB_DOWNLOAD", False),
        help="Also set ARACHNE_ALLOW_HUB_DOWNLOAD=1",
    )
    parser.add_argument("--weights-cache-dir", default=os.environ.get("ARACHNE_WEIGHTS_CACHE_DIR"))
    parser.add_argument(
        "--merge-into",
        default=os.environ.get("ARACHNE_MERGE_INTO"),
        help="After success: copy output_dir/final into this root's dit/ or avatar_single/",
    )
    args = parser.parse_args()

    if not args.checkpoint_dir or not str(args.checkpoint_dir).strip():
        sys.stderr.write("Set ARACHNE_CHECKPOINT_DIR or pass --checkpoint-dir\n")
        sys.exit(2)
    if not args.dataset_dir or not str(args.dataset_dir).strip():
        sys.stderr.write("Set ARACHNE_DATASET_DIR or pass --dataset-dir\n")
        sys.exit(2)

    cmd: list[str] = [
        sys.executable,
        str(train_py),
        "--mode",
        args.mode,
        "--checkpoint_dir",
        str(args.checkpoint_dir),
        "--dataset_dir",
        str(args.dataset_dir),
        "--output_dir",
        str(args.output_dir),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--max_steps",
        str(args.max_steps),
        "--save_every",
        str(args.save_every),
    ]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.allow_hub_download:
        cmd.append("--allow_hub_download")
    if args.weights_cache_dir:
        cmd.extend(["--weights_cache_dir", str(args.weights_cache_dir)])

    env = os.environ.copy()
    if str(root) not in env.get("PYTHONPATH", ""):
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(root) if not prev else f"{root}{os.pathsep}{prev}"

    print("[arachne_x_train] repo_root=", root)
    print("[arachne_x_train] cmd=", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    if args.merge_into:
        out_final = Path(args.output_dir).resolve() / "final"
        _install_final_into(final_dir=out_final, merge_into=Path(args.merge_into).resolve(), mode=args.mode)


if __name__ == "__main__":
    main()
