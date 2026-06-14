#!/usr/bin/env python3
"""
Pre-flight checks for ARACHNE-X-ULTRA-V3 NIGHTCORE pods (no DiT load).

Usage:
  python scripts/prod_doctor.py --checkpoint-dir /path/to/weights
  NULLXES_PRODUCTION=1 python scripts/prod_doctor.py --role worker
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _check_checkpoint_layout(ckpt: Path) -> List[str]:
    errors: List[str] = []
    required = (
        "tokenizer",
        "text_encoder",
        "vae",
        "scheduler",
        "avatar_single",
    )
    for sub in required:
        p = ckpt / sub
        if not p.is_dir():
            errors.append(f"missing checkpoint subdir: {sub}")
    audio_candidates = [
        ckpt / "audio" / "wav2vec2",
        ckpt / "chinese-wav2vec2-base",
    ]
    if not any((c / "config.json").is_file() for c in audio_candidates):
        errors.append("missing wav2vec weights (audio/wav2vec2 or chinese-wav2vec2-base)")
    return errors


def _check_cuda_probe() -> List[str]:
    errors: List[str] = []
    try:
        import torch
    except ImportError:
        errors.append("torch not installed")
        return errors
    if not torch.cuda.is_available():
        errors.append("CUDA not available (expected on GPU pod)")
        return errors
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / (1024**3)
    print(f"cuda:0 {props.name} total_memory={gb:.1f} GiB")
    if gb <= 45:
        print("note: <=45GB tier — full avatar infer blocked; train/LoRA only")
    elif gb <= 85:
        print("note: ~80GB tier — use operational profile @ 480p")
    else:
        print("note: >85GB tier — cinematic 720p available with explicit profile")
    return errors


def _check_prod_env(role: str) -> List[str]:
    if os.environ.get("NULLXES_PRODUCTION", "").strip().lower() not in ("1", "true", "yes", "on"):
        print("NULLXES_PRODUCTION not set — skipping strict prod env checks")
        return []
    sys.path.insert(0, str(_repo_root()))
    from arachne_x.runtime.prod_guard import validate_production_boot

    try:
        validate_production_boot(role=role)  # type: ignore[arg-type]
    except RuntimeError as e:
        return [str(e)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="ARACHNE-X production doctor (no DiT load)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("NULLXES_CHECKPOINT_DIR") or os.environ.get("ARACHNE_CHECKPOINT_DIR"),
    )
    parser.add_argument("--role", choices=("worker", "orchestrator", "any"), default="any")
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()

    errors: List[str] = []

    if args.checkpoint_dir:
        ckpt = Path(args.checkpoint_dir)
        if not ckpt.is_dir():
            errors.append(f"checkpoint dir not found: {ckpt}")
        else:
            errors.extend(_check_checkpoint_layout(ckpt))
    else:
        print("warning: no --checkpoint-dir / NULLXES_CHECKPOINT_DIR — layout check skipped")

    errors.extend(_check_prod_env(args.role))

    tts = (os.environ.get("NULLXES_TTS_PROVIDER") or os.environ.get("TTS_PROVIDER") or "").strip().lower()
    if tts in ("stub", "none", ""):
        if os.environ.get("NULLXES_PRODUCTION", "").strip().lower() in ("1", "true", "yes", "on"):
            errors.append("TTS provider must not be stub/none in production")

    if os.environ.get("ARACHNE_AUDIO_ENCODER", "").strip().lower() == "nullxes":
        errors.append("ARACHNE_AUDIO_ENCODER=nullxes is not supported until weights ship (use wav2vec)")

    if not args.skip_cuda:
        errors.extend(_check_cuda_probe())

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if sys.platform == "linux":
            print("warning: flash_attn import failed (Linux prod expects flash-attn wheel)")

    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("OK: prod_doctor checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
