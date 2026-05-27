#!/usr/bin/env python3
"""
Depth surgery: ULTRA-VIDEO 13.6B DiT (depth=48) -> FOUNDATION init (~30B, depth=72).

Copies blocks 0..47 strict; blocks 48..N-1 init from block 47 (neighbor clone).
Does NOT run training. Output is init-only checkpoint for continue pretrain.

Usage:
  export PYTHONPATH=/workspace/ARACHNE-X
  python scripts/expand_dit_depth.py \\
    --src-dit /workspace/weights/ARACHNE-X-ULTRA-VIDEO/dit \\
    --out-dit /workspace/weights/ARACHNE-FOUNDATION-30B-INIT/dit \\
    --target-depth 72
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.*)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_to_dict(config) -> dict:
    """Diffusers ConfigMixin may expose FrozenDict without .to_dict()."""
    if hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
        return config.to_dict()
    return dict(config)


def _load_old_state_dict(dit_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    sys.path.insert(0, str(_repo_root()))
    from arachne_x.modules.arachne_video_dit import LongCatVideoTransformer3DModel

    model = LongCatVideoTransformer3DModel.from_pretrained(
        str(dit_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    return model.state_dict(), _config_to_dict(model.config)


def expand_depth(
    src_dit: Path,
    out_dit: Path,
    target_depth: int,
    seed_block: int = 47,
    noise_std: float = 0.0,
) -> None:
    sys.path.insert(0, str(_repo_root()))
    from arachne_x.modules.arachne_video_dit import LongCatVideoTransformer3DModel

    old_sd, old_cfg = _load_old_state_dict(src_dit)
    src_depth = int(old_cfg.get("depth", 48))
    if target_depth <= src_depth:
        raise ValueError(f"target_depth {target_depth} must be > source depth {src_depth}")
    if seed_block >= src_depth:
        print(
            f"WARNING: seed_block {seed_block} >= src_depth {src_depth}; "
            f"using {src_depth - 1} (last existing block)",
            file=sys.stderr,
        )
        seed_block = src_depth - 1

    new_cfg = dict(old_cfg)
    new_cfg["depth"] = int(target_depth)
    # ABI name unchanged so existing tooling loads; canonical rename later.
    new_cfg["_class_name"] = old_cfg.get("_class_name", "LongCatVideoTransformer3DModel")

    new_model = LongCatVideoTransformer3DModel.from_config(new_cfg)
    new_sd = new_model.state_dict()

    copied = 0
    for k, v in old_sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k] = v.clone()
            copied += 1

    cloned_blocks = 0
    for new_i in range(src_depth, target_depth):
        for k, v in list(new_sd.items()):
            m = _BLOCK_RE.match(k)
            if not m or int(m.group(1)) != new_i:
                continue
            suffix = m.group(2)
            src_key = f"blocks.{seed_block}.{suffix}"
            if src_key not in old_sd:
                continue
            src = old_sd[src_key]
            if new_sd[k].shape != src.shape:
                continue
            t = src.clone()
            if noise_std > 0:
                t = t + torch.randn_like(t) * noise_std
            new_sd[k] = t
            cloned_blocks += 1

    incompatible = new_model.load_state_dict(new_sd, strict=False)
    out_dit.mkdir(parents=True, exist_ok=True)
    new_model.save_pretrained(str(out_dit), safe_serialization=True)

    meta = {
        "status": "init_only",
        "training": "not_run",
        "source": str(src_dit),
        "source_depth": src_depth,
        "target_depth": target_depth,
        "seed_block_for_new_layers": seed_block,
        "copied_tensors": copied,
        "cloned_block_tensors": cloned_blocks,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }
    (out_dit / "ARACHNE_FOUNDATION_INIT.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Saved init DiT -> {out_dit}")
    print(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Depth surgery ULTRA-VIDEO DiT -> FOUNDATION init")
    p.add_argument("--src-dit", required=True, help="Path to ULTRA-VIDEO/dit")
    p.add_argument("--out-dit", required=True, help="Output dit/ directory")
    p.add_argument("--target-depth", type=int, default=72)
    p.add_argument("--seed-block", type=int, default=47)
    p.add_argument("--noise-std", type=float, default=0.0)
    args = p.parse_args()
    expand_depth(
        Path(args.src_dit),
        Path(args.out_dit),
        target_depth=int(args.target_depth),
        seed_block=int(args.seed_block),
        noise_std=float(args.noise_std),
    )


if __name__ == "__main__":
    main()
