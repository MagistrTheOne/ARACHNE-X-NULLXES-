#!/usr/bin/env python3
"""
Depth surgery: ULTRA-VIDEO 13.6B DiT (depth=48) -> FOUNDATION init (e.g. depth=177 ~50B).

Copies blocks 0..47 strict; blocks 48..N-1 init from block 47 (neighbor clone).
Does NOT run training. Output is init-only checkpoint for continue pretrain.

Streaming safetensors I/O — never materializes the full target model in RAM.

Usage:
  export PYTHONPATH=/workspace/ARACHNE-X
  python scripts/expand_dit_depth.py \\
    --src-dit /workspace/weights/ARACHNE-X-ULTRA-VIDEO/dit \\
    --out-dit /workspace/weights/ARACHNE-FOUNDATION-50B-INIT/dit \\
    --target-depth 177
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.*)$")
_INDEX_CANDIDATES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)
_MAX_SHARD_BYTES = 5_000_000_000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_index(src_dit: Path) -> tuple[dict, str]:
    for name in _INDEX_CANDIDATES:
        path = src_dit / name
        if path.is_file():
            index = json.loads(path.read_text(encoding="utf-8"))
            return index, name
    single = src_dit / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        tensors = load_file(str(single))
        weight_map = {k: single.name for k in tensors.keys()}
        return {"metadata": {}, "weight_map": weight_map}, single.name
    raise FileNotFoundError(f"No safetensors index or single shard in {src_dit}")


def _load_config(src_dit: Path) -> dict:
    cfg_path = src_dit / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {src_dit}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _load_seed_block_tensors(
    src_dit: Path,
    weight_map: dict[str, str],
    seed_block: int,
) -> dict[str, torch.Tensor]:
    prefix = f"blocks.{seed_block}."
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            keys_by_shard[shard].append(key)

    seed: dict[str, torch.Tensor] = {}
    for shard, keys in keys_by_shard.items():
        shard_path = src_dit / shard
        data = load_file(str(shard_path))
        for key in keys:
            suffix = key[len(prefix) :]
            seed[suffix] = data[key].clone()
        del data
    if not seed:
        raise RuntimeError(f"No tensors found for seed block {seed_block}")
    return seed


def _build_output_plan(
    weight_map: dict[str, str],
    src_depth: int,
    target_depth: int,
    seed_block: int,
) -> list[tuple[str, str | None]]:
    """Return ordered (out_key, src_key) pairs. src_key=None => clone from seed block."""
    seed_prefix = f"blocks.{seed_block}."
    seed_suffixes = sorted(
        {
            key[len(seed_prefix) :]
            for key in weight_map
            if key.startswith(seed_prefix)
        }
    )
    if not seed_suffixes:
        raise RuntimeError(f"Seed block {seed_block} has no tensors in source index")

    plan: list[tuple[str, str | None]] = []
    for out_key in weight_map:
        plan.append((out_key, out_key))

    for new_i in range(src_depth, target_depth):
        for suffix in seed_suffixes:
            plan.append((f"blocks.{new_i}.{suffix}", None))

    return plan


class _ShardReader:
    def __init__(self, src_dit: Path, weight_map: dict[str, str]) -> None:
        self.src_dit = src_dit
        self.weight_map = weight_map
        self._open_shard: str | None = None
        self._open_data: dict[str, torch.Tensor] | None = None

    def get(self, src_key: str) -> torch.Tensor:
        shard = self.weight_map[src_key]
        if shard != self._open_shard:
            del self._open_data
            self._open_data = load_file(str(self.src_dit / shard))
            self._open_shard = shard
        assert self._open_data is not None
        return self._open_data[src_key]


def _stream_save_sharded(
    plan: list[tuple[str, str | None]],
    src_dit: Path,
    weight_map: dict[str, str],
    seed_tensors: dict[str, torch.Tensor],
    out_dit: Path,
    noise_std: float,
) -> tuple[int, int, dict[str, str]]:
    reader = _ShardReader(src_dit, weight_map)
    out_dit.mkdir(parents=True, exist_ok=True)

    shard_tensors: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0
    out_weight_map: dict[str, str] = {}
    copied = 0
    cloned = 0

    def flush_shard() -> None:
        nonlocal shard_idx, shard_tensors, shard_bytes
        if not shard_tensors:
            return
        shard_idx += 1
        fname = f"diffusion_pytorch_model-{shard_idx:05d}-of-00000.safetensors"
        for key in shard_tensors:
            out_weight_map[key] = fname
        save_file(shard_tensors, str(out_dit / fname))
        shard_tensors = {}
        shard_bytes = 0

    for out_key, src_key in plan:
        if src_key is None:
            m = _BLOCK_RE.match(out_key)
            if not m:
                raise RuntimeError(f"Expected block key, got {out_key}")
            suffix = m.group(2)
            if suffix not in seed_tensors:
                raise RuntimeError(f"Missing seed suffix {suffix} for {out_key}")
            tensor = seed_tensors[suffix].clone()
            if noise_std > 0:
                tensor = tensor + torch.randn_like(tensor) * noise_std
            cloned += 1
        else:
            tensor = reader.get(src_key).clone()
            copied += 1

        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if shard_bytes + nbytes > _MAX_SHARD_BYTES and shard_tensors:
            flush_shard()
        shard_tensors[out_key] = tensor
        shard_bytes += nbytes

    flush_shard()

    total_shards = shard_idx
    for fname in {v for v in out_weight_map.values()}:
        final = fname.replace("-of-00000", f"-of-{total_shards:05d}")
        (out_dit / fname).rename(out_dit / final)
        for key, mapped in list(out_weight_map.items()):
            if mapped == fname:
                out_weight_map[key] = final

    return copied, cloned, out_weight_map


def expand_depth(
    src_dit: Path,
    out_dit: Path,
    target_depth: int,
    seed_block: int = 47,
    noise_std: float = 0.0,
) -> None:
    old_cfg = _load_config(src_dit)
    index, index_name = _load_index(src_dit)
    weight_map: dict[str, str] = index["weight_map"]

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

    print(
        f"Streaming depth surgery: {src_depth} -> {target_depth}, seed_block={seed_block}",
        flush=True,
    )
    seed_tensors = _load_seed_block_tensors(src_dit, weight_map, seed_block)
    plan = _build_output_plan(weight_map, src_depth, target_depth, seed_block)

    new_cfg = dict(old_cfg)
    new_cfg["depth"] = int(target_depth)
    out_dit.mkdir(parents=True, exist_ok=True)
    (out_dit / "config.json").write_text(json.dumps(new_cfg, indent=2), encoding="utf-8")

    copied, cloned, out_weight_map = _stream_save_sharded(
        plan,
        src_dit,
        weight_map,
        seed_tensors,
        out_dit,
        noise_std,
    )
    del seed_tensors

    total_size = sum((out_dit / fname).stat().st_size for fname in set(out_weight_map.values()))
    out_index = {
        "metadata": {"total_size": total_size},
        "weight_map": out_weight_map,
    }
    (out_dit / index_name).write_text(json.dumps(out_index, indent=2), encoding="utf-8")

    meta = {
        "status": "init_only",
        "training": "not_run",
        "mode": "stream_safetensors",
        "source": str(src_dit),
        "source_depth": src_depth,
        "target_depth": target_depth,
        "seed_block_for_new_layers": seed_block,
        "copied_tensors": copied,
        "cloned_block_tensors": cloned,
        "output_shards": len(set(out_weight_map.values())),
        "total_size_bytes": total_size,
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
