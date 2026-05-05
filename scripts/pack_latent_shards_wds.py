#!/usr/bin/env python3
"""
Pack flat ``*.pt`` latent training files into WebDataset ``.tar`` shards.

Each member record:
  __key__     — stem of source filename
  sample.pt   — raw bytes of the original file (same as ``torch.load`` on disk)

Training: pass ``--wds_shards /out/shard-{000000..000009}.tar`` to ``scripts/train.py``.

Usage:
  python scripts/pack_latent_shards_wds.py \\
    --input_dir /data/latents_flat \\
    --output_dir /data/wds_shards \\
    --samples_per_shard 10000 \\
    --prefix shard
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import webdataset as wds
except ImportError as e:
    sys.stderr.write("Install webdataset: pip install webdataset\n")
    raise SystemExit(1) from e


def main():
    p = argparse.ArgumentParser(description="Pack ARACHNE-X .pt latents into WebDataset tar shards")
    p.add_argument("--input_dir", type=str, required=True, help="Directory with *.pt (LatentDataset format)")
    p.add_argument("--output_dir", type=str, required=True, help="Where to write shard-*.tar")
    p.add_argument("--samples_per_shard", type=int, default=10000)
    p.add_argument("--prefix", type=str, default="shard")
    p.add_argument("--max_shards", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    ind = os.path.abspath(args.input_dir)
    outd = os.path.abspath(args.output_dir)
    os.makedirs(outd, exist_ok=True)

    files = sorted(glob(os.path.join(ind, "*.pt")))
    if not files:
        sys.stderr.write(f"No .pt files in {ind}\n")
        sys.exit(2)

    n_per = max(1, args.samples_per_shard)
    num_shard_files = 0
    written = 0
    sink = None

    try:
        for i, fp in enumerate(files):
            if i % n_per == 0:
                if sink is not None:
                    sink.close()
                    sink = None
                if args.max_shards and num_shard_files >= args.max_shards:
                    break
                tar_path = os.path.join(outd, f"{args.prefix}_{num_shard_files:06d}.tar")
                print(f"[pack] opening {tar_path}")
                sink = wds.TarWriter(tar_path)
                num_shard_files += 1

            key = Path(fp).stem
            with open(fp, "rb") as f:
                blob = f.read()
            sink.write({"__key__": key, "sample.pt": blob})
            written += 1
    finally:
        if sink is not None:
            sink.close()

    last = max(0, num_shard_files - 1)
    print(f"[pack] wrote {written} samples into {num_shard_files} shard(s) under {outd}")
    print(f"[pack] train with: --wds_shards \"{outd}/{args.prefix}-{{000000..{last:06d}}}.tar\"")


if __name__ == "__main__":
    main()
