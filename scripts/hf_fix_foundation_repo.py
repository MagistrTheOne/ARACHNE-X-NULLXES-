#!/usr/bin/env python3
"""Fix ARACHNE-FOUNDATION-50B HF layout: README at root, DiT shards at root, remove dit/ prefix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

README = """---
license: apache-2.0
library_name: diffusers
tags:
- arachne
- nullxes
- text-to-video
- video-generation
- foundation
pipeline_tag: text-to-video
---

# ARACHNE-FOUNDATION-50B

NULLXES ARACHNE foundation video DiT checkpoint (depth 178).

**Core team:** NULLXES LLC — CEO [@MagistrTheOne](https://huggingface.co/MagistrTheOne) — ceo@nullxes.com

## Status

Surgical depth init from [ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO). Continue pretrain required before production serving.

## Layout

Safetensors shards and `config.json` live at **repository root** (Diffusers DiT layout).

## Training data

Smoke / foundation corpus: [ARACHNE-FOUNDATION-DATA-SMOKE](https://huggingface.co/MagistrTheOne/ARACHNE-FOUNDATION-DATA-SMOKE)

## Class

`ArachneVideoDiT3DModel` (legacy alias: `LongCatVideoTransformer3DModel`)
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="MagistrTheOne/ARACHNE-FOUNDATION-50B")
    p.add_argument("--dit-dir", default="/workspace/weights/ARACHNE-FOUNDATION-50B/dit")
    p.add_argument("--delete-dit-prefix", action="store_true", default=True)
    args = p.parse_args()

    dit = Path(args.dit_dir)
    if not (dit / "config.json").is_file():
        raise SystemExit(f"Missing config.json in {dit}")

    from huggingface_hub import HfApi

    api = HfApi()
    token = os.environ.get("HF_TOKEN")

    readme_path = dit.parent / "README.md"
    readme_path.write_text(README, encoding="utf-8")
    print(f"Wrote {readme_path}")

    if args.delete_dit_prefix:
        try:
            files = api.list_repo_files(args.repo, repo_type="model")
            dit_files = [f for f in files if f.startswith("dit/")]
            if dit_files:
                print(f"Deleting {len(dit_files)} files under dit/ ...")
                for f in dit_files:
                    api.delete_file(
                        path_in_repo=f,
                        repo_id=args.repo,
                        repo_type="model",
                        commit_message="Remove dit/ prefix (move to root)",
                        token=token,
                    )
        except Exception as exc:
            print(f"Warning: could not delete dit/ prefix: {exc}", file=sys.stderr)

    cmd = [
        "hf",
        "upload",
        args.repo,
        str(dit),
        ".",
        "--repo-type",
        "model",
        "--commit-message",
        "DiT weights at repo root + README model card",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cmd_readme = [
        "hf",
        "upload",
        args.repo,
        str(readme_path),
        "README.md",
        "--repo-type",
        "model",
        "--commit-message",
        "Add README model card",
    ]
    print("Running:", " ".join(cmd_readme))
    subprocess.run(cmd_readme, check=True)
    print("Done:", f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
