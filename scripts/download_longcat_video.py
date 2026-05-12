#!/usr/bin/env python3
"""
Скачивание base video weights с Hugging Face (HF repo `meituan-longcat/LongCat-Video`) для ARACHNE-X / NULLXES training.

Имя файла скрипта — историческое; логика — generic snapshot_download upstream bundle.

Использование:
  set HF_TOKEN=your_token_here   # Windows
  export HF_TOKEN=your_token_here # Linux/macOS
  python scripts/download_longcat_video.py

Или передать токен через аргумент (НЕ коммитить скрипт с токеном в команде):
  python scripts/download_longcat_video.py --token hf_xxxx

Скачается полный репозиторий в ./weights/LongCat-Video (в т.ч. dit/, vae/, tokenizer/, ...).
Для обучения base DiT: --checkpoint_dir ./weights/LongCat-Video (train.py возьмёт subfolder="dit").
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Download base video HF bundle for ARACHNE-X (meituan-longcat/LongCat-Video)")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "weights", "LongCat-Video"),
        help="Local directory to save the model (default: ./weights/LongCat-Video)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="meituan-longcat/LongCat-Video",
        help="Hugging Face repo id",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    token = args.token.strip()
    if not token:
        print(
            "Error: No Hugging Face token. Set HF_TOKEN or pass --token (do not commit tokens).",
            file=sys.stderr,
        )
        sys.exit(1)

    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {args.repo} -> {local_dir}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token if token else None,
    )
    print("Done.")
    print(f"  dit/ is at: {os.path.join(local_dir, 'dit')}")
    print("  For training (base): --checkpoint_dir", local_dir)


if __name__ == "__main__":
    main()
