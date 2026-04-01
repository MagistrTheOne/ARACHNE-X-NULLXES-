"""
Batch-export base training .pt files from MiraData-style metadata (HF or CSV).

Requires local video clips: join ``video_root`` with ``file_path`` from each row.
Install ``pip install datasets`` when using ``--from_hf``.

Example (after downloading clips per Mira README):

  python scripts/export_miradata_latents.py \\
    --checkpoint_dir /path/to/LongCat-Video \\
    --from_hf TencentARC/MiraData \\
    --split train \\
    --video_root /data/mira/clips \\
    --output_dir /data/mira_latents_pt \\
    --max_samples 1000 \\
    --caption_field dense_caption
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.loader import load_base_pipeline
from arachne_x.training_latent_export_base import export_base_latent_training_pt
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root


def _iter_rows_from_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _iter_rows_from_hf(name: str, split: str, streaming: bool):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install the `datasets` package: pip install datasets") from e
    ds = load_dataset(name, split=split, streaming=streaming)
    for row in ds:
        yield row


def main():
    parser = argparse.ArgumentParser(
        description="Export base latent .pt samples from MiraData metadata + local videos",
        epilog="HF mode needs: pip install datasets",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_hf", type=str, default=None, help="Dataset id, e.g. TencentARC/MiraData")
    src.add_argument("--from_csv", type=str, default=None, help="Meta CSV with file_path and caption columns")
    parser.add_argument("--split", type=str, default="train", help="HF split name")
    parser.add_argument("--streaming", action="store_true", help="Stream HF dataset (slower export, low RAM)")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory for clip files (joined with file_path)")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="Stop after N successful exports (0 = no limit)")
    parser.add_argument(
        "--caption_field",
        type=str,
        default="dense_caption",
        choices=[
            "dense_caption",
            "short_caption",
            "background_caption",
            "main_object_caption",
            "style_caption",
            "camera_caption",
        ],
        help="Column / field used as text prompt",
    )
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--seed_base", type=int, default=0, help="Per-sample seed = seed_base + index")
    parser.add_argument("--skip_errors", action="store_true", help="Log failures and continue")
    parser.add_argument(
        "--vae_sample_mode",
        type=str,
        default="argmax",
        choices=["sample", "argmax"],
    )
    add_resolve_args(parser)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    root = resolve_weights_root(
        args.checkpoint_dir,
        allow_hub=args.allow_hub_download,
        cache_dir=args.weights_cache_dir,
    )
    pipe = load_base_pipeline(root, device=device, torch_dtype=dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.from_hf:
        row_iter = _iter_rows_from_hf(args.from_hf, args.split, args.streaming)
    else:
        row_iter = _iter_rows_from_csv(args.from_csv)

    done = 0
    idx = 0
    for row in row_iter:
        if args.max_samples and done >= args.max_samples:
            break
        fp = row.get("file_path")
        if not fp:
            idx += 1
            continue
        cap = row.get(args.caption_field) or ""
        if isinstance(cap, str):
            prompt = cap.strip()
        else:
            prompt = str(cap).strip()
        if not prompt:
            prompt = " "

        video_path = fp if os.path.isabs(fp) else os.path.normpath(os.path.join(args.video_root, fp))
        out_name = f"{idx:08d}.pt"
        out_path = os.path.join(args.output_dir, out_name)

        try:
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"missing video: {video_path}")
            export_base_latent_training_pt(
                pipe,
                video_path=video_path,
                prompt=prompt,
                output_path=out_path,
                resolution=args.resolution,
                num_frames=args.num_frames,
                seed=args.seed_base + idx,
                device=device,
                vae_sample_mode=args.vae_sample_mode,  # type: ignore[arg-type]
            )
            done += 1
        except Exception as e:
            if args.skip_errors:
                print(f"[skip idx={idx}] {e}")
            else:
                raise
        idx += 1

    print(f"Finished: exported {done} samples under {args.output_dir}")


if __name__ == "__main__":
    main()
