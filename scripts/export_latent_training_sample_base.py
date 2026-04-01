"""
Export a single base (T2V) training sample (.pt) for ``train.py --mode base``.

Video → VAE latents → flow-matching noisy input + epsilon target; text via T5 (no audio).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.loader import load_base_pipeline
from arachne_x.training_latent_export_base import export_base_latent_training_pt
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Export one base DiT training .pt from a video clip")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--video", type=str, required=True, help="Path to a video file (e.g. Mira clip)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--output", type=str, default="sample_base.pt")
    parser.add_argument("--seed", type=int, default=None, help="RNG for VAE latent draw, noise, and timestep index")
    parser.add_argument(
        "--vae_sample_mode",
        type=str,
        default="argmax",
        choices=["sample", "argmax"],
        help="VAE latent draw: argmax (deterministic) or sample",
    )
    add_resolve_args(parser)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    root = resolve_weights_root(
        args.checkpoint_dir,
        allow_hub=args.allow_hub_download,
        cache_dir=args.weights_cache_dir,
    )
    pipe = load_base_pipeline(root, device=device, torch_dtype=dtype)

    export_base_latent_training_pt(
        pipe,
        video_path=args.video,
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        resolution=args.resolution,
        num_frames=args.num_frames,
        seed=args.seed,
        device=device,
        vae_sample_mode=args.vae_sample_mode,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
