"""
Export a single avatar training sample (.pt) compatible with LatentDataset / train.py.

Uses the same encode path as inference (prepare_latents, encode_prompt, windowed audio)
and flow-matching ``scale_noise`` so ``latents`` is a noisy input at timestep ``t``
and ``noise`` is the epsilon used inside ``scale_noise`` (regression target for MSE training).

Does not cover identity tokens or classifier-free doubled batches; use for pipelines
that train with a single conditioning branch (``do_classifier_free_guidance=False`` style tensors).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from diffusers.utils import load_image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.loader import load_avatar_pipeline
from arachne_x.training_latent_export import export_avatar_latent_training_pt
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Export one avatar training .pt sample")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument(
        "--prompt_compiler",
        type=str,
        default=None,
        choices=["off"],
        help="Apply deterministic prompt template merge before encode_prompt (train ≡ infer).",
    )
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--output", type=str, default="sample.pt")
    parser.add_argument("--seed", type=int, default=None, help="RNG for latent init and noise")
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
    pipe = load_avatar_pipeline(root, variant="single", device=device, torch_dtype=dtype)

    image = load_image(args.image)
    export_avatar_latent_training_pt(
        pipe,
        image=image,
        audio_path=args.audio,
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        prompt_compiler=args.prompt_compiler,
        image_path=args.image,
        resolution=args.resolution,
        num_frames=args.num_frames,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()
