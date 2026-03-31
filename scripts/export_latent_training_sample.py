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
import os
import sys
from pathlib import Path

import torch
from diffusers.utils import load_image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.inference_audio import build_avatar_windowed_audio_emb
from arachne_x.loader import load_avatar_pipeline
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Export one avatar training .pt sample")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
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

    scale_factor_spatial = pipe.vae_scale_factor_spatial * 2
    if pipe.dit.cp_split_hw is not None:
        scale_factor_spatial *= max(pipe.dit.cp_split_hw)

    image = load_image(args.image)
    height, width = pipe.get_condition_shape(image, args.resolution, scale_factor_spatial=scale_factor_spatial)
    pipe.check_inputs(args.prompt, args.negative_prompt, height, width, scale_factor_spatial)

    nf = args.num_frames
    if nf % pipe.vae_scale_factor_temporal != 1:
        nf = nf // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        print(f"Adjusted num_frames to {nf}")

    dit_dtype = pipe.dit.dtype
    prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        dtype=dit_dtype,
        device=device,
    )

    img_t = pipe.video_processor.preprocess(image, height=height, width=width, resize_mode="crop")
    img_t = img_t.to(device=device, dtype=prompt_embeds.dtype)

    gen = torch.Generator(device=device)
    if args.seed is not None:
        gen.manual_seed(int(args.seed))

    z0 = pipe.prepare_latents(
        image=img_t,
        batch_size=1,
        num_channels_latents=pipe.dit.config.in_channels,
        height=height,
        width=width,
        num_frames=nf,
        num_cond_frames=1,
        dtype=torch.float32,
        device=device,
        generator=gen,
    )

    eps = torch.randn(z0.shape, device=z0.device, dtype=torch.float32, generator=gen)
    sched = pipe.scheduler
    n_sched = int(sched.timesteps.shape[0])
    idx_gen = torch.Generator()
    if args.seed is not None:
        idx_gen.manual_seed(int(args.seed) + 1)
    idx = int(torch.randint(0, n_sched, (1,), generator=idx_gen).item())
    t = sched.timesteps[idx].view(1).to(device=device, dtype=torch.float32)
    noisy = sched.scale_noise(z0, t, eps)

    wav = build_avatar_windowed_audio_emb(pipe, args.audio, nf, device)
    audio_embs = pipe._prepare_audio_emb_for_dit(
        wav,
        num_frames=nf,
        batch_size=1,
        num_videos_per_prompt=1,
        device=device,
    )

    sample = {
        "latents": noisy.cpu().to(torch.float32),
        "noise": eps.cpu().to(torch.float32),
        "timesteps": t.cpu().to(torch.float32),
        "prompt_embeds": prompt_embeds.cpu().to(torch.float32),
        "prompt_mask": prompt_attention_mask.cpu(),
        "audio_embs": audio_embs.cpu().to(torch.float32),
    }
    out_abs = os.path.abspath(args.output)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(sample, out_abs)
    print(f"Saved {out_abs} keys={list(sample.keys())} latent_shape={tuple(sample['latents'].shape)}")


if __name__ == "__main__":
    main()
