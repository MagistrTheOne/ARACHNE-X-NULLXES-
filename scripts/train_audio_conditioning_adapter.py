#!/usr/bin/env python3
"""
Train audio-conditioning adapter on frozen base VIDEO DiT (experimental).

Usage (synthetic smoke, no dataset):
  python scripts/train_audio_conditioning_adapter.py --synthetic --steps 2 --output output/audio_i2v_adapter.safetensors

Usage (manifest + checkpoint):
  python scripts/train_audio_conditioning_adapter.py \\
    --checkpoint_dir weights/ARACHNE-X-ULTRA-VIDEO \\
    --manifest assets/training/audio_i2v_pairs.example.json \\
    --output output/audio_i2v_adapter.safetensors \\
    --steps 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.modules.audio_conditioning.adapter import AudioConditioningAdapter, AudioConditioningAdapterConfig
from arachne_x.modules.audio_conditioning.state_dict import save_audio_conditioning_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train audio-conditioning adapter (frozen VIDEO DiT)")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None, help="JSON manifest (assets/training/audio_i2v_pairs.example.json)")
    parser.add_argument("--output", type=str, default="output/audio_i2v_adapter.safetensors")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--synthetic", action="store_true", help="Run adapter-only synthetic backward smoke")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block_stride", type=int, default=2, help="Inject every N blocks from index 24")
    return parser.parse_args()


def _synthetic_step(adapter: AudioConditioningAdapter, device: str) -> torch.Tensor:
    b, t, w, s, c = 1, 13, 5, 12, 768
    audio = torch.randn(b, t, w, s, c, device=device)
    projected = adapter.project_audio_embs(audio)
    return projected.mean()


def _load_manifest(path: str) -> list:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("pairs", []))


def main() -> None:
    args = parse_args()
    device = args.device

    block_indices = tuple(range(24, 48, max(1, args.block_stride)))
    config = AudioConditioningAdapterConfig(block_indices=block_indices)
    adapter = AudioConditioningAdapter(config=config).to(device)

    if args.synthetic or not args.checkpoint_dir:
        print("[train-audio-i2v] synthetic adapter-only smoke")
        opt = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=args.lr)
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            loss = _synthetic_step(adapter, device)
            loss.backward()
            opt.step()
            print(f"step={step+1}/{args.steps} loss={float(loss.detach().cpu()):.6f}")
        save_audio_conditioning_adapter(adapter, args.output, metadata={"train_mode": "synthetic"})
        print(f"saved {args.output}")
        return

    if not args.manifest:
        raise SystemExit("--manifest required when training on real checkpoint")

    from arachne_x.loader import load_audio_i2v_pipeline

    pairs = _load_manifest(args.manifest)
    if not pairs:
        raise SystemExit("manifest has no pairs")

    pipe = load_audio_i2v_pipeline(args.checkpoint_dir, device=device)
    pipe.audio_adapter = adapter
    pipe._refresh_dit_wrapper()

    for param in pipe.dit.parameters():
        param.requires_grad = False
    pipe.vae.eval()
    pipe.text_encoder.eval()

    opt = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=args.lr)
    print(f"[train-audio-i2v] pairs={len(pairs)} trainable_params={adapter.trainable_parameter_count()}")

    for step in range(args.steps):
        pair = pairs[step % len(pairs)]
        opt.zero_grad(set_to_none=True)

        from diffusers.utils import load_image

        image = load_image(pair["image"])
        audio_path = pair["audio"]
        num_frames = int(pair.get("num_frames", 49))

        audio_emb = pipe.build_audio_emb_from_path(audio_path, num_frames)
        projected = adapter.project_audio_embs(audio_emb.to(device))

        # Proxy alignment loss: encourage non-zero but bounded adapter response (bootstrap before latent export).
        target = projected.detach().mean()
        loss = F.mse_loss(projected.mean(), target + 0.01 * torch.randn((), device=device))
        loss.backward()
        opt.step()
        print(f"step={step+1}/{args.steps} pair={pair.get('id', step)} proxy_loss={float(loss.detach().cpu()):.6f}")

    save_audio_conditioning_adapter(
        adapter,
        args.output,
        metadata={"train_mode": "manifest_proxy", "manifest": args.manifest, "steps": args.steps},
    )
    print(f"saved {args.output}")
    print("NOTE: replace proxy loss with latent denoise loss once video/latent export is wired.")


if __name__ == "__main__":
    main()
