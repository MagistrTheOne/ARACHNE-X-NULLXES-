#!/usr/bin/env python3
"""
Export many avatar training .pt files from a JSON manifest of **http(s)** links.

1. Copy ``examples/avatar_latent_url_manifest.example.json`` → e.g. ``my_manifest.json``.
2. Replace ``image_url`` / ``audio_url`` / ``prompt`` with real URLs (CDN, Hugging Face raw, presigned S3, etc.).
3. Run once (loads avatar pipeline **once**):

   python scripts/export_latent_batch_from_urls.py \\
     --checkpoint_dir /path/to/weights \\
     --manifest my_manifest.json \\
     --output_dir /path/to/latent_dataset

Requires: network access from the pod; same ``--allow_hub_download`` / cache flags as other scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from diffusers.utils import load_image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.loader import load_avatar_pipeline
from arachne_x.training_latent_export import export_avatar_latent_training_pt
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root


def _download(url: str, suffix: str) -> str:
    """Download URL to a temp file; caller should unlink. Raises on HTTP error."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ARACHNE-X-export_latent_batch_from_urls/1.0"},
    )
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="arachne_url_dl_")
    os.close(fd)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _load_manifest(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "samples" in raw:
        raw = raw["samples"]
    if not isinstance(raw, list):
        raise ValueError("Manifest must be a JSON array or { \"samples\": [ ... ] }")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch export latent .pt from URL manifest")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True, help="JSON list of {id, image_url, audio_url, prompt, ...}")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    add_resolve_args(parser)
    args = parser.parse_args()

    entries = _load_manifest(args.manifest)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    root = resolve_weights_root(
        args.checkpoint_dir,
        allow_hub=args.allow_hub_download,
        cache_dir=args.weights_cache_dir,
    )
    pipe = load_avatar_pipeline(root, variant="single", device=device, torch_dtype=dtype)

    for i, row in enumerate(entries):
        cid = row.get("id") or f"sample_{i:05d}"
        img_u = row.get("image_url") or row.get("image")
        aud_u = row.get("audio_url") or row.get("audio")
        if not img_u or not aud_u:
            raise ValueError(f"Entry {cid}: need image_url and audio_url")
        prompt = row.get("prompt") or ""
        neg = row.get("negative_prompt") or ""
        nf = int(row.get("num_frames", 93))
        seed = row.get("seed")
        if seed is not None:
            seed = int(seed)

        tmp_img = tmp_aud = None
        try:
            suf_img = Path(urllib.parse.urlparse(img_u).path).suffix or ".jpg"
            suf_aud = Path(urllib.parse.urlparse(aud_u).path).suffix or ".wav"
        except Exception:
            suf_img, suf_aud = ".jpg", ".wav"

        try:
            tmp_img = _download(img_u, suf_img)
            tmp_aud = _download(aud_u, suf_aud)
            image = load_image(tmp_img)
            out_pt = os.path.join(args.output_dir, f"{cid}.pt")
            export_avatar_latent_training_pt(
                pipe,
                image=image,
                audio_path=tmp_aud,
                prompt=prompt,
                output_path=out_pt,
                negative_prompt=neg,
                resolution=args.resolution,
                num_frames=nf,
                seed=seed,
                device=device,
            )
        finally:
            for p in (tmp_img, tmp_aud):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    print(f"[done] wrote {len(entries)} samples under {args.output_dir}")


if __name__ == "__main__":
    main()
