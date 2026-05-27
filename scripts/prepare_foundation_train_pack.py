#!/usr/bin/env python3
"""
Fetch a small high-quality MP4+caption corpus from HF, QC-filter, optional latent export.

Output tree (training-ready smoke pack):
  {out_root}/
    manifest.json          # samples + paths + QC scores
    raw/{id}.mp4
    meta/{id}.json
    qc_frames/{id}.jpg     # mid-frame preview for visual audit
    latents/{id}.pt        # if --export-latents
    train_launch.env       # env vars for full train when GPUs ready

Usage:
  export PYTHONPATH=/workspace/ARACHNE-X
  export NULLXES_CHECKPOINT_DIR=/workspace/weights/ARACHNE-X-ULTRA-VIDEO
  python scripts/prepare_foundation_train_pack.py \\
    --out /workspace/datasets/arachne-foundation-smoke \\
    --target-samples 32 \\
    --export-latents
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

_CAPTION_KEYS = ("caption", "prompt", "text", "title", "Title", "description", "Caption")


def _caption_from_meta(meta: dict[str, Any]) -> str:
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return meta.strip()
    if not isinstance(meta, dict):
        return ""
    for key in _CAPTION_KEYS:
        val = meta.get(key)
        if val and str(val).strip():
            return str(val).strip()
    for val in meta.values():
        if isinstance(val, str) and len(val.strip()) > 24:
            return val.strip()
    return ""


def _laplacian_blur_score(bgr: np.ndarray) -> float:
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _probe_video(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mid = max(n // 2, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"cannot read frame from {path}")
    duration_s = n / fps if fps > 0 else 0.0
    return {
        "frame_count": n,
        "fps": fps,
        "width": w,
        "height": h,
        "duration_s": duration_s,
        "blur_score": _laplacian_blur_score(frame),
        "preview_bgr": frame,
    }


def _qc_accept(probe: dict[str, Any], caption: str, args: argparse.Namespace) -> tuple[bool, str]:
    if probe["width"] < args.min_width or probe["height"] < args.min_height:
        return False, "resolution_low"
    if probe["frame_count"] < args.min_frames:
        return False, "too_few_frames"
    if probe["duration_s"] < args.min_duration_s:
        return False, "too_short"
    if probe["blur_score"] < args.min_blur_score:
        return False, "blurry"
    if len(caption) < args.min_caption_chars:
        return False, "caption_short"
    return True, "ok"


def _write_preview(path: Path, bgr: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _iter_openvid_wds_shard(shard_path: Path):
    """Yield (key, mp4_bytes, meta_dict) from one OpenVid WebDataset tar."""
    with tarfile.open(shard_path, "r:") as tar:
        pending_mp4: dict[str, bytes] = {}
        pending_json: dict[str, dict] = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            m = re.match(r"^(\d+)\.(mp4|json)$", name)
            if not m:
                continue
            key, ext = m.group(1), m.group(2)
            data = tar.extractfile(member)
            if data is None:
                continue
            blob = data.read()
            if ext == "mp4":
                pending_mp4[key] = blob
            else:
                try:
                    pending_json[key] = json.loads(blob.decode("utf-8"))
                except json.JSONDecodeError:
                    pending_json[key] = {"raw": blob.decode("utf-8", errors="replace")}
            if key in pending_mp4 and key in pending_json:
                yield key, pending_mp4.pop(key), pending_json.pop(key)


def _find_local_shard(out_root: Path, shard_hint: str | None) -> Path | None:
    if shard_hint:
        p = Path(shard_hint)
        if p.is_file():
            return p
    candidates = sorted(out_root.glob("**/shard-*.tar"))
    if candidates:
        return candidates[0]
    wds_root = out_root / "openvid-wds"
    if wds_root.is_dir():
        c = sorted(wds_root.glob("**/shard-*.tar"))
        if c:
            return c[0]
    return None


def _download_shard(out_root: Path, shard_index: int) -> Path:
    from huggingface_hub import hf_hub_download

    shard_name = f"data/train/shard-{shard_index:05d}.tar"
    out_root.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id="Dev-Jahn/OpenVid-1M-wds",
        repo_type="dataset",
        filename=shard_name,
        local_dir=str(out_root / "openvid-wds"),
    )
    return Path(local)


def _collect_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out)
    raw_dir = out_root / "raw"
    meta_dir = out_root / "meta"
    qc_dir = out_root / "qc_frames"
    for d in (raw_dir, meta_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    shard = _find_local_shard(out_root, args.shard_path)
    if shard is None:
        print(f"Downloading OpenVid-wds shard {args.shard_index:05d}...", flush=True)
        shard = _download_shard(out_root, args.shard_index)

    accepted: list[dict[str, Any]] = []
    scanned = 0
    for key, mp4_bytes, meta in _iter_openvid_wds_shard(shard):
        scanned += 1
        if len(accepted) >= args.target_samples:
            break
        caption = _caption_from_meta(meta)
        sample_id = f"openvid_{key}"
        mp4_path = raw_dir / f"{sample_id}.mp4"
        mp4_path.write_bytes(mp4_bytes)
        try:
            probe = _probe_video(mp4_path)
        except ValueError as exc:
            mp4_path.unlink(missing_ok=True)
            print(f"skip {sample_id}: probe_fail {exc}", flush=True)
            continue
        ok, reason = _qc_accept(probe, caption, args)
        if not ok:
            mp4_path.unlink(missing_ok=True)
            print(f"skip {sample_id}: {reason}", flush=True)
            continue
        meta_out = {
            "id": sample_id,
            "source": "Dev-Jahn/OpenVid-1M-wds",
            "shard": shard.name,
            "wds_key": key,
            "caption": caption,
            "qc": {k: probe[k] for k in ("frame_count", "fps", "width", "height", "duration_s", "blur_score")},
        }
        (meta_dir / f"{sample_id}.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
        _write_preview(qc_dir / f"{sample_id}.jpg", probe.pop("preview_bgr"))
        accepted.append(
            {
                "id": sample_id,
                "video": str(mp4_path.relative_to(out_root)),
                "meta": str((meta_dir / f"{sample_id}.json").relative_to(out_root)),
                "prompt": caption,
                "negative_prompt": args.negative_prompt,
                "resolution": args.resolution,
                "num_frames": args.num_frames,
                "seed": args.base_seed + len(accepted),
                **meta_out["qc"],
            }
        )
        print(f"accept {len(accepted)}/{args.target_samples} {sample_id} {probe['width']}x{probe['height']}", flush=True)

    if len(accepted) < args.target_samples:
        raise RuntimeError(
            f"Only {len(accepted)}/{args.target_samples} passed QC after scanning {scanned} from {shard}"
        )
    return accepted


def _export_latents(samples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from arachne_x.loader import load_base_pipeline
    from arachne_x.training_latent_export_base import export_base_latent_training_pt

    ckpt = args.checkpoint_dir or os.environ.get("NULLXES_CHECKPOINT_DIR", "")
    if not ckpt:
        raise ValueError("Set --checkpoint-dir or NULLXES_CHECKPOINT_DIR for latent export")
    out_root = Path(args.out)
    latent_dir = out_root / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"Loading base pipeline from {ckpt} on {device}...", flush=True)
    pipe = load_base_pipeline(ckpt, device=device, torch_dtype=dtype)

    for i, sample in enumerate(samples):
        out_pt = latent_dir / f"{sample['id']}.pt"
        if out_pt.is_file() and not args.force_export:
            sample["latent"] = str(out_pt.relative_to(out_root))
            print(f"skip export {sample['id']} (exists)", flush=True)
            continue
        video_path = out_root / sample["video"]
        export_base_latent_training_pt(
            pipe,
            video_path=str(video_path),
            prompt=sample["prompt"],
            output_path=str(out_pt),
            negative_prompt=sample.get("negative_prompt", ""),
            resolution=sample.get("resolution", args.resolution),
            num_frames=int(sample.get("num_frames", args.num_frames)),
            seed=int(sample.get("seed", args.base_seed + i)),
            device=device,
        )
        sample["latent"] = str(out_pt.relative_to(out_root))


def _write_manifest(samples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    manifest = {
        "version": 1,
        "kind": "arachne_foundation_video_smoke",
        "description": "QC-filtered OpenVid-wds samples ready for foundation continue pretrain export",
        "source_datasets": ["Dev-Jahn/OpenVid-1M-wds"],
        "training": {
            "mode": "base_video",
            "resolution": args.resolution,
            "num_frames": args.num_frames,
            "checkpoint_video": args.checkpoint_dir or os.environ.get("NULLXES_CHECKPOINT_DIR", ""),
            "checkpoint_foundation": os.environ.get("ARACHNE_FOUNDATION_CKPT", ""),
        },
        "samples": samples,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env_lines = [
        f"export ARACHNE_ROOT={os.environ.get('ARACHNE_ROOT', '/workspace/ARACHNE-X')}",
        f"export PYTHONPATH=$ARACHNE_ROOT",
        f"export NULLXES_CHECKPOINT_DIR={manifest['training']['checkpoint_video']}",
        f"export ARACHNE_FOUNDATION_CKPT={manifest['training']['checkpoint_foundation'] or '/workspace/weights/ARACHNE-FOUNDATION-50B'}",
        f"export ARACHNE_DATASET_MANIFEST={out_root / 'manifest.json'}",
        f"export ARACHNE_LATENTS_DIR={out_root / 'latents'}",
        "# Full 50B train needs multi-GPU; smoke on 13.6B:",
        "# python $ARACHNE_ROOT/scripts/train_lora_base.py --manifest $ARACHNE_DATASET_MANIFEST --latents_dir $ARACHNE_LATENTS_DIR",
    ]
    (out_root / "train_launch.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_root / 'manifest.json'} ({len(samples)} samples)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare ARACHNE foundation training smoke pack")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--target-samples", type=int, default=32)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-path", default=None, help="Local OpenVid-wds shard tar")
    p.add_argument("--min-width", type=int, default=640)
    p.add_argument("--min-height", type=int, default=360)
    p.add_argument("--min-frames", type=int, default=48)
    p.add_argument("--min-duration-s", type=float, default=2.0)
    p.add_argument("--min-blur-score", type=float, default=80.0)
    p.add_argument("--min-caption-chars", type=int, default=24)
    p.add_argument("--resolution", default="480p")
    p.add_argument("--num-frames", type=int, default=93)
    p.add_argument("--negative-prompt", default="blurry, low quality, watermark, static, flicker")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--export-latents", action="store_true")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force-export", action="store_true")
    args = p.parse_args()

    samples = _collect_samples(args)
    if args.export_latents:
        _export_latents(samples, args)
    _write_manifest(samples, args)


if __name__ == "__main__":
    main()
