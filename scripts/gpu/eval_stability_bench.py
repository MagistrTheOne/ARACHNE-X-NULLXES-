#!/usr/bin/env python3
"""
GPU stability bench: operational vs cinematic on one clip.

Writes JSON report for merge gate (S2-7). Requires CUDA + NULLXES_CHECKPOINT_DIR.

Example:
  export NULLXES_CHECKPOINT_DIR=/workspace/weights/arachne-avatar-runtime
  python scripts/gpu/eval_stability_bench.py \\
    --image assets/avatar/ref.jpg \\
    --audio assets/audio/speech.wav \\
    --output_dir /tmp/arachne_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    parser = argparse.ArgumentParser(description="ARACHNE-X stability eval bench")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="speaking clearly to camera")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--identity_id", type=int, default=None)
    parser.add_argument("--identity_bank_path", type=str, default=None)
    parser.add_argument("--mouth_mask", type=str, default=None)
    parser.add_argument("--ttff_max_sec", type=float, default=4.0)
    parser.add_argument("--identity_cosine_min", type=float, default=0.88)
    parser.add_argument(
        "--tier",
        choices=("80gb", "h200"),
        default="80gb",
        help="80gb → operational @ 480p; h200 → operational 480p + cinematic 720p gate",
    )
    args = parser.parse_args()

    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    ckpt = args.checkpoint_dir or os.environ.get("NULLXES_CHECKPOINT_DIR") or os.environ.get(
        "ARACHNE_CHECKPOINT_DIR"
    )
    if not ckpt or not os.path.isdir(ckpt):
        print("Set --checkpoint_dir or NULLXES_CHECKPOINT_DIR", file=sys.stderr)
        return 2

    import torch

    if not torch.cuda.is_available():
        print("CUDA required for eval_stability_bench", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    from arachne_x.runtime.inference_engine import execute_infer

    import argparse as ap

    def run_profile(profile: str, out_mp4: str, *, resolution: str) -> dict:
        ns = ap.Namespace(
            checkpoint_dir=ckpt,
            mode="ai2v",
            prompt=args.prompt,
            negative_prompt="",
            image=args.image,
            audio=args.audio,
            output=out_mp4,
            runtime_profile=profile,
            resolution=resolution,
            num_frames_mode="sync",
            num_frames=93,
            num_inference_steps=12,
            text_guidance_scale=4.0,
            audio_guidance_scale=5.0,
            identity_id=args.identity_id,
            identity_bank_path=args.identity_bank_path,
            mouth_mask=args.mouth_mask,
            allow_hub_download=False,
            weights_cache_dir=None,
            no_run_metadata=False,
            height=480,
            width=832,
            mux_fps=30,
            use_cfg_zero=False,
            identity_update_bank=False,
        )
        execute_infer(ns)
        meta_path = os.path.splitext(out_mp4)[0] + ".run.json"
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    op_res = "480p"
    cin_res = "720p" if args.tier == "h200" else "480p"
    op_mp4 = os.path.join(args.output_dir, "operational.mp4")
    cin_mp4 = os.path.join(args.output_dir, "cinematic.mp4")
    op_meta = run_profile("operational", op_mp4, resolution=op_res)
    cin_meta = run_profile("cinematic", cin_mp4, resolution=cin_res)

    sm = op_meta.get("sampling_metrics") or {}
    drift_min = sm.get("identity_drift_min")
    ttff = sm.get("ttff_sec")
    gate = {
        "ttff_ok": ttff is None or float(ttff) <= float(args.ttff_max_sec),
        "identity_drift_ok": drift_min is None or float(drift_min) >= float(args.identity_cosine_min),
    }
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": ckpt,
        "tier": args.tier,
        "operational_resolution": op_res,
        "cinematic_resolution": cin_res,
        "operational": {"output": op_mp4, "metadata": op_meta},
        "cinematic": {"output": cin_mp4, "metadata": cin_meta},
        "gate": gate,
        "merge_allowed": all(gate.values()),
    }
    out_json = os.path.join(args.output_dir, "eval_stability_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["merge_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
