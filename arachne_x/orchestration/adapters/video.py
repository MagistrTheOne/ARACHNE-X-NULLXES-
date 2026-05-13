from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from ..presets import get_video_profile
from ..subprocess_utils import run_python_script


VIDEO_SCRIPT = r"""
import json
import os
import sys
import numpy as np
import torch
from torchvision.io import write_video
from arachne_x.loader import load_base_pipeline

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
profile = cfg["profile"]
pipe = load_base_pipeline(cfg["checkpoint_dir"], device="cuda", torch_dtype=torch.bfloat16)

if profile.get("lora_file"):
    lora_path = os.path.join(cfg["checkpoint_dir"], profile["lora_file"])
    pipe.dit.load_lora(lora_path, profile.get("lora_key", "cfg_step_lora"))
    pipe.dit.enable_loras([profile.get("lora_key", "cfg_step_lora")])

g = torch.Generator(device="cuda").manual_seed(int(cfg.get("seed", 778)))
out = pipe.generate_t2v(
    prompt=cfg["prompt"],
    negative_prompt=cfg["negative_prompt"],
    height=int(profile["height"]),
    width=int(profile["width"]),
    num_frames=int(profile["num_frames"]),
    num_inference_steps=int(profile["num_inference_steps"]),
    use_distill=bool(profile.get("use_distill", False)),
    guidance_scale=float(profile["guidance_scale"]),
    generator=g,
)[0]

video = torch.from_numpy(np.array(out))
video = (video * 255).clamp(0, 255).to(torch.uint8)
write_video(
    cfg["output_path"],
    video,
    fps=int(profile.get("fps", 30)),
    video_codec="libx264",
    options={"crf": str(profile.get("crf", "18"))},
)
json.dump({"video_path": cfg["output_path"]}, open(cfg["result_path"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
"""


def run_video(
    *,
    python_bin: str,
    work_dir: str | Path,
    checkpoint_dir: str,
    prompt: str,
    negative_prompt: str,
    profile_name: str,
    seed: int,
) -> Tuple[Dict[str, object], float]:
    work = Path(work_dir)
    output_path = str(work / "video.mp4")
    (work / "video_prompt.txt").write_text(prompt, encoding="utf-8")
    (work / "negative_prompt.txt").write_text(negative_prompt, encoding="utf-8")
    return run_python_script(
        python_bin=python_bin,
        script_text=VIDEO_SCRIPT,
        config={
            "checkpoint_dir": checkpoint_dir,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "profile": get_video_profile(profile_name),
            "seed": seed,
            "output_path": output_path,
        },
        work_dir=work,
        name="video",
    )
