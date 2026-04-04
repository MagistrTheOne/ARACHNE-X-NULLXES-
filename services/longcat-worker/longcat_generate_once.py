#!/usr/bin/env python3
"""
Single-job LongCat-Video inference (text / image / video-continuation).

Run only via torchrun (upstream uses torch.distributed), e.g.:

  cd /path/to/LongCat-Video
  PYTHONPATH=. torchrun --standalone --nproc_per_node=1 \\
    /path/to/longcat_generate_once.py --checkpoint_dir ./weights/LongCat-Video --job_json /tmp/job.json

Job JSON (required keys depend on task):
  task: "text-to-video" | "image-to-video" | "video-continuation"
  prompt: str
  output_mp4: str (absolute path to write final video)
  negative_prompt: optional str
  image_path: path to image file (image-to-video)
  video_path: path to conditioning mp4 (video-continuation)
  num_frames, height, width, num_inference_steps, guidance_scale: optional overrides
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Optional

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from diffusers.utils import load_image, load_video
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

_DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def torch_gc() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _load_job(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        job = json.load(f)
    if not isinstance(job, dict):
        raise SystemExit("job_json must be an object")
    return job


def _setup_distributed(context_parallel_size: int) -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    return local_rank, global_rank, num_processes


def _build_pipeline(checkpoint_dir: str, local_rank: int, enable_compile: bool) -> LongCatVideoPipeline:
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
    )
    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)
    return pipe


def _generator(local_rank: int, global_rank: int) -> torch.Generator:
    g = torch.Generator(device=local_rank)
    g.manual_seed(42 + global_rank)
    return g


def _ensure_parent_dir(path: str, local_rank: int) -> None:
    if local_rank != 0:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _validate_job(job: dict) -> Optional[str]:
    if "output_mp4" not in job:
        return "job_json missing output_mp4"
    if "prompt" not in job:
        return "job_json missing prompt"
    task = job.get("task", "text-to-video")
    if task == "image-to-video" and not job.get("image_path"):
        return "image-to-video requires image_path"
    if task == "video-continuation" and not job.get("video_path"):
        return "video-continuation requires video_path"
    if task not in ("text-to-video", "image-to-video", "video-continuation"):
        return f"unknown task: {task}"
    return None


def run_text_to_video(
    pipe: LongCatVideoPipeline,
    job: dict,
    local_rank: int,
    global_rank: int,
    checkpoint_dir: str,
    enable_compile: bool,
) -> None:
    prompt = job["prompt"]
    negative_prompt = job.get("negative_prompt") or _DEFAULT_NEGATIVE
    out_path = job["output_mp4"]
    spatial_refine_only = bool(job.get("spatial_refine_only", False))
    height = int(job.get("height", 480))
    width = int(job.get("width", 832))
    num_frames = int(job.get("num_frames", 93))
    steps_base = int(job.get("num_inference_steps", 50))
    guidance = float(job.get("guidance_scale", 4.0))
    generator = _generator(local_rank, global_rank)
    _ensure_parent_dir(out_path, local_rank)

    output = pipe.generate_t2v(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps_base,
        guidance_scale=guidance,
        generator=generator,
    )[0]

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(
            out_path + ".stage0.mp4",
            output_tensor,
            fps=15,
            video_codec="libx264",
            options={"crf": "18"},
        )
    del output
    torch_gc()

    cfg_step_lora_path = os.path.join(checkpoint_dir, "lora/cfg_step_lora.safetensors")
    pipe.dit.load_lora(cfg_step_lora_path, "cfg_step_lora")
    pipe.dit.enable_loras(["cfg_step_lora"])
    if enable_compile:
        pipe.dit = torch.compile(pipe.dit)

    output_distill = pipe.generate_t2v(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=16,
        use_distill=True,
        guidance_scale=1.0,
        generator=generator,
    )[0]
    pipe.dit.disable_all_loras()

    if local_rank == 0:
        output_processed_tensor = torch.from_numpy(np.array(output_distill))
        output_processed_tensor = (output_processed_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(
            out_path + ".stage1.mp4",
            output_processed_tensor,
            fps=15,
            video_codec="libx264",
            options={"crf": "18"},
        )

    refinement_lora_path = os.path.join(checkpoint_dir, "lora/refinement_lora.safetensors")
    pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
    pipe.dit.enable_loras(["refinement_lora"])
    pipe.dit.enable_bsa()
    if enable_compile:
        pipe.dit = torch.compile(pipe.dit)

    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
    del output_distill
    torch_gc()

    output_refine = pipe.generate_refine(
        prompt=prompt,
        stage1_video=stage1_video,
        num_inference_steps=steps_base,
        generator=generator,
        spatial_refine_only=spatial_refine_only,
    )[0]

    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    if local_rank == 0:
        output_tensor = torch.from_numpy(output_refine)
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        fps = 15 if spatial_refine_only else 30
        write_video(out_path, output_tensor, fps=fps, video_codec="libx264", options={"crf": "10"})


def run_image_to_video(
    pipe: LongCatVideoPipeline,
    job: dict,
    local_rank: int,
    global_rank: int,
    checkpoint_dir: str,
    enable_compile: bool,
) -> None:
    image_path = job["image_path"]
    image = load_image(image_path)
    prompt = job["prompt"]
    negative_prompt = job.get("negative_prompt") or _DEFAULT_NEGATIVE
    out_path = job["output_mp4"]
    spatial_refine_only = bool(job.get("spatial_refine_only", False))
    num_frames = int(job.get("num_frames", 93))
    steps_base = int(job.get("num_inference_steps", 50))
    guidance = float(job.get("guidance_scale", 4.0))
    resolution = job.get("resolution", "480p")
    generator = _generator(local_rank, global_rank)
    target_size = image.size
    _ensure_parent_dir(out_path, local_rank)

    output = pipe.generate_i2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=steps_base,
        guidance_scale=guidance,
        generator=generator,
    )[0]

    if local_rank == 0:
        out_list = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        out_list = [PIL.Image.fromarray(img) for img in out_list]
        out_list = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in out_list]
        output_tensor = torch.from_numpy(np.array(out_list))
        write_video(out_path + ".stage0.mp4", output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
    del output
    torch_gc()

    cfg_step_lora_path = os.path.join(checkpoint_dir, "lora/cfg_step_lora.safetensors")
    pipe.dit.load_lora(cfg_step_lora_path, "cfg_step_lora")
    pipe.dit.enable_loras(["cfg_step_lora"])
    if enable_compile:
        pipe.dit = torch.compile(pipe.dit)

    output_distill = pipe.generate_i2v(
        image=image,
        prompt=prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_inference_steps=16,
        use_distill=True,
        guidance_scale=1.0,
        generator=generator,
    )[0]
    pipe.dit.disable_all_loras()

    if local_rank == 0:
        output_processed = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
        output_processed = [PIL.Image.fromarray(img) for img in output_processed]
        output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]
        output_processed_tensor = torch.from_numpy(np.array(output_processed))
        write_video(
            out_path + ".stage1.mp4",
            output_processed_tensor,
            fps=15,
            video_codec="libx264",
            options={"crf": "18"},
        )

    refinement_lora_path = os.path.join(checkpoint_dir, "lora/refinement_lora.safetensors")
    pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
    pipe.dit.enable_loras(["refinement_lora"])
    pipe.dit.enable_bsa()
    if enable_compile:
        pipe.dit = torch.compile(pipe.dit)

    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
    del output_distill
    torch_gc()

    output_refine = pipe.generate_refine(
        image=image,
        prompt=prompt,
        stage1_video=stage1_video,
        num_cond_frames=1,
        num_inference_steps=steps_base,
        generator=generator,
        spatial_refine_only=spatial_refine_only,
    )[0]

    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    if local_rank == 0:
        output_refine_list = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        output_refine_list = [PIL.Image.fromarray(img) for img in output_refine_list]
        output_refine_list = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_refine_list]
        output_tensor = torch.from_numpy(np.array(output_refine_list))
        fps = 15 if spatial_refine_only else 30
        write_video(out_path, output_tensor, fps=fps, video_codec="libx264", options={"crf": "10"})


def run_video_continuation(
    pipe: LongCatVideoPipeline,
    job: dict,
    local_rank: int,
    global_rank: int,
    checkpoint_dir: str,
    enable_compile: bool,
) -> None:
    import cv2

    video_path = job["video_path"]
    video = load_video(video_path)
    prompt = job["prompt"]
    negative_prompt = job.get("negative_prompt") or _DEFAULT_NEGATIVE
    out_path = job["output_mp4"]
    spatial_refine_only = bool(job.get("spatial_refine_only", False))
    num_cond_frames = int(job.get("num_cond_frames", 13))
    num_frames = int(job.get("num_frames", 93))
    steps_base = int(job.get("num_inference_steps", 50))
    guidance = float(job.get("guidance_scale", 4.0))
    resolution = job.get("resolution", "480p")
    generator = _generator(local_rank, global_rank)
    _ensure_parent_dir(out_path, local_rank)

    cap = cv2.VideoCapture(video_path)
    current_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()

    target_fps = 15
    target_size = video[0].size
    stride = max(1, round(current_fps / target_fps))

    output = pipe.generate_vc(
        video=video[::stride],
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_cond_frames=num_cond_frames,
        num_inference_steps=steps_base,
        guidance_scale=guidance,
        generator=generator,
        use_kv_cache=True,
        offload_kv_cache=False,
    )[0]

    if local_rank == 0:
        out_list = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        out_list = [PIL.Image.fromarray(img) for img in out_list]
        out_list = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in out_list]
        combined = video[::stride] + out_list[num_cond_frames:]
        output_tensor = torch.from_numpy(np.array(combined))
        write_video(out_path + ".stage0.mp4", output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
    del output
    torch_gc()

    cfg_step_lora_path = os.path.join(checkpoint_dir, "lora/cfg_step_lora.safetensors")
    pipe.dit.load_lora(cfg_step_lora_path, "cfg_step_lora")
    pipe.dit.enable_loras(["cfg_step_lora"])
    if enable_compile:
        pipe.dit = torch.compile(pipe.dit)

    output_distill = pipe.generate_vc(
        video=video[::stride],
        prompt=prompt,
        resolution=resolution,
        num_frames=num_frames,
        num_cond_frames=num_cond_frames,
        num_inference_steps=16,
        use_distill=True,
        guidance_scale=1.0,
        generator=generator,
        use_kv_cache=True,
        offload_kv_cache=False,
        enhance_hf=False,
    )[0]
    pipe.dit.disable_all_loras()

    if local_rank == 0:
        output_processed = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
        output_processed = [PIL.Image.fromarray(img) for img in output_processed]
        output_processed = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_processed]
        combined = video[::stride] + output_processed[num_cond_frames:]
        output_tensor = torch.from_numpy(np.array(combined))
        write_video(
            out_path + ".stage1.mp4",
            output_tensor,
            fps=15,
            video_codec="libx264",
            options={"crf": "18"},
        )

    refinement_lora_path = os.path.join(checkpoint_dir, "lora/refinement_lora.safetensors")
    pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
    pipe.dit.enable_loras(["refinement_lora"])
    pipe.dit.enable_bsa()
    if enable_compile:
        pipe.dit = torch.compile(pipe.dit)

    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
    del output_distill
    torch_gc()

    target_fps_ref = 30
    stride_ref = max(1, round(current_fps / target_fps_ref))
    cur_num_cond_frames = num_cond_frames if spatial_refine_only else num_cond_frames * 2

    output_refine = pipe.generate_refine(
        video=video[::stride_ref],
        prompt=prompt,
        stage1_video=stage1_video,
        num_cond_frames=cur_num_cond_frames,
        num_inference_steps=steps_base,
        generator=generator,
        spatial_refine_only=spatial_refine_only,
    )[0]

    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    if local_rank == 0:
        output_refine_list = [(output_refine[i] * 255).astype(np.uint8) for i in range(output_refine.shape[0])]
        output_refine_list = [PIL.Image.fromarray(img) for img in output_refine_list]
        output_refine_list = [frame.resize(target_size, PIL.Image.BICUBIC) for frame in output_refine_list]
        combined = video[::stride_ref] + output_refine_list[cur_num_cond_frames:]
        output_tensor = torch.from_numpy(np.array(combined))
        fps = 15 if spatial_refine_only else 30
        write_video(out_path, output_tensor, fps=fps, video_codec="libx264", options={"crf": "10"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_json", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--enable_compile", action="store_true")
    args = parser.parse_args()

    job = _load_job(args.job_json)
    err = _validate_job(job)
    if err:
        print(err, file=sys.stderr)
        sys.exit(1)
    task = job.get("task", "text-to-video")

    local_rank, global_rank, _num_proc = _setup_distributed(args.context_parallel_size)
    pipe = _build_pipeline(args.checkpoint_dir, local_rank, args.enable_compile)

    try:
        if task == "text-to-video":
            run_text_to_video(
                pipe,
                job,
                local_rank,
                global_rank,
                args.checkpoint_dir,
                args.enable_compile,
            )
        elif task == "image-to-video":
            run_image_to_video(
                pipe,
                job,
                local_rank,
                global_rank,
                args.checkpoint_dir,
                args.enable_compile,
            )
        elif task == "video-continuation":
            run_video_continuation(
                pipe,
                job,
                local_rank,
                global_rank,
                args.checkpoint_dir,
                args.enable_compile,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
