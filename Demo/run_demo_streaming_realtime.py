#!/usr/bin/env python3
"""
ARACHNE-X Real-Time Streaming Demo
Production-ready streaming inference on NVIDIA H200
"""

import os
import json
import time
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.utils import load_image

from arachne_x.loader import load_avatar_pipeline
from arachne_x.audio_process.torch_utils import save_video_ffmpeg


def audio_stream_generator(audio_path: str, chunk_duration: float = 0.5, sample_rate: int = 16000):
    """
    Generator that yields audio chunks for streaming.
    Args:
        audio_path: Path to audio file.
        chunk_duration: Duration of each chunk in seconds.
        sample_rate: Sample rate (16000 Hz).
    """
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    chunk_samples = int(chunk_duration * sample_rate)
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i+chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        yield chunk


def main():
    parser = argparse.ArgumentParser(description="ARACHNE-X Real-Time Streaming Inference")
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--audio', type=str, required=True, help='Input audio path')
    parser.add_argument('--prompt', type=str, default='A person speaking naturally', help='Text prompt')
    parser.add_argument('--output_dir', type=str, default='./outputs_streaming', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./weights/ARACHNE-X-Avatar', help='Checkpoint directory')
    parser.add_argument('--num_inference_steps', type=int, default=8, help='Denoising steps (8 = fast distilled)')
    parser.add_argument('--resolution', type=str, default='480p', choices=['480p', '720p'], help='Video resolution')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        load_device = f"cuda:{local_rank}"
        pipe_device = "cuda"
        pipe_dtype = torch.bfloat16
    else:
        load_device = "cpu"
        pipe_device = "cpu"
        pipe_dtype = torch.float32
    
    # Load models
    print("[*] Loading models...")
    pipe = load_avatar_pipeline(
        args.checkpoint_dir,
        variant="single",
        device=load_device,
        torch_dtype=pipe_dtype,
    )
    pipe.to(pipe_device)
    
    # Load image
    image = load_image(args.image)
    
    # Create audio stream generator
    audio_gen = audio_stream_generator(args.audio, chunk_duration=0.5, sample_rate=16000)
    
    # Generate streaming
    print("[*] Starting real-time streaming generation...")
    print(f"    Prompt: {args.prompt}")
    print(f"    Resolution: {args.resolution}")
    print(f"    Steps: {args.num_inference_steps} (distilled)")
    
    frames = []
    start_time = time.time()
    frame_count = 0
    
    for frame_np in pipe.generate_streaming_ai2v(
        image=image,
        prompt=args.prompt,
        audio_stream=audio_gen,
        resolution=args.resolution,
        num_frames=93,
        num_inference_steps=args.num_inference_steps,
        text_guidance_scale=4.0,
        audio_guidance_scale=4.0,
    ):
        frames.append(frame_np)
        frame_count += 1
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        if frame_count % 10 == 0:
            print(f"    [Frame {frame_count}] {fps:.1f} FPS, {elapsed:.1f}s elapsed")
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n[✓] Generation complete!")
    print(f"    Frames: {frame_count}")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Average FPS: {avg_fps:.1f}")
    
    # Save video
    if frames:
        output_video = np.stack(frames)
        output_path = os.path.join(args.output_dir, 'streaming_output.mp4')
        save_video_ffmpeg(
            torch.from_numpy(output_video),
            output_path,
            args.audio,
            fps=16,
            quality=5
        )
        print(f"[✓] Video saved to: {output_path}")


if __name__ == '__main__':
    main()

