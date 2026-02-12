import argparse
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video
from diffusers.utils import load_image, load_video

from arachne_x.loader import load_base_pipeline, load_avatar_pipeline
from arachne_x.audio_process.torch_utils import save_video_ffmpeg


def _save_video(frames: np.ndarray, path: str, fps: int = 30) -> None:
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    write_video(path, torch.from_numpy(frames), fps=fps, video_codec="libx264", options={"crf": "18"})


def _audio_stream_generator(audio_path: str, chunk_duration: float = 0.5, sample_rate: int = 16000):
    import librosa

    audio, _ = librosa.load(audio_path, sr=sample_rate)
    chunk_samples = int(chunk_duration * sample_rate)
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        yield chunk

def _load_mask_tensor(mask_path: Optional[str]) -> Optional[torch.Tensor]:
    if not mask_path:
        return None
    img = Image.open(mask_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def main():
    parser = argparse.ArgumentParser(description="ARACHNE-X inference entrypoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["t2v", "i2v", "vc", "ai2v", "at2v", "avc", "streaming_ai2v"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--num_cond_frames", type=int, default=13)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--text_guidance_scale", type=float, default=4.0)
    parser.add_argument("--audio_guidance_scale", type=float, default=4.0)
    parser.add_argument("--identity_id", type=int, default=None)
    parser.add_argument("--identity_strength", type=float, default=1.0)
    parser.add_argument("--identity_negative_strength", type=float, default=0.0)
    parser.add_argument("--identity_update_bank", action="store_true")
    parser.add_argument("--identity_update_momentum", type=float, default=0.25)
    parser.add_argument("--emotion_id", type=str, default=None)
    parser.add_argument("--emotion_intensity", type=float, default=0.0)
    parser.add_argument("--emotion_guidance_scale", type=float, default=0.0)
    parser.add_argument("--mouth_mask", type=str, default=None)
    parser.add_argument("--disable_phoneme_conditioning", action="store_true")
    parser.add_argument("--phoneme_stream_scale", type=float, default=None)
    parser.add_argument("--output", type=str, default="output.mp4")
    args = parser.parse_args()
    emotion_id = args.emotion_id
    if isinstance(emotion_id, str):
        s = emotion_id.strip()
        if s.lstrip("-").isdigit():
            emotion_id = int(s)
    mouth_mask_tensor = _load_mask_tensor(args.mouth_mask)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.mode in ("t2v", "i2v", "vc"):
        pipe = load_base_pipeline(args.checkpoint_dir, device=device, torch_dtype=torch_dtype)

        if args.mode == "t2v":
            out = pipe.generate_t2v(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.text_guidance_scale,
            )[0]
            _save_video(out, args.output, fps=30)

        elif args.mode == "i2v":
            if not args.image:
                raise ValueError("--image is required for i2v")
            image = load_image(args.image)
            out = pipe.generate_i2v(
                image=image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                resolution=args.resolution,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.text_guidance_scale,
            )[0]
            _save_video(out, args.output, fps=30)

        elif args.mode == "vc":
            if not args.video:
                raise ValueError("--video is required for vc")
            video = load_video(args.video)
            out = pipe.generate_vc(
                video=video,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                resolution=args.resolution,
                num_frames=args.num_frames,
                num_cond_frames=args.num_cond_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.text_guidance_scale,
                use_kv_cache=True,
                offload_kv_cache=False,
            )[0]
            _save_video(out, args.output, fps=30)

        return

    pipe = load_avatar_pipeline(
        args.checkpoint_dir,
        variant="multi" if args.mode == "avc" else "single",
        device=device,
        torch_dtype=torch_dtype,
    )
    if hasattr(pipe, "phoneme_enabled"):
        pipe.phoneme_enabled = not args.disable_phoneme_conditioning
    if args.phoneme_stream_scale is not None and hasattr(pipe, "phoneme_stream_scale"):
        pipe.phoneme_stream_scale = float(args.phoneme_stream_scale)

    if args.mode in ("ai2v", "streaming_ai2v"):
        if not args.image or not args.audio:
            raise ValueError("--image and --audio are required for ai2v")
        image = load_image(args.image)
        if args.mode == "ai2v":
            out = pipe.generate_ai2v(
                image=image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                audio_path=args.audio,
                resolution=args.resolution,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                text_guidance_scale=args.text_guidance_scale,
                audio_guidance_scale=args.audio_guidance_scale,
                identity_id=args.identity_id,
                identity_strength=args.identity_strength,
                identity_negative_strength=args.identity_negative_strength,
                update_identity_bank=args.identity_update_bank,
                identity_update_momentum=args.identity_update_momentum,
                emotion_id=emotion_id,
                emotion_intensity=args.emotion_intensity,
                emotion_guidance_scale=args.emotion_guidance_scale,
                mouth_zone_masks=mouth_mask_tensor,
            )[0]
            save_video_ffmpeg(out, args.output, fps=30)
        else:
            audio_gen = _audio_stream_generator(args.audio)
            frames = []
            for frame in pipe.generate_streaming_ai2v(
                image=image,
                prompt=args.prompt,
                audio_stream=audio_gen,
                resolution=args.resolution,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                text_guidance_scale=args.text_guidance_scale,
                audio_guidance_scale=args.audio_guidance_scale,
                identity_id=args.identity_id,
                identity_strength=args.identity_strength,
                identity_negative_strength=args.identity_negative_strength,
                emotion_id=emotion_id,
                emotion_intensity=args.emotion_intensity,
                emotion_guidance_scale=args.emotion_guidance_scale,
                mouth_zone_masks=mouth_mask_tensor,
            ):
                frames.append(frame)
            save_video_ffmpeg(np.stack(frames, axis=0), args.output, fps=30)
        return

    if args.mode == "at2v":
        if not args.audio:
            raise ValueError("--audio is required for at2v")
        out = pipe.generate_at2v(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            audio_path=args.audio,
            resolution=args.resolution,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            text_guidance_scale=args.text_guidance_scale,
            audio_guidance_scale=args.audio_guidance_scale,
            identity_id=args.identity_id,
            identity_strength=args.identity_strength,
            identity_negative_strength=args.identity_negative_strength,
            emotion_id=emotion_id,
            emotion_intensity=args.emotion_intensity,
            emotion_guidance_scale=args.emotion_guidance_scale,
            mouth_zone_masks=mouth_mask_tensor,
        )[0]
        save_video_ffmpeg(out, args.output, fps=30)
        return

    if args.mode == "avc":
        if not args.video or not args.audio:
            raise ValueError("--video and --audio are required for avc")
        video = load_video(args.video)
        out = pipe.generate_avc(
            video=video,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            audio_path=args.audio,
            resolution=args.resolution,
            num_frames=args.num_frames,
            num_cond_frames=args.num_cond_frames,
            num_inference_steps=args.num_inference_steps,
            text_guidance_scale=args.text_guidance_scale,
            audio_guidance_scale=args.audio_guidance_scale,
            identity_id=args.identity_id,
            identity_strength=args.identity_strength,
            identity_negative_strength=args.identity_negative_strength,
            update_identity_bank=args.identity_update_bank,
            identity_update_momentum=args.identity_update_momentum,
            emotion_id=emotion_id,
            emotion_intensity=args.emotion_intensity,
            emotion_guidance_scale=args.emotion_guidance_scale,
            mouth_zone_masks=mouth_mask_tensor,
        )[0]
        save_video_ffmpeg(out, args.output, fps=30)
        return


if __name__ == "__main__":
    main()
