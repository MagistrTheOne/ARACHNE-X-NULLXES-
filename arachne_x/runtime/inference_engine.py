"""
Programmatic inference entrypoint for ARACHNE-X (GTM runtime layer).

``scripts/infer.py`` is a thin CLI over :func:`execute_infer`.
Services and tests should import :class:`InferenceEngine` or call ``execute_infer(ns)`` directly.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video
from diffusers.utils import load_image, load_video

from arachne_x.loader import load_avatar_pipeline, load_base_pipeline
from arachne_x.audio_process.torch_utils import save_video_ffmpeg
from arachne_x.inference_audio import build_avatar_windowed_audio_emb
from arachne_x.pipeline_arachne_x_video_avatar import retrieve_latents
from arachne_x.tts import create_speech_synthesizer
from arachne_x.tts.chunking import iter_audio_micro_turns_from_file
from arachne_x.tts.realtime import DEFAULT_MICRO_TURN_SECONDS
from arachne_x.weights_resolve import resolve_weights_root


def save_video_numpy(frames: np.ndarray, path: str, fps: int = 30) -> None:
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    write_video(path, torch.from_numpy(frames), fps=fps, video_codec="libx264", options={"crf": "18"})


def load_mask_tensor(mask_path: Optional[str]) -> Optional[torch.Tensor]:
    if not mask_path:
        return None
    img = Image.open(mask_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def get_hw_for_resolution(resolution: str, height: int, width: int) -> tuple[int, int]:
    if resolution == "720p" and (height, width) == (480, 832):
        return (768, 1280)
    return (height, width)


def build_audio_emb(pipe, audio_path: str, num_frames: int, device: str, sample_rate: int = 16000) -> torch.Tensor:
    return build_avatar_windowed_audio_emb(pipe, audio_path, num_frames, device, sample_rate)


def resolve_avatar_wav_path(args: argparse.Namespace) -> Tuple[str, bool]:
    """Returns (path_to_wav, is_temp). Prefer explicit --audio over --speak_text."""
    if getattr(args, "audio", None):
        p = os.path.abspath(args.audio)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"--audio not found: {p}")
        return p, False
    speak = (getattr(args, "speak_text", None) or "").strip()
    if not speak:
        raise ValueError("Provide --audio or non-empty --speak_text for this mode.")
    tts_kw = {}
    prov = (args.tts_provider or "").strip().lower()
    if prov in ("longcat_audiodit", "audiodit"):
        tts_kw = dict(
            audiodit_nfe=args.audiodit_nfe,
            audiodit_guidance_strength=args.audiodit_guidance_strength,
            audiodit_guidance_method=args.audiodit_guidance_method,
            audiodit_prompt_audio=args.audiodit_prompt_audio,
            audiodit_prompt_text=args.audiodit_prompt_text,
            audiodit_seed=args.audiodit_seed,
        )
    synth = create_speech_synthesizer(
        args.tts_provider,
        model_id=args.tts_model,
        device_map=args.tts_device_map,
        language=args.tts_language,
        speaker=args.tts_speaker,
        instruct=args.tts_instruct or None,
        attn_implementation=args.tts_attn,
        **tts_kw,
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="arachne_tts_")
    os.close(fd)
    synth.synthesize_to_path(speak, tmp_path)
    return tmp_path, True


def maybe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def resolve_lora_rank_alpha(
    lora_path: str,
    lora_rank: Optional[int],
    lora_alpha: Optional[float],
    lora_meta_json: Optional[str],
) -> Tuple[int, float]:
    meta_path = lora_meta_json
    if meta_path is None and lora_path:
        candidate = os.path.join(os.path.dirname(os.path.abspath(lora_path)), "lora_train_meta.json")
        if os.path.isfile(candidate):
            meta_path = candidate
    rank, alpha = lora_rank, lora_alpha
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if rank is None:
            rank = int(meta.get("lora_rank", 128))
        if alpha is None:
            alpha = float(meta.get("lora_alpha", 64.0))
    if rank is None:
        rank = 128
    if alpha is None:
        alpha = 64.0
    return rank, alpha


def maybe_load_avatar_lora(pipe, args: argparse.Namespace) -> None:
    if not getattr(args, "lora_path", None):
        return
    path = args.lora_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--lora_path not found: {path}")
    rank, alpha = resolve_lora_rank_alpha(
        path,
        getattr(args, "lora_rank", None),
        getattr(args, "lora_alpha", None),
        getattr(args, "lora_meta_json", None),
    )
    key = getattr(args, "lora_key", "train")
    pipe.dit.load_lora(path, key, multiplier=1.0, lora_network_dim=rank, lora_network_alpha=alpha)
    pipe.dit.enable_loras([key])
    print(f"[lora] loaded {path} key={key} rank={rank} alpha={alpha}")


def build_video_latent(pipe, video, num_cond_frames: int, height: int, width: int, device: str) -> torch.Tensor:
    video_tensor = pipe.video_processor.preprocess_video(
        pipe.video_processor,
        video,
        height=height,
        width=width,
        resize_mode="crop",
    ).to(device=device, dtype=pipe.dit.dtype)
    cond_videos = video_tensor[:, :, -num_cond_frames:]
    cond_videos_latents = retrieve_latents(pipe.vae.encode(cond_videos), generator=None, sample_mode="argmax")
    return pipe.normalize_latents(cond_videos_latents)


def execute_infer(args: argparse.Namespace) -> None:
    """
    Run one inference job from a populated ``argparse.Namespace`` (same contract as ``scripts/infer.py`` CLI).
    """
    emotion_id = args.emotion_id
    if isinstance(emotion_id, str):
        s = emotion_id.strip()
        if s.lstrip("-").isdigit():
            emotion_id = int(s)

    args.height, args.width = get_hw_for_resolution(args.resolution, args.height, args.width)
    mouth_mask_tensor = load_mask_tensor(args.mouth_mask)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    checkpoint_dir = resolve_weights_root(
        args.checkpoint_dir,
        allow_hub=args.allow_hub_download,
        cache_dir=args.weights_cache_dir,
    )

    if args.mode in ("t2v", "i2v", "vc"):
        pipe = load_base_pipeline(checkpoint_dir, device=device, torch_dtype=torch_dtype)

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
            save_video_numpy(out, args.output, fps=30)

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
            save_video_numpy(out, args.output, fps=30)

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
            save_video_numpy(out, args.output, fps=30)

        return

    pipe = load_avatar_pipeline(
        checkpoint_dir,
        variant="multi" if args.mode == "avc" else "single",
        device=device,
        torch_dtype=torch_dtype,
    )
    maybe_load_avatar_lora(pipe, args)
    if hasattr(pipe, "phoneme_enabled"):
        pipe.phoneme_enabled = not args.disable_phoneme_conditioning
    if args.phoneme_stream_scale is not None and hasattr(pipe, "phoneme_stream_scale"):
        pipe.phoneme_stream_scale = float(args.phoneme_stream_scale)

    if args.identity_bank_path and os.path.exists(args.identity_bank_path):
        loaded = pipe.load_identity_bank(
            args.identity_bank_path,
            strict=args.identity_bank_load_strict,
        )
        print(
            f"[identity-bank] loaded from {loaded['source']} (rows={loaded['rows_loaded']}, cols={loaded['cols_loaded']})"
        )
    elif args.identity_bank_path and args.mode != "enroll_identity":
        print(f"[identity-bank] warning: file not found at {args.identity_bank_path}; continuing with in-memory bank")

    if args.mode == "enroll_identity":
        if not args.image:
            raise ValueError("--image is required for enroll_identity")
        if args.identity_id is None:
            raise ValueError("--identity_id is required for enroll_identity")
        save_path = args.identity_bank_save_path or args.identity_bank_path
        if not save_path:
            raise ValueError("For enroll_identity, provide --identity_bank_save_path or --identity_bank_path.")

        image = load_image(args.image)
        enroll_info = pipe.enroll_identity_from_image(
            image=image,
            identity_id=args.identity_id,
            resolution=args.resolution,
            resize_mode="crop",
            momentum=args.identity_update_momentum,
        )
        pipe.save_identity_bank(save_path)
        print(
            "[identity-bank] enrolled identity_id(s)={} batch_size={} saved={}".format(
                enroll_info["identity_ids"],
                enroll_info["batch_size"],
                save_path,
            )
        )
        return

    if args.mode in ("ai2v", "streaming_ai2v"):
        if not args.image:
            raise ValueError("--image is required for ai2v / streaming_ai2v")
        wav_path, wav_is_temp = resolve_avatar_wav_path(args)
        image = load_image(args.image)
        try:
            if args.mode == "ai2v":
                audio_emb = build_audio_emb(
                    pipe,
                    audio_path=wav_path,
                    num_frames=args.num_frames,
                    device=device,
                )
                out = pipe.generate_ai2v(
                    image=image,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    resolution=args.resolution,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    text_guidance_scale=args.text_guidance_scale,
                    audio_guidance_scale=args.audio_guidance_scale,
                    audio_emb=audio_emb,
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
                save_video_ffmpeg(out, args.output, wav_path, fps=30)
                if args.identity_update_bank:
                    save_path = args.identity_bank_save_path or args.identity_bank_path
                    if save_path:
                        pipe.save_identity_bank(save_path)
                        print(f"[identity-bank] updated and saved to {save_path}")
            else:
                audio_gen = iter_audio_micro_turns_from_file(
                    wav_path,
                    chunk_duration_sec=args.audio_chunk_sec,
                    sample_rate=16000,
                )
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
                save_video_ffmpeg(np.stack(frames, axis=0), args.output, wav_path, fps=30)
                if args.identity_bank_save_path:
                    pipe.save_identity_bank(args.identity_bank_save_path)
                    print(f"[identity-bank] saved to {args.identity_bank_save_path}")
        finally:
            if wav_is_temp:
                maybe_unlink(wav_path)
        return

    if args.mode == "at2v":
        wav_path, wav_is_temp = resolve_avatar_wav_path(args)
        try:
            audio_emb = build_audio_emb(
                pipe,
                audio_path=wav_path,
                num_frames=args.num_frames,
                device=device,
            )
            out = pipe.generate_at2v(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                text_guidance_scale=args.text_guidance_scale,
                audio_guidance_scale=args.audio_guidance_scale,
                audio_emb=audio_emb,
                identity_id=args.identity_id,
                identity_strength=args.identity_strength,
                identity_negative_strength=args.identity_negative_strength,
                emotion_id=emotion_id,
                emotion_intensity=args.emotion_intensity,
                emotion_guidance_scale=args.emotion_guidance_scale,
                mouth_zone_masks=mouth_mask_tensor,
            )[0]
            save_video_ffmpeg(out, args.output, wav_path, fps=30)
        finally:
            if wav_is_temp:
                maybe_unlink(wav_path)
        return

    if args.mode == "avc":
        if not args.video:
            raise ValueError("--video is required for avc")
        wav_path, wav_is_temp = resolve_avatar_wav_path(args)
        try:
            video = load_video(args.video)
            audio_emb = build_audio_emb(
                pipe,
                audio_path=wav_path,
                num_frames=args.num_frames,
                device=device,
            )
            video_latent = build_video_latent(
                pipe,
                video=video,
                num_cond_frames=args.num_cond_frames,
                height=args.height,
                width=args.width,
                device=device,
            )
            out = pipe.generate_avc(
                video=video,
                video_latent=video_latent,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                audio_emb=audio_emb,
                height=args.height,
                width=args.width,
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
            save_video_ffmpeg(out, args.output, wav_path, fps=30)
            if args.identity_update_bank:
                save_path = args.identity_bank_save_path or args.identity_bank_path
                if save_path:
                    pipe.save_identity_bank(save_path)
                    print(f"[identity-bank] updated and saved to {save_path}")
        finally:
            if wav_is_temp:
                maybe_unlink(wav_path)
        return


class InferenceEngine:
    """
    Thin facade for programmatic inference.

    Example::

        from arachne_x.runtime import InferenceEngine
        import argparse

        ns = argparse.Namespace(...)
        InferenceEngine.execute(ns)
    """

    @staticmethod
    def execute(args: argparse.Namespace) -> None:
        execute_infer(args)
