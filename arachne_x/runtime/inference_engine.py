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
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.io import write_video
from diffusers.utils import load_image, load_video

from arachne_x.loader import load_avatar_pipeline, load_base_pipeline, load_audio_i2v_pipeline
from arachne_x.audio_process.torch_utils import save_video_ffmpeg
from arachne_x.inference_audio import build_avatar_windowed_audio_emb, default_embedding_fps
from arachne_x.inference_frames import (
    audio_duration_sec,
    resolve_num_frames,
    suggest_embedding_fps,
)
from arachne_x.pipeline_arachne_x_video_avatar import retrieve_latents
from arachne_x.tts import create_speech_synthesizer
from arachne_x.tts.chunking import iter_audio_micro_turns_from_file
from arachne_x.tts.realtime import DEFAULT_MICRO_TURN_SECONDS
from arachne_x.runtime.prompt_compiler_runtime import apply_prompt_compiler, resolve_imagine_compiler_backend
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


def build_audio_emb(
    pipe,
    audio_path: str,
    num_frames: int,
    device: str,
    sample_rate: int = 16000,
    embedding_fps: Optional[float] = None,
) -> torch.Tensor:
    return build_avatar_windowed_audio_emb(
        pipe, audio_path, num_frames, device, sample_rate, embedding_fps=embedding_fps
    )


def configure_avatar_pipe(pipe, args: argparse.Namespace) -> None:
    if hasattr(pipe, "phoneme_enabled"):
        pipe.phoneme_enabled = not args.disable_phoneme_conditioning
    if args.phoneme_stream_scale is not None and hasattr(pipe, "phoneme_stream_scale"):
        pipe.phoneme_stream_scale = float(args.phoneme_stream_scale)
    pipe.skip_audio_noise_floor = bool(getattr(args, "skip_audio_noise_floor", False))
    if getattr(args, "no_hybrid_renderer", False):
        pipe.hybrid_renderer_enabled = False
    if getattr(args, "hybrid_mouth_strength", None) is not None:
        pipe.hybrid_renderer_mouth_strength = float(args.hybrid_mouth_strength)
    if getattr(args, "hybrid_temporal_alpha", None) is not None:
        pipe.hybrid_renderer_temporal_alpha = float(args.hybrid_temporal_alpha)


def apply_avatar_frame_budget(
    args: argparse.Namespace, pipe, wav_path: Optional[str]
) -> Dict[str, Any]:
    """Resolve ``args.num_frames`` and embedding fps from audio duration (avatar modes)."""
    if not wav_path or not os.path.isfile(wav_path):
        return {}
    mode = getattr(args, "num_frames_mode", "explicit") or "explicit"
    mux_fps = float(getattr(args, "mux_fps", 30))
    base_fps = getattr(args, "audio_embedding_fps", None)
    base_fps = float(base_fps) if base_fps is not None else default_embedding_fps(pipe)
    dur = audio_duration_sec(wav_path)

    if mode == "explicit":
        _, info = resolve_num_frames("explicit", dur, base_fps, explicit=args.num_frames, mux_fps=mux_fps)
        info["chosen"] = int(args.num_frames)
        chosen = int(args.num_frames)
    else:
        chosen, info = resolve_num_frames(
            mode, dur, base_fps, explicit=args.num_frames, mux_fps=mux_fps
        )
        args.num_frames = chosen

    embedding_fps = base_fps
    if getattr(args, "embedding_fps_auto", False) or (
        mode == "duration" and chosen > info.get("sync_max_frames", chosen)
    ):
        embedding_fps = suggest_embedding_fps(dur, chosen, base_fps)

    if chosen > info.get("sync_max_frames", chosen) and embedding_fps <= base_fps:
        print(
            f"[frame-budget] warning: num_frames={chosen} exceeds sync_max={info.get('sync_max_frames')} "
            f"at embedding_fps={base_fps:.1f}; tail may clamp (try --embedding_fps_auto)"
        )

    info["embedding_fps_final"] = embedding_fps
    info["chosen"] = chosen
    pipe.inference_embedding_fps = embedding_fps
    args._resolved_embedding_fps = embedding_fps

    print(
        "[frame-budget] "
        f"mode={mode} duration_sec={dur:.3f} sync_max={info.get('sync_max_frames')} "
        f"duration_frames={info.get('duration_frames')} chosen={chosen} embedding_fps={embedding_fps:.1f}"
    )
    return info


def resolved_embedding_fps(args: argparse.Namespace, pipe) -> Optional[float]:
    fps = getattr(args, "_resolved_embedding_fps", None)
    if fps is not None:
        return float(fps)
    if getattr(args, "audio_embedding_fps", None) is not None:
        return float(args.audio_embedding_fps)
    return getattr(pipe, "inference_embedding_fps", None)


def save_avatar_mp4(
    frames,
    output_path: str,
    wav_path: Optional[str],
    args: argparse.Namespace,
) -> None:
    fps = int(getattr(args, "mux_fps", 30))
    export_crf = getattr(args, "export_crf", None)
    save_video_ffmpeg(
        frames,
        output_path,
        wav_path,
        fps=fps,
        high_quality_save=bool(getattr(args, "high_quality_save", False)),
        export_crf=export_crf,
    )


def write_run_metadata(
    args: argparse.Namespace,
    frame_budget: Optional[Dict[str, Any]] = None,
) -> None:
    if getattr(args, "no_run_metadata", False):
        return
    path = getattr(args, "run_metadata_json", None)
    if not path:
        base, _ = os.path.splitext(args.output)
        path = base + ".run.json"
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "output": os.path.abspath(args.output),
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "num_frames_mode": getattr(args, "num_frames_mode", "explicit"),
        "mux_fps": getattr(args, "mux_fps", 30),
        "num_inference_steps": args.num_inference_steps,
        "text_guidance_scale": args.text_guidance_scale,
        "audio_guidance_scale": args.audio_guidance_scale,
        "embedding_fps": getattr(args, "_resolved_embedding_fps", None),
        "audio_embedding_fps_cli": getattr(args, "audio_embedding_fps", None),
        "embedding_fps_auto": bool(getattr(args, "embedding_fps_auto", False)),
        "identity_id": args.identity_id,
        "identity_strength": args.identity_strength,
        "identity_bank_path": args.identity_bank_path,
        "use_cfg_zero": bool(getattr(args, "use_cfg_zero", False)),
        "export_crf": getattr(args, "export_crf", None),
        "high_quality_save": bool(getattr(args, "high_quality_save", False)),
        "mouth_mask": args.mouth_mask,
        "preset_hint": getattr(args, "preset_hint", None),
    }
    if frame_budget:
        payload["frame_budget"] = frame_budget
    compiler_meta = getattr(args, "_prompt_compiler_meta", None)
    if compiler_meta:
        payload["prompt_compiler"] = compiler_meta
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[run-metadata] wrote {path}")


def resolve_imagine_speak_text(args: argparse.Namespace, *, source_user_text: str = "") -> str:
    """
    TTS line for imagine_i2v: explicit --speak_text, else short user intent, else --audio forbidden path.
    """
    explicit = (getattr(args, "speak_text", None) or "").strip()
    if explicit:
        return explicit
    if getattr(args, "audio", None):
        raise ValueError("imagine_i2v generates speech internally; omit --audio or use --mode audio_i2v")
    source = (source_user_text or args.prompt or "").strip()
    if not source:
        raise ValueError("imagine_i2v requires --prompt or --speak_text")
    if len(source) > 320:
        raise ValueError(
            "imagine_i2v: user prompt too long for auto TTS; pass --speak_text with the spoken line"
        )
    return source


def synthesize_imagine_wav(args: argparse.Namespace, speak_text: str) -> Tuple[str, bool]:
    """TTS for imagine_i2v (temp wav)."""
    saved = dict(
        speak_text=args.speak_text,
        audio=args.audio,
    )
    try:
        args.speak_text = speak_text
        args.audio = None
        return resolve_avatar_wav_path(args)
    finally:
        args.speak_text = saved["speak_text"]
        args.audio = saved["audio"]


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


def _disable_torch_compile_for_lora_infer() -> None:
    """LoRA patches DiT forwards; torch.compile/inductor often fails on denoise (fallback eager)."""
    try:
        import torch._dynamo as dynamo

        dynamo.config.suppress_errors = True
        if hasattr(dynamo, "reset"):
            dynamo.reset()
        print("[lora] torch._dynamo suppress_errors=True (eager fallback for LoRA infer)")
    except Exception:
        pass


def maybe_load_avatar_lora(pipe, args: argparse.Namespace) -> None:
    if not getattr(args, "lora_path", None):
        return
    _disable_torch_compile_for_lora_infer()
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

    if args.mode in ("t2v", "i2v", "vc", "audio_i2v", "imagine_i2v"):
        if args.mode == "imagine_i2v":
            args._imagine_source_prompt = (args.prompt or "").strip()
            if getattr(args, "prompt_compiler", None) is None:
                args.prompt_compiler = resolve_imagine_compiler_backend(args)
        args._prompt_compiler_meta = apply_prompt_compiler(args)

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

    if args.mode == "audio_i2v":
        if not args.image:
            raise ValueError("--image is required for audio_i2v")
        if not args.audio:
            raise ValueError("--audio is required for audio_i2v")
        pipe = load_audio_i2v_pipeline(
            checkpoint_dir,
            device=device,
            torch_dtype=torch_dtype,
            audio_adapter_path=getattr(args, "audio_conditioning_adapter", None),
        )
        scale = float(getattr(args, "audio_conditioning_scale", 0.0))
        image = load_image(args.image)
        out = pipe.generate_audio_i2v(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            audio_path=args.audio,
            resolution=args.resolution,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            text_guidance_scale=args.text_guidance_scale,
            audio_conditioning_scale=scale,
            embedding_fps=getattr(args, "audio_embedding_fps", None),
        )[0]
        write_run_metadata(
            args,
            {
                "mode": "audio_i2v",
                "audio_conditioning_scale": scale,
                "audio_conditioning_adapter": getattr(args, "audio_conditioning_adapter", None),
                "num_frames": args.num_frames,
            },
        )
        save_video_numpy(out, args.output, fps=30)
        return

    if args.mode == "imagine_i2v":
        if not args.image:
            raise ValueError("--image is required for imagine_i2v")
        compiler_meta = getattr(args, "_prompt_compiler_meta", {}) or {}
        source_prompt = (
            (getattr(args, "_imagine_source_prompt", None) or "").strip()
            or (compiler_meta.get("source_user_text") or "").strip()
        )
        speak_text = resolve_imagine_speak_text(args, source_user_text=source_prompt)
        wav_path, wav_is_temp = synthesize_imagine_wav(args, speak_text)
        try:
            pipe = load_audio_i2v_pipeline(
                checkpoint_dir,
                device=device,
                torch_dtype=torch_dtype,
                audio_adapter_path=getattr(args, "audio_conditioning_adapter", None),
            )
            scale = float(getattr(args, "audio_conditioning_scale", 1.0))
            if scale == 0.0:
                scale = 1.0
            image = load_image(args.image)
            out = pipe.generate_audio_i2v(
                image=image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                audio_path=wav_path,
                resolution=args.resolution,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                text_guidance_scale=args.text_guidance_scale,
                audio_conditioning_scale=scale,
                embedding_fps=getattr(args, "audio_embedding_fps", None),
            )[0]
            write_run_metadata(
                args,
                {
                    "mode": "imagine_i2v",
                    "speak_text": speak_text,
                    "source_user_text": source_prompt,
                    "audio_conditioning_scale": scale,
                    "audio_conditioning_adapter": getattr(args, "audio_conditioning_adapter", None),
                    "prompt_compiler": getattr(args, "prompt_compiler", None),
                    "compiler_meta": compiler_meta,
                    "num_frames": args.num_frames,
                    "tts_provider": args.tts_provider,
                },
            )
            save_avatar_mp4(out, args.output, wav_path, args)
            print(f"[imagine-i2v] muxed TTS audio speak_text={speak_text[:96]!r}")
        finally:
            if wav_is_temp:
                maybe_unlink(wav_path)
        return

    pipe = load_avatar_pipeline(
        checkpoint_dir,
        variant="multi" if args.mode == "avc" else "single",
        device=device,
        torch_dtype=torch_dtype,
    )
    maybe_load_avatar_lora(pipe, args)
    configure_avatar_pipe(pipe, args)

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
        args._prompt_compiler_meta = apply_prompt_compiler(args, wav_path=wav_path)
        frame_budget = apply_avatar_frame_budget(args, pipe, wav_path)
        emb_fps = resolved_embedding_fps(args, pipe)
        image = load_image(args.image)
        use_cfg_zero = bool(getattr(args, "use_cfg_zero", False))
        try:
            if args.mode == "ai2v":
                audio_emb = build_audio_emb(
                    pipe,
                    audio_path=wav_path,
                    num_frames=args.num_frames,
                    device=device,
                    embedding_fps=emb_fps,
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
                    use_cfg_zero=use_cfg_zero,
                )[0]
                save_avatar_mp4(out, args.output, wav_path, args)
                write_run_metadata(args, frame_budget)
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
                    use_cfg_zero=use_cfg_zero,
                ):
                    frames.append(frame)
                save_avatar_mp4(np.stack(frames, axis=0), args.output, wav_path, args)
                write_run_metadata(args, frame_budget)
                if args.identity_bank_save_path:
                    pipe.save_identity_bank(args.identity_bank_save_path)
                    print(f"[identity-bank] saved to {args.identity_bank_save_path}")
        finally:
            if wav_is_temp:
                maybe_unlink(wav_path)
        return

    if args.mode == "at2v":
        wav_path, wav_is_temp = resolve_avatar_wav_path(args)
        args._prompt_compiler_meta = apply_prompt_compiler(args, wav_path=wav_path)
        frame_budget = apply_avatar_frame_budget(args, pipe, wav_path)
        emb_fps = resolved_embedding_fps(args, pipe)
        try:
            audio_emb = build_audio_emb(
                pipe,
                audio_path=wav_path,
                num_frames=args.num_frames,
                device=device,
                embedding_fps=emb_fps,
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
            save_avatar_mp4(out, args.output, wav_path, args)
            write_run_metadata(args, frame_budget)
        finally:
            if wav_is_temp:
                maybe_unlink(wav_path)
        return

    if args.mode == "avc":
        if not args.video:
            raise ValueError("--video is required for avc")
        wav_path, wav_is_temp = resolve_avatar_wav_path(args)
        args._prompt_compiler_meta = apply_prompt_compiler(args, wav_path=wav_path)
        frame_budget = apply_avatar_frame_budget(args, pipe, wav_path)
        emb_fps = resolved_embedding_fps(args, pipe)
        try:
            video = load_video(args.video)
            audio_emb = build_audio_emb(
                pipe,
                audio_path=wav_path,
                num_frames=args.num_frames,
                device=device,
                embedding_fps=emb_fps,
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
            save_avatar_mp4(out, args.output, wav_path, args)
            write_run_metadata(args, frame_budget)
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
