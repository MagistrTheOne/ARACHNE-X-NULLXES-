import argparse

from arachne_x.tts.realtime import DEFAULT_MICRO_TURN_SECONDS
from arachne_x.weights_resolve import add_resolve_args
from arachne_x.runtime.inference_engine import execute_infer


def main():
    parser = argparse.ArgumentParser(description="ARACHNE-X inference entrypoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["t2v", "i2v", "vc", "ai2v", "at2v", "avc", "streaming_ai2v", "enroll_identity"],
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument(
        "--speak_text",
        type=str,
        default=None,
        help="Synthesize speech via --tts_provider and use the WAV as avatar audio conditioning (with --audio takes precedence).",
    )
    parser.add_argument(
        "--tts_provider",
        type=str,
        default="qwen",
        help="TTS backend when using --speak_text (qwen | audiodit; legacy alias longcat_audiodit). See requirements-tts.txt / requirements-audiodit.txt",
    )
    parser.add_argument(
        "--tts_model",
        type=str,
        default=None,
        help="HF id or local path (qwen default: Qwen3-TTS-CustomVoice; audiodit default: meituan-longcat/LongCat-AudioDiT-1B HF weights).",
    )
    parser.add_argument(
        "--tts_device_map",
        type=str,
        default=None,
        help='TTS device: Qwen ``device_map`` or AudioDiT ``.to(device)`` (default: cuda:0 if CUDA else cpu).',
    )
    parser.add_argument("--tts_language", type=str, default="English")
    parser.add_argument("--tts_speaker", type=str, default="Ryan")
    parser.add_argument("--tts_instruct", type=str, default=None, help="Optional style instruct (Qwen 1.7B CustomVoice).")
    parser.add_argument(
        "--tts_attn",
        type=str,
        default=None,
        help='Attention impl for Qwen3TTSModel (e.g. flash_attention_2, sdpa). Default: auto.',
    )
    parser.add_argument("--audiodit_nfe", type=int, default=16, help="AudioDiT ODE steps (only audiodit / longcat_audiodit).")
    parser.add_argument(
        "--audiodit_guidance_strength",
        type=float,
        default=4.0,
        help="AudioDiT CFG/APG strength (only audiodit / longcat_audiodit).",
    )
    parser.add_argument(
        "--audiodit_guidance_method",
        type=str,
        default="cfg",
        choices=["cfg", "apg"],
        help="AudioDiT guidance method (only audiodit / longcat_audiodit).",
    )
    parser.add_argument(
        "--audiodit_prompt_audio",
        type=str,
        default=None,
        help="Optional reference WAV for voice cloning (only audiodit / longcat_audiodit; requires --audiodit_prompt_text).",
    )
    parser.add_argument(
        "--audiodit_prompt_text",
        type=str,
        default=None,
        help="Transcript of --audiodit_prompt_audio (only audiodit / longcat_audiodit).",
    )
    parser.add_argument(
        "--audiodit_seed",
        type=int,
        default=1024,
        help="RNG seed for AudioDiT (only audiodit / longcat_audiodit).",
    )
    parser.add_argument(
        "--audio_chunk_sec",
        type=float,
        default=DEFAULT_MICRO_TURN_SECONDS,
        help=f"Streaming avatar: fixed chunk duration for audio micro-turns (default {DEFAULT_MICRO_TURN_SECONDS}s).",
    )
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument(
        "--num_frames_mode",
        type=str,
        default="explicit",
        choices=["explicit", "sync", "duration", "min"],
        help="Avatar: explicit=use --num_frames; sync=max lipsync window; duration=match audio @ mux_fps; min=min(sync,duration).",
    )
    parser.add_argument(
        "--mux_fps",
        type=int,
        default=30,
        help="Mux FPS for avatar MP4 export and duration frame budget.",
    )
    parser.add_argument(
        "--audio_embedding_fps",
        type=float,
        default=None,
        help="Wav2Vec embedding temporal fps (default 16*vae_stride, typically 64).",
    )
    parser.add_argument(
        "--embedding_fps_auto",
        action="store_true",
        help="Raise embedding fps when num_frames exceeds sync cap (long clips, no retrain).",
    )
    parser.add_argument("--num_cond_frames", type=int, default=13)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--text_guidance_scale", type=float, default=4.0)
    parser.add_argument("--audio_guidance_scale", type=float, default=4.0)
    parser.add_argument("--identity_id", type=int, default=None)
    parser.add_argument("--identity_strength", type=float, default=1.0)
    parser.add_argument("--identity_negative_strength", type=float, default=0.0)
    parser.add_argument("--identity_update_bank", action="store_true")
    parser.add_argument("--identity_update_momentum", type=float, default=0.25)
    parser.add_argument("--identity_bank_path", type=str, default=None)
    parser.add_argument("--identity_bank_save_path", type=str, default=None)
    parser.add_argument("--identity_bank_load_strict", action="store_true")
    parser.add_argument("--emotion_id", type=str, default=None)
    parser.add_argument("--emotion_intensity", type=float, default=0.0)
    parser.add_argument("--emotion_guidance_scale", type=float, default=0.0)
    parser.add_argument("--mouth_mask", type=str, default=None)
    parser.add_argument(
        "--hybrid_mouth_strength",
        type=float,
        default=None,
        help="Hybrid mouth renderer strength (default pipe 0.35). Requires --mouth_mask.",
    )
    parser.add_argument(
        "--hybrid_temporal_alpha",
        type=float,
        default=None,
        help="Hybrid seam temporal blend alpha (default pipe 0.70).",
    )
    parser.add_argument(
        "--no_hybrid_renderer",
        action="store_true",
        help="Disable hybrid mouth postprocess even if --mouth_mask is set.",
    )
    parser.add_argument("--disable_phoneme_conditioning", action="store_true")
    parser.add_argument("--phoneme_stream_scale", type=float, default=None)
    parser.add_argument(
        "--skip_audio_noise_floor",
        action="store_true",
        help="Skip synthetic noise floor in get_audio_embedding (clean studio WAV).",
    )
    parser.add_argument(
        "--use_cfg_zero",
        action="store_true",
        help="CFG-zero style scale on text branch in generate_ai2v (experimental A/B).",
    )
    parser.add_argument(
        "--export_crf",
        type=int,
        default=None,
        help="Final ffmpeg libx264 CRF for avatar MP4 mux (e.g. 18).",
    )
    parser.add_argument(
        "--high_quality_save",
        action="store_true",
        help="Lossless-ish final mux (libx264 crf 0, veryslow).",
    )
    parser.add_argument(
        "--run_metadata_json",
        type=str,
        default=None,
        help="Write run metadata JSON (default: <output>.run.json).",
    )
    parser.add_argument(
        "--no_run_metadata",
        action="store_true",
        help="Do not write run metadata sidecar.",
    )
    parser.add_argument(
        "--preset_hint",
        type=str,
        default=None,
        help="Optional label stored in run metadata (e.g. elena_sync).",
    )
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional .safetensors LoRA for avatar DiT (after train_lora_avatar.py).",
    )
    parser.add_argument("--lora_key", type=str, default="train", help="LoRA slot name for load_lora/enable_loras")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank (must match training). If omitted, try lora_train_meta.json beside --lora_path, else 128.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha (must match training). If omitted, try meta JSON, else 64.",
    )
    parser.add_argument(
        "--lora_meta_json",
        type=str,
        default=None,
        help="Optional path to lora_train_meta.json (overrides auto-discovery next to --lora_path).",
    )
    add_resolve_args(parser)
    args = parser.parse_args()
    execute_infer(args)


if __name__ == "__main__":
    main()
