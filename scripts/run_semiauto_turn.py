from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arachne_x.orchestration import TurnInput, run_turn


def _default_out(character: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_character = (character or "megan").strip().lower().replace(" ", "_")
    return f"output/turn_{safe_character}_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="NULLXES FURIA-EIDOLON local semiautomatic turn runner")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="User text input. If omitted, use --audio.")
    src.add_argument("--audio", type=str, help="Input audio path for ASR.")
    parser.add_argument("--character", type=str, default="megan")
    parser.add_argument("--out", type=str, default=None, help="Turn output directory.")
    parser.add_argument("--safety", type=str, default="prod", choices=["prod", "redteam"])
    parser.add_argument("--video-profile", type=str, default="fast_distill_9x16")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--repo-root", type=str, default=str(ROOT))
    parser.add_argument("--video-python", type=str, default=None, help="Python in .venv for video.")
    parser.add_argument("--stage3-python", type=str, default=None, help="Python in .venv_stage3 for ASR/LLM/TTS.")
    parser.add_argument("--whisper-model", type=str, default="/workspace/ARACHNE-X/weights/openai-whisper-large-v3-turbo")
    parser.add_argument("--llm-model", type=str, default="/workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--tts-model",
        type=str,
        default="/workspace/ARACHNE-X/weights/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )
    parser.add_argument("--video-checkpoint", type=str, default="/workspace/ARACHNE-X/weights/ARACHNE-X-ULTRA-VIDEO")
    parser.add_argument("--attn", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa"])
    args = parser.parse_args()

    turn = TurnInput(
        text=args.text,
        audio_path=args.audio,
        character=args.character,
        output_dir=args.out or _default_out(args.character),
        safety_mode=args.safety,
        video_profile=args.video_profile,
        enable_tts=not args.no_tts,
        enable_video=not args.no_video,
    )
    manifest = run_turn(
        turn,
        repo_root=args.repo_root,
        video_python=args.video_python,
        stage3_python=args.stage3_python,
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        video_checkpoint=args.video_checkpoint,
        attn_implementation=args.attn,
    )
    print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
