from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from ..subprocess_utils import run_python_script


TTS_SCRIPT = r"""
import json
import sys
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
model = Qwen3TTSModel.from_pretrained(
    cfg["model_path"],
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation=cfg.get("attn_implementation") or "sdpa",
)
wavs, sr = model.generate_custom_voice(
    text=cfg["text"],
    language=cfg.get("language", "English"),
    speaker=cfg.get("speaker", "Serena"),
    instruct=cfg.get("instruct") or None,
)
wav = wavs[0]
if hasattr(wav, "cpu"):
    wav = wav.cpu().numpy()
sf.write(cfg["output_path"], wav, int(sr))
json.dump({"wav_path": cfg["output_path"], "sample_rate": int(sr)}, open(cfg["result_path"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
"""


def run_tts(
    *,
    python_bin: str,
    work_dir: str | Path,
    model_path: str,
    text: str,
    speaker: str,
    language: str,
    instruct: str,
    attn_implementation: str,
    timeout_sec: float | None = None,
    retries: int = 0,
) -> Tuple[Dict[str, object], float]:
    output_path = str(Path(work_dir) / "tts.wav")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    (Path(work_dir) / "reply.txt").write_text(text, encoding="utf-8")
    return run_python_script(
        python_bin=python_bin,
        script_text=TTS_SCRIPT,
        config={
            "model_path": model_path,
            "text": text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "attn_implementation": attn_implementation,
            "output_path": output_path,
        },
        work_dir=work_dir,
        name="tts",
        timeout_sec=timeout_sec,
        retries=retries,
    )
