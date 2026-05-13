from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from ..subprocess_utils import run_python_script


ASR_SCRIPT = r"""
import json
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    cfg["model_path"],
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
).to(device)
processor = AutoProcessor.from_pretrained(cfg["model_path"])
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=dtype,
    device=device,
)
out = asr_pipe(cfg["audio_path"], generate_kwargs={"language": cfg.get("language", "english"), "task": "transcribe"})
text = out["text"].strip()
open(cfg["text_path"], "w", encoding="utf-8").write(text)
json.dump({"text": text, "text_path": cfg["text_path"]}, open(cfg["result_path"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
"""


def run_asr(
    *,
    python_bin: str,
    work_dir: str | Path,
    audio_path: str,
    model_path: str,
    language: str = "english",
    timeout_sec: float | None = None,
    retries: int = 0,
) -> Tuple[Dict[str, object], float]:
    text_path = str(Path(work_dir) / "asr.txt")
    return run_python_script(
        python_bin=python_bin,
        script_text=ASR_SCRIPT,
        config={
            "audio_path": audio_path,
            "model_path": model_path,
            "language": language,
            "text_path": text_path,
        },
        work_dir=work_dir,
        name="asr",
        timeout_sec=timeout_sec,
        retries=retries,
    )
