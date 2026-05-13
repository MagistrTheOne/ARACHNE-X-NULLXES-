from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from ..subprocess_utils import run_python_script


ASR_SCRIPT = r"""
import json
import re
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
generate_kwargs = {"language": cfg.get("language", "english"), "task": "transcribe"}
initial_prompt = (cfg.get("initial_prompt") or "").strip()
if initial_prompt:
    try:
        prompt_ids = processor.get_prompt_ids(initial_prompt, return_tensors="pt")
        generate_kwargs["prompt_ids"] = prompt_ids.to(device) if hasattr(prompt_ids, "to") else prompt_ids
    except TypeError:
        generate_kwargs["prompt_ids"] = processor.get_prompt_ids(initial_prompt)

out = asr_pipe(cfg["audio_path"], generate_kwargs=generate_kwargs)
raw_text = out["text"].strip()


def normalize_brand_terms(text):
    replacements = [
        (r"\bnull\s+access\b", "NULLXES"),
        (r"\bnull\s*x\s*e\s*s\b", "NULLXES"),
        (r"\bnull\s*xes\b", "NULLXES"),
        (r"\bnullexes\b", "NULLXES"),
        (r"\bnullxes\b", "NULLXES"),
        (r"\bnowx\s*e+e?s\b", "NULLXES"),
        (r"\bnolix'?s\b", "NULLXES"),
        (r"\bnollix'?s\b", "NULLXES"),
        (r"\bnulix'?s\b", "NULLXES"),
        (r"\bno\s*licks\b", "NULLXES"),
        (r"\bmeg\s*null\b", "Meg Null"),
        (r"\bmegan\s+null\b", "Meg Null"),
        (r"\bmagnol\b", "Meg Null"),
        (r"\bforia[\s,.-]+it[\s,.-]+alone\b", "FURIA-EIDOLON"),
        (r"\bforia[\s,.-]+eidolon\b", "FURIA-EIDOLON"),
        (r"\bfury[\s,.-]+eidolon\b", "FURIA-EIDOLON"),
        (r"\bfuria[\s,.-]+eidolon\b", "FURIA-EIDOLON"),
        (r"\bfury[\s,.-]+eidelon\b", "FURIA-EIDOLON"),
        (r"\beidolon\b", "EIDOLON"),
    ]
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bN\s*U\s*L\s*L\s*X\s*E\s*S\b", "NULLXES", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bF\s*U\s*R\s*I\s*A\s*[-\s]*E\s*I\s*D\s*O\s*L\s*O\s*N\b", "FURIA-EIDOLON", normalized, flags=re.IGNORECASE)
    return normalized.strip()


text = normalize_brand_terms(raw_text)
open(cfg["raw_text_path"], "w", encoding="utf-8").write(raw_text)
open(cfg["text_path"], "w", encoding="utf-8").write(text)
json.dump(
    {
        "text": text,
        "raw_text": raw_text,
        "text_path": cfg["text_path"],
        "raw_text_path": cfg["raw_text_path"],
        "initial_prompt": initial_prompt,
    },
    open(cfg["result_path"], "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)
"""


DEFAULT_INITIAL_PROMPT = (
    "NULLXES is spelled N U L L X E S. FURIA-EIDOLON is spelled F U R I A, hyphen, "
    "E I D O L O N. Meg Null is the digital executive from NULLXES."
)


def run_asr(
    *,
    python_bin: str,
    work_dir: str | Path,
    audio_path: str,
    model_path: str,
    language: str = "english",
    initial_prompt: str = DEFAULT_INITIAL_PROMPT,
    timeout_sec: float | None = None,
    retries: int = 0,
) -> Tuple[Dict[str, object], float]:
    text_path = str(Path(work_dir) / "asr.txt")
    raw_text_path = str(Path(work_dir) / "asr_raw.txt")
    return run_python_script(
        python_bin=python_bin,
        script_text=ASR_SCRIPT,
        config={
            "audio_path": audio_path,
            "model_path": model_path,
            "language": language,
            "initial_prompt": initial_prompt,
            "text_path": text_path,
            "raw_text_path": raw_text_path,
        },
        work_dir=work_dir,
        name="asr",
        timeout_sec=timeout_sec,
        retries=retries,
    )
