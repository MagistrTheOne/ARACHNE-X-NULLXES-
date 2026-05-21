from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch

from arachne_x.prompt_compiler.templates import is_chinese_text, merge_avatar_defaults

logger = logging.getLogger(__name__)

_gemma_model = None
_gemma_tokenizer = None
_gemma_model_id: Optional[str] = None

GEMMA_AVATAR_SYS_EN = (
    "You rewrite short user intents into detailed video scene descriptions for "
    "audio-driven talking-head avatar generation. Requirements: "
    "1) The subject faces the camera and speaks with accurate lip sync. "
    "2) Static camera, fixed framing, no zoom or pan. "
    "3) Describe only visible actions and appearance; no speculation. "
    "4) Output English only, 80-180 words, no quotes around the whole answer."
)

GEMMA_AVATAR_SYS_ZH = (
    "将用户简短意图改写为音频驱动数字人说话视频的画面描述。"
    "人物面向镜头、口型与语音同步；固定机位无推拉摇移；"
    "只描述可见动作与外观；中文80-180字，整段回答不加引号。"
)

GEMMA_IMAGINE_SYS_EN = (
    "You rewrite short user intents into detailed image-to-video scene descriptions. "
    "The clip includes synchronized speech and ambient context implied by the user. "
    "Requirements: subject faces camera; describe visible motion, expression, and setting; "
    "static camera; English only, 80-180 words, no quotes around the whole answer."
)

GEMMA_IMAGINE_SYS_ZH = (
    "将用户简短意图改写为图生视频画面描述，包含与语音同步的自然动作和场景。"
    "人物面向镜头；描述可见动作、表情与环境；固定机位；中文80-180字。"
)


def _gemma_model_path() -> str:
    return (os.environ.get("ARACHNE_GEMMA_MODEL") or "google/gemma-2-2b-it").strip()


def _load_gemma():
    global _gemma_model, _gemma_tokenizer, _gemma_model_id
    model_id = _gemma_model_path()
    if _gemma_model is not None and _gemma_tokenizer is not None and _gemma_model_id == model_id:
        return _gemma_model, _gemma_tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Gemma prompt compiler requires CUDA (RunPod GPU path).")

    dtype = torch.bfloat16
    model_path = Path(model_id)
    local_only = model_path.is_dir() and (model_path / "config.json").is_file()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_only,
    )
    model.eval()
    _gemma_model = model
    _gemma_tokenizer = tokenizer
    _gemma_model_id = model_id
    logger.info("prompt_compiler gemma loaded model_id=%s", model_id)
    return model, tokenizer


def expand_with_gemma(
    user_text: str,
    *,
    mode: str,
    locale: str = "auto",
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """RunPod path: local Gemma instruction expansion."""
    t0 = time.perf_counter()
    text = (user_text or "").strip()
    if not text:
        return "", (time.perf_counter() - t0) * 1000.0

    model, tokenizer = _load_gemma()
    use_zh = locale == "zh" or (locale == "auto" and is_chinese_text(text))
    if mode in ("imagine_i2v", "audio_i2v", "i2v"):
        sys_prompt = GEMMA_IMAGINE_SYS_ZH if use_zh else GEMMA_IMAGINE_SYS_EN
    else:
        sys_prompt = GEMMA_AVATAR_SYS_ZH if use_zh else GEMMA_AVATAR_SYS_EN

    # Gemma-2 chat templates reject {"role": "system"} — fold instructions into user turn.
    user_content = f"{sys_prompt.strip()}\n\n{text.strip()}"
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_content}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        prompt_ids = tokenizer(
            f"{user_content}\n\nAssistant:",
            return_tensors="pt",
        ).input_ids

    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    with torch.inference_mode():
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, prompt_ids.shape[-1] :]
    expanded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    pos, _ = merge_avatar_defaults(expanded, "", mode=mode, locale=locale)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "prompt_compiler gemma mode=%s latency_ms=%.1f chars_in=%d chars_out=%d",
        mode,
        latency_ms,
        len(text),
        len(pos),
    )
    return pos, latency_ms
