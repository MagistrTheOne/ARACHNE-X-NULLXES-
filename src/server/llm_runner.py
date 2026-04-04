"""HuggingFace causal LM chat completion (Qwen-compatible chat templates)."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_tok = None
_model = None
_primary_key: Optional[tuple[str, str]] = None
_fallback_tok = None
_fallback_model = None
_fallback_key: Optional[tuple[str, str]] = None


def _load_primary(cfg: Dict[str, Any]):
    global _tok, _model, _primary_key
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = str(cfg.get("model_id") or "Qwen/Qwen2.5-0.5B-Instruct")
    dm = str(cfg.get("device_map") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    key = (mid, dm)
    with _lock:
        if _tok is not None and _model is not None and _primary_key == key:
            return _tok, _model
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            mid,
            device_map=dm if dm != "cpu" else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if dm == "cpu":
            _model = _model.to("cpu")
        _primary_key = key
        logger.info("LLM primary loaded: %s", mid)
        return _tok, _model


def _load_fallback(cfg: Dict[str, Any]):
    global _fallback_tok, _fallback_model, _fallback_key
    fb = (cfg.get("fallback_model_id") or "").strip()
    if not fb:
        return None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dm = str(cfg.get("device_map") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    key = (fb, dm)
    with _lock:
        if _fallback_tok is not None and _fallback_model is not None and _fallback_key == key:
            return _fallback_tok, _fallback_model
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _fallback_tok = AutoTokenizer.from_pretrained(fb, trust_remote_code=True)
        _fallback_model = AutoModelForCausalLM.from_pretrained(
            fb,
            device_map=dm if dm != "cpu" else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if dm == "cpu":
            _fallback_model = _fallback_model.to("cpu")
        _fallback_key = key
        logger.info("LLM fallback loaded: %s", fb)
        return _fallback_tok, _fallback_model


def generate_reply_sync(
    messages: List[dict[str, str]],
    cfg: Dict[str, Any],
    *,
    system_prompt: str,
    emotion_hint: str = "",
) -> str:
    """
    Generate assistant reply text. Uses primary model; on failure uses fallback if configured.
    """
    max_new = int(cfg.get("max_new_tokens", 512))
    tok, model = _load_primary(cfg)
    msgs = []
    if system_prompt.strip():
        base = system_prompt.strip()
        if emotion_hint:
            base = base + "\nUser affect (numeric hint): " + emotion_hint
        msgs.append({"role": "system", "content": base})
    msgs.extend(messages)
    try:
        return _run_generate(tok, model, msgs, max_new)
    except Exception as e:
        logger.warning("LLM primary failed: %s", e)
        ftok, fmodel = _load_fallback(cfg)
        if fmodel is None:
            raise
        return _run_generate(ftok, fmodel, msgs, max_new)


def _run_generate(tok, model, msgs: List[dict[str, str]], max_new: int) -> str:
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"
    inputs = tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    pad_id = tok.pad_token_id or tok.eos_token_id
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pad_id,
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    text = tok.decode(gen, skip_special_tokens=True).strip()
    return text or "I'm here. How can I help?"
