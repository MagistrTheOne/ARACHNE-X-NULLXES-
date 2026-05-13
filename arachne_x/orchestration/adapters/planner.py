from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from ..presets import get_character_preset
from ..subprocess_utils import run_python_script, write_json


PLANNER_SCRIPT = r'''
import json
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
attn = cfg.get("attn_implementation") or "sdpa"

tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_path"],
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    attn_implementation=attn,
)
planner_lora_path = (cfg.get("planner_lora_path") or "").strip()
if planner_lora_path:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, planner_lora_path)
    model.eval()

system = """You are the NULLXES FURIA-EIDOLON local orchestrator planner.
Return only valid JSON. No markdown. No comments.
Meg Null must be written as Meg Null. NULLXES must be written exactly as NULLXES.
Correct ASR mistakes such as Null Access, Nullexes, Nowx EES, Magnol.
The visual must be corporate, cinematic, non-explicit, safe for work.
Use short, factual, executive English. Avoid generic AI hype, purple prose, overpromising, and vague marketing filler.
If user_text specifies Employee name, Role/Title, Organization/Company, or Visual/Appearance, copy those fields into employee and adapt the reply.
The video prompt must preserve the character preset and may only add small scene/action details from user intent.
Avoid holograms, floating interfaces, neon overload, readable text, logos, open mouth, teeth."""

user = {
    "character": cfg["character"],
    "user_text": cfg["user_text"],
    "session_context": cfg.get("session_context") or [],
    "reply_slots": {
        "role": "use explicit employee name, role, and organization from user_text when present",
        "tone": "calm, precise, confident",
        "length": "1-2 short sentences",
        "must_avoid": ["neuroslop", "generic AI assistant phrasing", "unverifiable claims"],
    },
    "video_slots": {
        "base_visual_source": "character_preset",
        "allowed_variation": "subtle posture, camera, lighting, or prop details only",
        "must_avoid": ["holograms", "floating screens", "readable text", "logos", "open mouth", "teeth"],
    },
    "character_preset": cfg["character_preset"],
    "video_profile": cfg["video_profile"],
    "required_schema": cfg["required_schema"],
}

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
]
chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([chat], return_tensors="pt").to(model.device)
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=700,
        temperature=0.3,
        top_p=0.8,
        do_sample=True,
    )
raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("Planner did not return JSON: " + text[:500])
    return json.loads(match.group(0))

plan = extract_json(raw)
json.dump({"raw": raw, "plan": plan}, open(cfg["result_path"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
'''


def fallback_plan(
    *,
    character: str,
    user_text: str,
    video_profile: str,
    enable_tts: bool,
    enable_video: bool,
    safety_mode: str,
) -> Dict[str, object]:
    preset = get_character_preset(character)
    reply = (
        "Hello. I am Megan from NULLXES, and I am ready to coordinate the next phase with calm precision."
        if character == "megan"
        else user_text
    )
    return {
        "character": character,
        "normalized_user_text": user_text,
        "reply_text": reply,
        "video": {
            "enabled": enable_video,
            "mode": "t2v",
            "profile": video_profile,
            "positive_prompt": preset["positive_prompt"],
            "negative_prompt": preset["negative_prompt"],
            "seed": 778,
        },
        "tts": {
            "enabled": enable_tts,
            "speaker": preset["speaker"],
            "language": preset["language"],
            "instruct": preset["tts_instruct"],
        },
        "safety": {"mode": safety_mode, "allowed": True, "notes": ["fallback_plan"]},
    }


def run_planner(
    *,
    python_bin: str,
    work_dir: str | Path,
    model_path: str,
    character: str,
    user_text: str,
    video_profile: str,
    enable_tts: bool,
    enable_video: bool,
    safety_mode: str,
    attn_implementation: str,
    session_context: list[str] | None = None,
    planner_lora_path: str | None = None,
    timeout_sec: float | None = None,
    retries: int = 0,
) -> Tuple[Dict[str, object], float]:
    required_schema = {
        "character": "string",
        "employee": {
            "name": "string, optional, copy from user_text if provided",
            "role": "string, optional, copy from user_text if provided",
            "organization": "string, optional, copy from user_text if provided",
            "visual_description": "string, optional, copy from user_text if provided",
        },
        "normalized_user_text": "string",
        "reply_text": "string",
        "video": {
            "enabled": "boolean",
            "mode": "t2v",
            "profile": video_profile,
            "positive_prompt": "string",
            "negative_prompt": "string",
            "seed": "integer",
        },
        "tts": {
            "enabled": "boolean",
            "speaker": "string",
            "language": "English",
            "instruct": "string",
        },
        "safety": {"mode": safety_mode, "allowed": "boolean", "notes": ["string"]},
    }
    fallback = fallback_plan(
        character=character,
        user_text=user_text,
        video_profile=video_profile,
        enable_tts=enable_tts,
        enable_video=enable_video,
        safety_mode=safety_mode,
    )
    write_json(Path(work_dir) / "fallback_action_plan.json", fallback)
    try:
        result, elapsed = run_python_script(
            python_bin=python_bin,
            script_text=PLANNER_SCRIPT,
            config={
                "model_path": model_path,
                "character": character,
                "user_text": user_text,
                "video_profile": video_profile,
                "character_preset": get_character_preset(character),
                "required_schema": required_schema,
                "attn_implementation": attn_implementation,
                "session_context": session_context or [],
                "planner_lora_path": planner_lora_path or "",
            },
            work_dir=work_dir,
            name="planner",
            timeout_sec=timeout_sec,
            retries=retries,
        )
        plan = result.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("planner result missing plan")
        return plan, elapsed
    except Exception:
        return fallback, 0.0
