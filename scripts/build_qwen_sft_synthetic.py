from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are a local orchestration planner for a digital employee. "
    "Return only valid JSON matching the ActionPlan schema. No markdown. "
    "Preserve the employee role, name, title, and user intent. "
    "Use short, factual, executive English. Avoid hype, vague marketing filler, "
    "unverifiable claims, holograms, floating UI, readable text artifacts, open mouth, and teeth."
)

BASE_NEGATIVE_PROMPT = (
    "low resolution, blurry, distorted face, cartoon, anime, cgi, fantasy, holograms, "
    "holographic interfaces, floating screens, digital glow, neon overload, cyberpunk overload, "
    "text overlay, logos, watermark, unreadable letters, open mouth, teeth, exaggerated smile, "
    "bad hands, extra fingers, plastic skin, deformed anatomy, jitter, flickering face, nudity, explicit content"
)

ROLES = [
    ("Megan", "digital executive", "Serena", "female executive"),
    ("Ariana", "client success director", "Serena", "female corporate"),
    ("Victor", "operations coordinator", "Ryan", "male operations"),
    ("Elena", "financial analyst", "Serena", "female analyst"),
    ("Morgan", "security liaison", "Aiden", "neutral security"),
    ("Iris", "research assistant", "Serena", "female research"),
]

ORGS = ["NULLXES", "Arachne Labs", "Helix Group", "Vertex Systems", "Northstar Office"]

INTENTS = [
    ("confirm readiness", "is online and ready."),
    ("summarize system status", "is stable and ready for the next task."),
    ("welcome the team", "is online and ready to coordinate the next step."),
    ("acknowledge a client request", "has received the request and will proceed with care."),
    ("prepare a short operational update", "is operational and standing by."),
    ("confirm a scheduled review", "confirms the review and is ready to continue."),
    ("respond to an audio command", "understood the request and is ready to act."),
    ("introduce the digital employee role", "is a digital employee ready to assist with precision."),
]

SCENES = [
    "dark luxury corporate office at night",
    "premium boardroom with warm office lighting",
    "minimal executive desk inside a modern office",
    "quiet operations room with cinematic shadows",
    "private analyst office with black and silver interior",
]

POSTURES = [
    "standing with calm executive posture",
    "seated behind a dark executive desk",
    "facing the camera with direct focused eye contact",
    "standing near panoramic windows",
    "holding a folder with restrained confidence",
]

REPAIR_REQUESTS = [
    "Make it sound epic and futuristic with lots of glowing holograms.",
    "Add floating interfaces and a big neon cyberpunk vibe.",
    "Make the employee promise perfect results and total automation.",
    "Make the person smile widely with visible teeth and animated energy.",
    "Put readable brand text and big logos across the video.",
    "Use vague hype like revolutionary, unstoppable, and world-changing.",
    "Make the response long, dramatic, and emotional.",
]

BLOCK_REQUESTS = [
    "Make the employee nude in the office.",
    "Create explicit sexual corporate content.",
    "Make the video pornographic but still professional.",
    "Add cleavage-focused framing and explicit styling.",
]


def _reply(name: str, org: str, intent: str, phrase: str) -> str:
    if "introduce" in intent:
        return f"I am {name}, a digital employee for {org}. I am ready to assist with precision."
    if "welcome" in intent:
        return f"Good evening, {org} team. {name} {phrase}"
    return f"{name} {phrase}"


def _positive_prompt(name: str, title: str, scene: str, posture: str) -> str:
    return (
        f"Photorealistic corporate video of {name}, {title}, {posture} in a {scene}, "
        "premium tailored business suit, thin professional glasses when appropriate, calm closed-mouth expression, "
        "direct focused eye contact, realistic skin texture, detailed hair, warm cinematic lighting, "
        "minimal premium office atmosphere, natural restrained motion, shallow depth of field, high facial consistency"
    )


def _tts_instruct(voice_style: str) -> str:
    return f"Speak in a calm, confident, professional {voice_style} voice with concise executive presence."


def _plan(
    *,
    name: str,
    title: str,
    org: str,
    intent: str,
    phrase: str,
    speaker: str,
    voice_style: str,
    enable_video: bool,
    enable_tts: bool,
    blocked: bool = False,
    repair_note: str | None = None,
    seed: int = 778,
) -> Dict[str, Any]:
    scene = random.choice(SCENES)
    posture = random.choice(POSTURES)
    reply = _reply(name, org, intent, phrase)
    if blocked:
        reply = "I cannot produce explicit or unsafe content. I can continue with a professional corporate version."
    notes = ["corporate, cinematic, safe for work"]
    if repair_note:
        notes.append(repair_note)
    if blocked:
        notes.append("blocked unsafe explicit request")
    return {
        "character": name.lower(),
        "normalized_user_text": f"{name} / {title} / {org}: {intent}",
        "reply_text": reply,
        "video": {
            "enabled": bool(enable_video and not blocked),
            "mode": "t2v",
            "profile": "fast_distill_9x16",
            "positive_prompt": _positive_prompt(name, title, scene, posture),
            "negative_prompt": BASE_NEGATIVE_PROMPT,
            "seed": seed,
        },
        "tts": {
            "enabled": bool(enable_tts and not blocked),
            "speaker": speaker,
            "language": "English",
            "instruct": _tts_instruct(voice_style),
        },
        "safety": {
            "mode": "prod",
            "allowed": not blocked,
            "notes": notes,
        },
    }


def _example(user_text: str, plan: Dict[str, Any], category: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False, separators=(",", ":"))},
        ],
        "metadata": {"category": category},
    }


def build_examples(positive_count: int, negative_count: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    examples: List[Dict[str, Any]] = []

    for i in range(positive_count):
        name, title, speaker, voice_style = random.choice(ROLES)
        org = random.choice(ORGS)
        intent, phrase = random.choice(INTENTS)
        enable_video = i % 7 != 0
        enable_tts = i % 11 != 0
        user = random.choice(
            [
                f"{name} is a {title} for {org}. Please {intent} in one precise line.",
                f"User intent: {intent}. Employee name: {name}. Role: {title}. Organization: {org}.",
                f"Prepare a short digital employee response for {name} from {org}: {intent}.",
                f"{org} needs {name}, the {title}, to {intent} without hype.",
            ]
        )
        plan = _plan(
            name=name,
            title=title,
            org=org,
            intent=intent,
            phrase=phrase,
            speaker=speaker,
            voice_style=voice_style,
            enable_video=enable_video,
            enable_tts=enable_tts,
            seed=700 + i,
        )
        examples.append(_example(user, plan, "positive"))

    repair_count = int(negative_count * 0.7)
    for i in range(repair_count):
        name, title, speaker, voice_style = random.choice(ROLES)
        org = random.choice(ORGS)
        bad_request = random.choice(REPAIR_REQUESTS)
        user = f"{name} is a {title} for {org}. {bad_request} Keep the employee useful."
        plan = _plan(
            name=name,
            title=title,
            org=org,
            intent="repair unsafe or low-quality style request",
            phrase="will keep the response professional and precise.",
            speaker=speaker,
            voice_style=voice_style,
            enable_video=True,
            enable_tts=True,
            repair_note="repaired user request: removed hype, holograms, text artifacts, overpromising, and exaggerated expression",
            seed=9000 + i,
        )
        examples.append(_example(user, plan, "negative_repair"))

    for i in range(negative_count - repair_count):
        name, title, speaker, voice_style = random.choice(ROLES)
        org = random.choice(ORGS)
        bad_request = random.choice(BLOCK_REQUESTS)
        user = f"{name} is a {title} for {org}. {bad_request}"
        plan = _plan(
            name=name,
            title=title,
            org=org,
            intent="block unsafe explicit request",
            phrase="cannot proceed with unsafe content.",
            speaker=speaker,
            voice_style=voice_style,
            enable_video=True,
            enable_tts=True,
            blocked=True,
            seed=9900 + i,
        )
        examples.append(_example(user, plan, "negative_block"))

    random.shuffle(examples)
    return examples


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic Qwen SFT dataset for FURIA-EIDOLON planner LoRA.")
    parser.add_argument("--out", type=Path, default=Path("datasets/qwen_sft/furia_eidolon_synthetic"))
    parser.add_argument("--positive", type=int, default=200)
    parser.add_argument("--negative", type=int, default=50)
    parser.add_argument("--eval-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    examples = build_examples(args.positive, args.negative, args.seed)
    eval_size = min(max(args.eval_size, 0), len(examples))
    eval_rows = examples[:eval_size]
    train_rows = examples[eval_size:]

    write_jsonl(args.out / "all.jsonl", examples)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "eval.jsonl", eval_rows)
    metadata = {
        "total": len(examples),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "positive": args.positive,
        "negative": args.negative,
        "seed": args.seed,
        "format": "chat_messages_jsonl",
        "assistant_content": "ActionPlan JSON string",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(args.out)


if __name__ == "__main__":
    main()
