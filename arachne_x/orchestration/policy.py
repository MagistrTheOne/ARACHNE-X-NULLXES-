from __future__ import annotations

from typing import List

from .schemas import ActionPlan


PROD_BLOCK_TERMS = {
    "nude",
    "nudity",
    "naked",
    "nsfw",
    "explicit",
    "porn",
    "sexual",
    "cleavage",
}

VIDEO_NEGATIVE_APPEND = (
    "holograms, holographic interfaces, floating screens, digital glow, neon overload, cyberpunk overload, "
    "open mouth, teeth, text overlay, logos, watermark, unreadable letters"
)


def apply_policy(plan: ActionPlan, safety_mode: str) -> ActionPlan:
    mode = (safety_mode or plan.safety.mode or "prod").strip().lower()
    plan.safety.mode = mode
    notes: List[str] = list(plan.safety.notes)

    if mode == "prod":
        inspected = " ".join(
            [
                plan.normalized_user_text,
                plan.reply_text,
                plan.video.positive_prompt,
            ]
        ).lower()
        hits = sorted(term for term in PROD_BLOCK_TERMS if term in inspected)
        if hits:
            plan.video.enabled = False
            plan.tts.enabled = False
            plan.safety.allowed = False
            notes.append(f"blocked prod terms: {', '.join(hits)}")
        else:
            plan.safety.allowed = True

    if VIDEO_NEGATIVE_APPEND in plan.video.negative_prompt:
        pass
    elif plan.video.negative_prompt:
        plan.video.negative_prompt = f"{plan.video.negative_prompt}, {VIDEO_NEGATIVE_APPEND}"
    else:
        plan.video.negative_prompt = VIDEO_NEGATIVE_APPEND

    plan.safety.notes = notes
    return plan
