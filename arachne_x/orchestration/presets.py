from __future__ import annotations

from typing import Dict


MEGAN_POSITIVE_PROMPT = (
    "Elegant brunette woman named Megan, digital executive from NULLXES, standing in a dark luxury futuristic "
    "corporate office at night, premium cinematic lighting, deep black and silver interior, thin black glasses, "
    "wearing a sleek black tailored business suit, calm closed-mouth expression, direct focused eye contact, "
    "realistic skin texture, detailed hair strands, professional executive posture, city skyline visible through "
    "large panoramic windows, soft rim light on hair and shoulders, warm overhead office lights, polished dark "
    "marble floor, minimalist premium enterprise atmosphere, slow cinematic camera push-in, smooth natural motion, "
    "photorealistic, high facial consistency, shallow depth of field, 4K commercial corporate video quality"
)

MEGAN_NEGATIVE_PROMPT = (
    "low resolution, blurry, distorted face, cartoon, anime, cgi, fantasy, holograms, holographic interfaces, "
    "floating screens, digital glow, neon overload, cyberpunk overload, sci-fi creatures, text overlay, logos, "
    "watermark, unreadable letters, crowd scenes, outdoor settings, daytime, open mouth, teeth, exaggerated smile, "
    "messy clothing, bad hands, extra fingers, plastic skin, deformed anatomy, jitter, flickering face, nudity, "
    "explicit content"
)

MEGAN_TTS_INSTRUCT = "Speak in a warm, confident, elegant female corporate voice with calm executive presence."

CHARACTER_PRESETS: Dict[str, Dict[str, str]] = {
    "megan": {
        "display_name": "Megan / Meg Null",
        "speaker": "Serena",
        "language": "English",
        "tts_instruct": MEGAN_TTS_INSTRUCT,
        "positive_prompt": MEGAN_POSITIVE_PROMPT,
        "negative_prompt": MEGAN_NEGATIVE_PROMPT,
        "default_video_profile": "fast_distill_9x16",
        "identity_loras": [],
        "style_rules": (
            "Keep Megan photorealistic, closed-mouth, corporate, restrained, and premium. "
            "Do not add holograms, floating UI, readable brand text, neon overload, or exaggerated expression."
        ),
    }
}

VIDEO_PROFILES: Dict[str, Dict[str, object]] = {
    "fast_distill_9x16": {
        "height": 832,
        "width": 480,
        "num_frames": 93,
        "num_inference_steps": 16,
        "guidance_scale": 1.0,
        "use_distill": True,
        "fps": 30,
        "crf": "18",
        "lora_key": "cfg_step_lora",
        "lora_file": "lora/cfg_step_lora.safetensors",
    },
    "megan_identity_fast_9x16": {
        "height": 832,
        "width": 480,
        "num_frames": 93,
        "num_inference_steps": 16,
        "guidance_scale": 1.0,
        "use_distill": True,
        "fps": 30,
        "crf": "18",
        "loras": [
            {"file": "lora/cfg_step_lora.safetensors", "key": "cfg_step_lora"},
        ],
    },
    "quality_480p_9x16": {
        "height": 832,
        "width": 480,
        "num_frames": 93,
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
        "use_distill": False,
        "fps": 30,
        "crf": "18",
    },
}


def get_character_preset(character: str) -> Dict[str, str]:
    return CHARACTER_PRESETS.get((character or "").strip().lower(), CHARACTER_PRESETS["megan"])


def get_video_profile(profile: str) -> Dict[str, object]:
    return VIDEO_PROFILES.get((profile or "").strip(), VIDEO_PROFILES["fast_distill_9x16"])
