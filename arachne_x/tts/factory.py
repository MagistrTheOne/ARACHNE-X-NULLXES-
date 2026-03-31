from __future__ import annotations

from typing import Optional

from .base import SpeechSynthesizer


def create_speech_synthesizer(
    provider: str,
    *,
    model_id: Optional[str] = None,
    device_map: Optional[str] = None,
    language: str = "English",
    speaker: str = "Ryan",
    instruct: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> SpeechSynthesizer:
    """
    Factory for TTS backends. Core ``requirements.txt`` does not pin provider packages.

    Supported ``provider`` values:
    - ``qwen``: Qwen3 CustomVoice via ``qwen-tts`` (install ``requirements-tts.txt``).
    """
    p = (provider or "").strip().lower()
    if p == "qwen":
        import torch

        from .qwen import Qwen3CustomVoiceSynthesizer, QwenCustomVoiceSettings

        dm = device_map or ("cuda:0" if torch.cuda.is_available() else "cpu")
        mid = model_id or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        settings = QwenCustomVoiceSettings(
            model_id=mid,
            device_map=dm,
            language=language,
            speaker=speaker,
            instruct=instruct,
            attn_implementation=attn_implementation,
        )
        return Qwen3CustomVoiceSynthesizer(settings)
    raise ValueError(f"Unknown --tts_provider {provider!r}. Supported: qwen")
