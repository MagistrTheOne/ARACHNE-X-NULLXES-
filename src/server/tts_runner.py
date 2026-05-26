"""TTS to float32 mono 16 kHz PCM (edge-tts or espeak-ng)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def synthesize_pcm_f32_16k(text: str, tts_cfg: Dict[str, Any]) -> np.ndarray:
    """
    Synthesize speech; returns float32 mono 16 kHz. Raises on failure.
    """
    text = (text or "").strip()
    if not text:
        return np.zeros((0,), dtype=np.float32)

    backend = str(tts_cfg.get("backend") or "edge_tts").lower().replace("-", "_")
    sample_rate = 16000

    if backend in ("qwen3_tts", "qwen_tts", "qwen"):
        from arachne_x.tts.factory import create_speech_synthesizer

        model_id = tts_cfg.get("model_id")
        dm = tts_cfg.get("device_map") or None
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="nx_tts_")
        os.close(fd)
        try:
            syn = create_speech_synthesizer(
                "qwen",
                model_id=model_id,
                device_map=dm,
                language=str(tts_cfg.get("language") or "English"),
                speaker=str(tts_cfg.get("speaker") or "Ryan"),
                instruct=tts_cfg.get("instruct"),
            )
            syn.synthesize_to_path(text, path)
            audio, _ = librosa.load(path, sr=sample_rate, mono=True)
            return np.asarray(audio, dtype=np.float32)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    if backend in ("edge", "edge_tts", "edgetts"):
        from arachne_x.speech.providers.edge_tts import EdgeTTSSpeechSynthesizer

        voice = str(tts_cfg.get("voice") or "en-US-AriaNeural")
        rate = tts_cfg.get("rate")
        rate_s = str(rate) if rate is not None else None
        syn = EdgeTTSSpeechSynthesizer(voice=voice, rate=rate_s)
        fd, wav = tempfile.mkstemp(suffix=".wav", prefix="nx_edge_")
        os.close(fd)
        try:
            syn.synthesize_to_wav(text, Path(wav), sample_rate=sample_rate)
            audio, _ = librosa.load(wav, sr=sample_rate, mono=True)
            return np.asarray(audio, dtype=np.float32)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

    if backend in ("espeak", "espeak_ng", "espeak-ng"):
        from arachne_x.speech.providers.espeak import EspeakSpeechSynthesizer

        syn = EspeakSpeechSynthesizer(binary=tts_cfg.get("espeak_binary"))
        fd, wav = tempfile.mkstemp(suffix=".wav", prefix="nx_espeak_")
        os.close(fd)
        try:
            syn.synthesize_to_wav(text, Path(wav), sample_rate=sample_rate)
            audio, _ = librosa.load(wav, sr=sample_rate, mono=True)
            return np.asarray(audio, dtype=np.float32)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

    raise ValueError(f"Unsupported tts backend: {backend!r}")
