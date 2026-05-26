"""Frozen-base audio conditioning for VIDEO DiT (experimental audio_i2v)."""

from .adapter import AudioConditioningAdapter, AudioConditioningAdapterConfig
from .audio_encode import AudioEncoderRuntime, build_windowed_audio_emb, encode_wav2vec_audio
from .state_dict import load_audio_conditioning_adapter, save_audio_conditioning_adapter
from .wrapped_dit import AudioConditionedVideoDiTWrapper

__all__ = [
    "AudioConditioningAdapter",
    "AudioConditioningAdapterConfig",
    "AudioConditionedVideoDiTWrapper",
    "AudioEncoderRuntime",
    "build_windowed_audio_emb",
    "encode_wav2vec_audio",
    "load_audio_conditioning_adapter",
    "save_audio_conditioning_adapter",
]
