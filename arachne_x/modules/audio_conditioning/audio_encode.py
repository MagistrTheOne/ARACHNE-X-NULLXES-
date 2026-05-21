"""Standalone wav2vec audio encoding for audio-conditioned VIDEO i2v."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from ...audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from ...inference_frames import DEFAULT_EMBEDDING_FPS


@dataclass
class AudioEncoderRuntime:
    audio_encoder: Wav2Vec2ModelWrapper
    feature_extractor: Wav2Vec2FeatureExtractor
    device: str = "cuda"

    @classmethod
    def from_checkpoint(
        cls,
        wav2vec_path: str,
        device: str = "cuda",
    ) -> "AudioEncoderRuntime":
        audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(device)
        audio_encoder.feature_extractor._freeze_parameters()
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_path,
            local_files_only=True,
        )
        return cls(audio_encoder=audio_encoder, feature_extractor=feature_extractor, device=device)


def _loudness_norm(speech_array: np.ndarray, sample_rate: int) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(speech_array)) + 1e-8))
    target = 0.05
    if rms < 1e-6:
        return speech_array
    return speech_array * (target / rms)


def encode_wav2vec_audio(
    runtime: AudioEncoderRuntime,
    speech_array: np.ndarray,
    *,
    fps: float = DEFAULT_EMBEDDING_FPS,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Returns wav2vec hidden-state stack ``[T, 12, 768]`` (time x layers x dim).
    """
    speech_array = np.ascontiguousarray(speech_array, dtype=np.float32)
    speech_array = _loudness_norm(speech_array, sample_rate)
    audio_duration = len(speech_array) / float(sample_rate)
    video_length = audio_duration * fps
    audio_feature = np.squeeze(
        runtime.feature_extractor(speech_array, sampling_rate=sample_rate).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=runtime.device).unsqueeze(0)
    with torch.no_grad():
        embeddings = runtime.audio_encoder(
            audio_feature,
            seq_len=int(video_length),
            output_hidden_states=True,
        )
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    return rearrange(audio_emb, "b s d -> s b d").contiguous()


def build_windowed_audio_emb(
    full_audio_emb: torch.Tensor,
    num_frames: int,
    *,
    audio_window: int = 5,
    vae_stride: int = 4,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Build ``[1, T, W, S, C]`` windows aligned to VIDEO temporal stride.
    """
    if full_audio_emb.dim() != 3:
        raise ValueError(f"full_audio_emb must be [T,S,C], got {tuple(full_audio_emb.shape)}")
    audio_window = max(1, 2 * (audio_window // 2) + 1)
    dev = device or full_audio_emb.device
    indices = torch.arange(audio_window, device=dev) - (audio_window // 2)
    center_indices = (
        torch.arange(0, vae_stride * num_frames, vae_stride, device=dev).unsqueeze(1)
        + indices.unsqueeze(0)
    )
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    return full_audio_emb[center_indices][None, ...].to(dev)


def load_audio_from_path(path: str, sample_rate: int = 16000) -> np.ndarray:
    speech_array, _ = librosa.load(path, sr=sample_rate)
    return speech_array
