"""
Shape-compatible audio encoder stub for Phase C training.

Output contract matches wav2vec stack in ``get_audio_embedding``:
  ``[T, 12, 768]`` (time × hidden_layers × dim).

Until trained weights exist, delegates to the pipeline wav2vec path via inject hook.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from arachne_x.pipeline_arachne_x_video_avatar import ArachneXVideoAvatarPipeline


def resolve_audio_encoder_backend() -> str:
    return (os.environ.get("ARACHNE_AUDIO_ENCODER") or "wav2vec").strip().lower()


def encode_avatar_audio(
    pipe: "ArachneXVideoAvatarPipeline",
    speech_array: np.ndarray,
    *,
    fps: float,
    device: str,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Dispatch audio encoding. ``nullxes`` uses trained plate when loaded; else wav2vec.
    """
    backend = resolve_audio_encoder_backend()
    if backend == "nullxes":
        enc = getattr(pipe, "_nullxes_audio_encoder", None)
        if enc is not None and getattr(enc, "is_loaded", False):
            return enc.encode(speech_array, fps=fps, device=device, sample_rate=sample_rate)
    return _encode_wav2vec_baseline(pipe, speech_array, fps=fps, device=device, sample_rate=sample_rate)


def _encode_wav2vec_baseline(
    pipe: "ArachneXVideoAvatarPipeline",
    speech_array: np.ndarray,
    *,
    fps: float,
    device: str,
    sample_rate: int,
) -> torch.Tensor:
    from einops import rearrange

    audio_duration = len(speech_array) / sample_rate
    video_length = audio_duration * fps
    speech_array = pipe._loudness_norm(speech_array, sample_rate)
    if not getattr(pipe, "skip_audio_noise_floor", False):
        speech_array = pipe._add_noise_floor(speech_array)
    speech_array = pipe._smooth_transients(speech_array)
    audio_feature = np.squeeze(
        pipe.wav2vec_feature_extractor(speech_array, sampling_rate=sample_rate).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device).unsqueeze(0)
    with pipe.metrics.timeit("wav2vec_encode"):
        embeddings = pipe.audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    return rearrange(audio_emb, "b s d -> s b d").contiguous()


class NullxesAudioEncoder:
    """
    Trainable plate placeholder. Load ``safetensors`` into ``self.core`` when ready.

    ``is_loaded`` remains False until weights are present — pipeline falls back to wav2vec.
    """

    is_loaded: bool = False

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self.weights_path = weights_path
        self.core: Any = None
        if weights_path and os.path.isfile(weights_path):
            self._try_load(weights_path)

    def _try_load(self, path: str) -> None:
        del path
        # Phase C train delivers weights; infer enables ARACHNE_AUDIO_ENCODER=nullxes
        self.is_loaded = False

    def encode(
        self,
        speech_array: np.ndarray,
        *,
        fps: float,
        device: str,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        if not self.is_loaded or self.core is None:
            raise RuntimeError("NullxesAudioEncoder weights not loaded")
        raise NotImplementedError("Nullxes audio plate training not shipped yet")
