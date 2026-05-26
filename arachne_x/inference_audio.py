"""
Shared audio embedding windowing for avatar inference / export (matches scripts/infer.py logic).

Incremental streaming (S2): ``IncrementalStreamingAudioEmb`` encodes a short audio prefix
for chunk-0 denoise, then full utterance before later chunks — TTFF decoupled from total
audio duration when ``ARACHNE_INCREMENTAL_WAV2VEC=1``.
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

import librosa
import numpy as np
import torch

from arachne_x.inference_frames import DEFAULT_EMBEDDING_FPS, round_to_4n_plus_1
from arachne_x.runtime.chunk_stitch import slice_audio_emb_temporal

if TYPE_CHECKING:
    from arachne_x.pipeline_arachne_x_video_avatar import ArachneXVideoAvatarPipeline


def incremental_wav2vec_enabled() -> bool:
    return os.environ.get("ARACHNE_INCREMENTAL_WAV2VEC", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def default_embedding_fps(pipe: "ArachneXVideoAvatarPipeline") -> float:
    audio_stride = int(getattr(pipe, "vae_scale_factor_temporal", 4))
    audio_stride = max(audio_stride, 1)
    emb_fps = getattr(pipe, "inference_embedding_fps", None)
    if emb_fps is not None:
        return float(emb_fps)
    return float(16 * audio_stride)


def min_embedding_timesteps_for_frames(
    num_frames: int,
    *,
    audio_stride: int,
    audio_window: int = 5,
) -> int:
    """Minimum wav2vec time steps to window ``num_frames`` without edge-only clamp."""
    nf = max(1, int(num_frames))
    stride = max(1, int(audio_stride))
    window = max(1, 2 * (int(audio_window) // 2) + 1)
    max_center = stride * (nf - 1)
    return max(1, max_center + (window // 2) + 1)


def min_audio_samples_for_frames(
    num_frames: int,
    *,
    embedding_fps: float,
    sample_rate: int = 16000,
    audio_stride: int = 4,
    audio_window: int = 5,
) -> int:
    """PCM samples required so wav2vec produces enough embedding timesteps."""
    min_t = min_embedding_timesteps_for_frames(
        num_frames,
        audio_stride=audio_stride,
        audio_window=audio_window,
    )
    min_dur_sec = float(min_t) / max(float(embedding_fps), 1.0)
    return max(1, int(math.ceil(min_dur_sec * float(sample_rate))))


def _incremental_min_samples_override() -> Optional[int]:
    raw_ms = (os.environ.get("ARACHNE_INCREMENTAL_WAV2VEC_MIN_MS") or "").strip()
    if raw_ms:
        try:
            ms = max(100, min(4000, int(raw_ms)))
            return int(16000 * ms / 1000)
        except ValueError:
            pass
    raw_samples = (os.environ.get("ARACHNE_INCREMENTAL_WAV2VEC_MIN_SAMPLES") or "").strip()
    if raw_samples:
        try:
            return max(1600, min(256_000, int(raw_samples)))
        except ValueError:
            pass
    return None


def drain_audio_stream(audio_stream: Iterator[np.ndarray]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for chunk in audio_stream:
        if chunk is None:
            continue
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            chunks.append(arr)
    if not chunks:
        raise ValueError("`audio_stream` yielded no chunks.")
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


class IncrementalStreamingAudioEmb:
    """
    Partial wav2vec for chunk 0, full wav2vec before chunk 1+.

    Used by ``generate_streaming_ai2v`` / ``generate_chunked_ai2v`` TTFF path.
    """

    def __init__(
        self,
        pipe: "ArachneXVideoAvatarPipeline",
        full_audio: np.ndarray,
        *,
        num_frames: int,
        first_chunk_frames: Optional[int],
        device: Union[str, torch.device],
        sample_rate: int = 16000,
    ) -> None:
        self.pipe = pipe
        self.full_audio = np.asarray(full_audio, dtype=np.float32).reshape(-1)
        self.num_frames = max(1, int(num_frames))
        self.sample_rate = int(sample_rate)
        self.device = device
        self.embedding_fps = default_embedding_fps(pipe)
        self.audio_stride = max(int(getattr(pipe, "vae_scale_factor_temporal", 4)), 1)
        self.audio_window = max(1, 2 * (int(getattr(pipe.dit, "audio_window", 5)) // 2) + 1)

        total = self.num_frames
        first_raw = int(first_chunk_frames) if first_chunk_frames else 9
        self.first_chunk_frames = round_to_4n_plus_1(min(first_raw, total))

        frame_min_samples = min_audio_samples_for_frames(
            self.first_chunk_frames,
            embedding_fps=self.embedding_fps,
            sample_rate=self.sample_rate,
            audio_stride=self.audio_stride,
            audio_window=self.audio_window,
        )
        override = _incremental_min_samples_override()
        self.prefix_samples = min(len(self.full_audio), override or frame_min_samples)

        self._partial_prepared: Optional[torch.Tensor] = None
        self._full_prepared: Optional[torch.Tensor] = None
        self._partial_wav2vec_sec: Optional[float] = None
        self._full_wav2vec_sec: Optional[float] = None

    def _record_wav2vec_metrics(self) -> None:
        rsm = getattr(self.pipe, "runtime_sampling_metrics", None)
        if rsm is None:
            return
        if self._partial_wav2vec_sec is not None:
            rsm.wav2vec_partial_sec = float(self._partial_wav2vec_sec)
        if self._full_wav2vec_sec is not None:
            rsm.wav2vec_full_sec = float(self._full_wav2vec_sec)

    def _build_partial(self) -> None:
        if self._partial_prepared is not None:
            return
        prefix = self.full_audio[: self.prefix_samples]
        t0 = time.perf_counter()
        prefix_emb = self.pipe.get_audio_embedding(
            prefix,
            fps=self.embedding_fps,
            device=self.device,
            sample_rate=self.sample_rate,
        )
        self._partial_wav2vec_sec = time.perf_counter() - t0
        self._partial_prepared = self.pipe._build_windowed_audio_embedding(
            prefix_emb,
            num_frames=self.first_chunk_frames,
            device=self.device,
        )
        self._record_wav2vec_metrics()

    def _build_full(self) -> None:
        if self._full_prepared is not None:
            return
        t0 = time.perf_counter()
        full_emb = self.pipe.get_audio_embedding(
            self.full_audio,
            fps=self.embedding_fps,
            device=self.device,
            sample_rate=self.sample_rate,
        )
        self._full_wav2vec_sec = time.perf_counter() - t0
        self._full_prepared = self.pipe._build_windowed_audio_embedding(
            full_emb,
            num_frames=self.num_frames,
            device=self.device,
        )
        self._record_wav2vec_metrics()

    def chunk_slice(self, chunk_idx: int, start: int, end: int, n_gen: int) -> torch.Tensor:
        del n_gen
        if chunk_idx == 0:
            self._build_partial()
            assert self._partial_prepared is not None
            return slice_audio_emb_temporal(
                self._partial_prepared,
                start,
                min(end, int(self._partial_prepared.shape[1])),
            )
        self._build_full()
        assert self._full_prepared is not None
        return slice_audio_emb_temporal(
            self._full_prepared,
            start,
            min(end, int(self._full_prepared.shape[1])),
        )

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "prefixSamples": int(self.prefix_samples),
            "prefixDurationMs": int(1000 * self.prefix_samples / self.sample_rate),
            "firstChunkFrames": int(self.first_chunk_frames),
            "wav2vecPartialSec": self._partial_wav2vec_sec,
            "wav2vecFullSec": self._full_wav2vec_sec,
        }


def build_avatar_windowed_audio_emb(
    pipe: "ArachneXVideoAvatarPipeline",
    audio_path: str,
    num_frames: int,
    device: Union[str, torch.device],
    sample_rate: int = 16000,
    embedding_fps: Optional[float] = None,
) -> torch.Tensor:
    """
    Load wav, run ``get_audio_embedding``, build [1, T, W, S, C] windows (same as ``scripts/infer._build_audio_emb``).
    """
    speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    audio_stride = int(getattr(pipe, "vae_scale_factor_temporal", 4))
    audio_stride = max(audio_stride, 1)
    fps = float(embedding_fps) if embedding_fps is not None else default_embedding_fps(pipe)
    full_audio_emb = pipe.get_audio_embedding(
        speech_array,
        fps=fps,
        device=device,
        sample_rate=sr,
    )
    audio_window = int(getattr(pipe.dit, "audio_window", 5))
    audio_window = max(1, 2 * (audio_window // 2) + 1)
    indices = torch.arange(audio_window, device=full_audio_emb.device) - (audio_window // 2)
    center_indices = torch.arange(
        0,
        audio_stride * num_frames,
        audio_stride,
        device=full_audio_emb.device,
    ).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    return full_audio_emb[center_indices][None, ...].to(device)
