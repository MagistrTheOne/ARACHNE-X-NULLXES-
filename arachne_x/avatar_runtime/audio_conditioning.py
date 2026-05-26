"""Audio and emotion conditioning helpers for avatar inference."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import loguru
import numpy as np
import pyloudnorm as pyln
import scipy.signal as ss
import torch

from arachne_x.utils.monitoring import sha256_of_audio_array


class AudioConditioningMixin:
    """Audio embedding and emotion-channel methods."""

    def _loudness_norm(self, audio_array, sr=16000, lufs=-23, threshold=100):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > threshold:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def _add_noise_floor(self, audio, noise_db=-45):
        noise_amp = 10 ** (noise_db / 20)
        noise = np.random.randn(len(audio)) * noise_amp
        return audio + noise

    def _smooth_transients(self, audio, sr=16000):
        b, a = ss.butter(3, 3000 / (sr/2))
        return ss.lfilter(b, a, audio)

    def _normalize_emotion_ids(
        self,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]],
        batch_size: int,
    ) -> Optional[List[int]]:
        if emotion_id is None:
            return None

        if isinstance(emotion_id, (int, str)):
            raw_items: List[Union[int, str]] = [emotion_id] * batch_size
        elif isinstance(emotion_id, torch.Tensor):
            raw_items = [int(x) for x in emotion_id.detach().cpu().view(-1).tolist()]
        elif isinstance(emotion_id, (list, tuple)):
            raw_items = list(emotion_id)
        else:
            raise TypeError(
                f"`emotion_id` must be int, str, list, torch.Tensor, or None. Got {type(emotion_id)}."
            )

        if len(raw_items) == 1 and batch_size > 1:
            raw_items = raw_items * batch_size
        if len(raw_items) != batch_size:
            raise ValueError(
                f"`emotion_id` length must be 1 or equal to batch size ({batch_size}), got {len(raw_items)}."
            )

        resolved: List[int] = []
        for item in raw_items:
            if isinstance(item, str):
                key = item.strip().lower()
                if key not in self.emotion_label_to_id:
                    raise ValueError(
                        f"Unknown emotion label `{item}`. Allowed: {sorted(self.emotion_label_to_id.keys())}."
                    )
                resolved.append(int(self.emotion_label_to_id[key]))
            else:
                idx = int(item)
                if idx < 0 or idx >= self.emotion_num_classes:
                    raise ValueError(
                        f"`emotion_id` {idx} is out of range [0, {self.emotion_num_classes - 1}]."
                    )
                resolved.append(idx)
        return resolved

    def _apply_emotion_channel(
        self,
        audio_emb: torch.Tensor,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]],
        emotion_intensity: float,
        batch_size: int,
        num_videos_per_prompt: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, bool]:
        if not self.emotion_enabled or emotion_id is None:
            return audio_emb, False
        if emotion_intensity <= 0:
            return audio_emb, False
        if audio_emb.ndim != 5:
            return audio_emb, False

        ids = self._normalize_emotion_ids(emotion_id, batch_size=batch_size)
        expanded_ids: List[int] = []
        for idx in ids:
            expanded_ids.extend([idx] * num_videos_per_prompt)

        emb = self.emotion_embedding(
            torch.tensor(expanded_ids, dtype=torch.long, device=self.emotion_embedding.weight.device)
        ).to(device=device, dtype=audio_emb.dtype)
        emb = self.emotion_proj.to(device=device, dtype=audio_emb.dtype)(emb)

        B = audio_emb.shape[0]
        emotion_ctx = emb.view(B, 1, 1, 1, -1)

        audio_rms = (
            audio_emb.to(torch.float32).pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
        )
        emotion_rms = (
            emotion_ctx.to(torch.float32).pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
        )
        requested_scale = torch.full_like(audio_rms, float(emotion_intensity))
        max_scale = (audio_rms * float(self.emotion_lipsync_guard_ratio)) / emotion_rms
        safe_scale = torch.minimum(requested_scale, max_scale)
        safe_scale = torch.clamp(safe_scale, min=0.0)

        conditioned = audio_emb + emotion_ctx * safe_scale.to(device=device, dtype=audio_emb.dtype)

        requested = float(emotion_intensity)
        applied = float(safe_scale.mean().item())
        clipped = bool(applied + 1e-6 < requested)
        self.metrics.record("emotion_intensity_requested", requested)
        self.metrics.record("emotion_intensity_applied", applied)
        self.metrics.record("emotion_lipsync_guard_triggered", int(clipped))
        self.metrics.record("emotion_lipsync_guard_ratio", float(self.emotion_lipsync_guard_ratio))

        return conditioned.contiguous(), applied > 0.0

    @torch.no_grad()
    def get_audio_embedding(self, speech_array, fps=32, device='cpu', sample_rate=16000):
            
        # optional disk cache for audio embeddings to accelerate repeated runs
        cache_dir = getattr(self, 'audio_cache_dir', './audio_cache')
        os.makedirs(cache_dir, exist_ok=True)

        from arachne_x.modules.audio.nullxes_audio_encoder import resolve_audio_encoder_backend

        enc_tag = resolve_audio_encoder_backend()
        key = (
            sha256_of_audio_array(np.ascontiguousarray(speech_array))
            + f"_fps{fps}_sr{sample_rate}_enc{enc_tag}_wav2vec_only_v4"
        )
        cache_path = os.path.join(cache_dir, key + '.npz')

        if os.path.exists(cache_path):
            try:
                with self.metrics.timeit('audio_cache_load'):
                    npz = np.load(cache_path)
                    if "audio_emb_final" in npz:
                        cached = npz["audio_emb_final"]
                    else:
                        cached = npz["audio_emb"]
                    audio_emb = torch.from_numpy(cached).to(device=device)
                    # return shape (T, B, D)
                    return audio_emb
            except Exception as exc:
                loguru.logger.debug("Audio cache load failed; recomputing. Error: {}", exc)

        from arachne_x.modules.audio.nullxes_audio_encoder import encode_avatar_audio

        speech_pre = np.ascontiguousarray(speech_array)
        audio_emb = encode_avatar_audio(
            self,
            speech_pre,
            fps=fps,
            device=device,
            sample_rate=sample_rate,
        )

        try:
            payload = {
                "audio_emb": audio_emb.cpu().numpy(),
                "audio_emb_final": audio_emb.cpu().numpy(),
            }
            np.savez_compressed(cache_path, **payload)
            self.metrics.record('audio_cache_saved', 1)
        except Exception as exc:
            loguru.logger.warning("Audio cache save failed; continuing without cache. Error: {}", exc)

        return audio_emb

    def _build_windowed_audio_embedding(
        self,
        full_audio_emb: torch.Tensor,
        num_frames: int,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        if full_audio_emb.dim() == 2:
            full_audio_emb = full_audio_emb.unsqueeze(1)
        if full_audio_emb.dim() != 3:
            raise ValueError(
                f"Expected full audio embedding with shape [T, S, C], got {tuple(full_audio_emb.shape)}."
            )
        if full_audio_emb.shape[0] <= 0:
            raise ValueError("Audio embedding has no timesteps.")

        audio_window = int(getattr(self.dit, "audio_window", 5))
        audio_window = max(1, 2 * (audio_window // 2) + 1)
        audio_stride = max(int(self.vae_scale_factor_temporal), 1)

        offsets = torch.arange(audio_window, device=full_audio_emb.device) - (audio_window // 2)
        center_indices = torch.arange(
            0,
            audio_stride * int(num_frames),
            audio_stride,
            device=full_audio_emb.device,
        ).unsqueeze(1) + offsets.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)

        windowed = full_audio_emb[center_indices][None, ...]  # [1, T, W, S, C]
        return windowed.to(device=device, dtype=self.dit.dtype)

    def _prepare_audio_emb_for_dit(
        self,
        audio_emb: torch.Tensor,
        *,
        num_frames: int,
        batch_size: int,
        num_videos_per_prompt: int,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        if audio_emb is None:
            raise ValueError("`audio_emb` is required for audio-driven generation.")

        if not torch.is_tensor(audio_emb):
            audio_emb = torch.as_tensor(audio_emb)

        if audio_emb.dim() == 3:
            audio_emb = self._build_windowed_audio_embedding(audio_emb, num_frames=num_frames, device=device)
        elif audio_emb.dim() == 4:
            audio_emb = audio_emb.unsqueeze(0)
        elif audio_emb.dim() != 5:
            raise ValueError(
                f"`audio_emb` must be 3D [T,S,C], 4D [T,W,S,C], or 5D [B,T,W,S,C], got {tuple(audio_emb.shape)}."
            )

        expected_time = int(num_frames)
        if audio_emb.shape[1] != expected_time:
            raise ValueError(
                f"`audio_emb` time dimension mismatch: expected {expected_time}, got {audio_emb.shape[1]}."
            )

        expected_window = max(1, 2 * (int(getattr(self.dit, "audio_window", 5)) // 2) + 1)
        if audio_emb.shape[2] != expected_window:
            raise ValueError(
                f"`audio_emb` window dimension mismatch: expected {expected_window}, got {audio_emb.shape[2]}."
            )

        target_batch = int(batch_size) * int(num_videos_per_prompt)
        source_batch = int(audio_emb.shape[0])
        if source_batch == 1 and target_batch > 1:
            audio_emb = audio_emb.expand(target_batch, -1, -1, -1, -1).contiguous()
        elif source_batch == int(batch_size) and int(num_videos_per_prompt) > 1:
            audio_emb = audio_emb.repeat_interleave(int(num_videos_per_prompt), dim=0)
        elif source_batch != target_batch:
            raise ValueError(
                f"`audio_emb` batch dimension mismatch: expected 1, {batch_size}, or {target_batch}, got {source_batch}."
            )

        return audio_emb.to(device=device, dtype=self.dit.dtype, non_blocking=True)
