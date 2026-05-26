"""Audio, phoneme, and emotion conditioning helpers for avatar inference."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import loguru
import numpy as np
import pyloudnorm as pyln
import scipy.signal as ss
import torch
import torch.nn.functional as F

from arachne_x.utils.monitoring import sha256_of_audio_array


class AudioConditioningMixin:
    """Audio embedding, phoneme stream, and emotion-channel methods."""

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

    def _apply_multistream_fusion(
        self,
        audio_emb: torch.Tensor,
        fused_emb: Optional[torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        if fused_emb is None:
            return audio_emb
        if audio_emb.dim() == 3:
            audio_bt = audio_emb.permute(1, 0, 2).contiguous()
        elif audio_emb.dim() == 2:
            audio_bt = audio_emb.unsqueeze(0)
        else:
            return audio_emb

        fused = fused_emb.to(device=device, dtype=audio_bt.dtype)
        proj = self.multi_stream_fusion_proj.to(device=device, dtype=audio_bt.dtype)
        fused_proj = proj(fused)  # [B, min_t, 768]
        fused_proj = fused_proj.permute(0, 2, 1)
        fused_proj = F.interpolate(
            fused_proj, size=audio_bt.shape[1], mode="linear", align_corners=False
        )
        fused_proj = fused_proj.permute(0, 2, 1)
        audio_bt = audio_bt + self.multi_stream_fusion_scale * fused_proj

        if audio_emb.dim() == 3:
            return audio_bt.permute(1, 0, 2).contiguous()
        return audio_bt.squeeze(0)

    @torch.no_grad()
    def _extract_phoneme_timeline(
        self,
        speech_array: np.ndarray,
        sample_rate: int,
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, Any]]:
        if not self.phoneme_enabled or self.phoneme_aligner is None or target_len <= 0:
            return None

        try:
            with self.metrics.timeit("phoneme_extract"):
                phoneme_out = self.phoneme_aligner.extract(
                    speech_array,
                    sample_rate=sample_rate,
                    target_len=target_len,
                )

            phoneme_probs = phoneme_out["phoneme_probs"].to(device=device, dtype=dtype)
            phoneme_ids = phoneme_out["phoneme_ids"].to(device=device, dtype=torch.long)
            confidence = phoneme_out["confidence"].to(device=device, dtype=dtype)

            self.metrics.record("phoneme_voiced_ratio", float(phoneme_out.get("voiced_ratio", 0.0)))
            self.metrics.record("phoneme_silence_ratio", float(phoneme_out.get("silence_ratio", 0.0)))
            self.metrics.record("phoneme_fricative_ratio", float(phoneme_out.get("fricative_ratio", 0.0)))
            self.metrics.record("phoneme_plosive_ratio", float(phoneme_out.get("plosive_ratio", 0.0)))
            self.metrics.record("phoneme_confidence_mean", float(confidence.mean().item()))

            return {
                "phoneme_probs": phoneme_probs,
                "phoneme_ids": phoneme_ids,
                "confidence": confidence,
            }
        except Exception as exc:
            self.metrics.record("phoneme_fallback_count", 1)
            if not self.phoneme_fallback_to_wav2vec:
                raise
            loguru.logger.warning(
                "Phoneme extraction failed; using wav2vec fallback only. Error: {}",
                exc,
            )
            return None

    def _inject_phoneme_conditioning(
        self,
        audio_emb: torch.Tensor,
        phoneme_probs: torch.Tensor,
        confidence: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if audio_emb.dim() not in (2, 3):
            return audio_emb

        phoneme_proj = self.phoneme_proj.to(device=device, dtype=audio_emb.dtype)
        phoneme_ctx = phoneme_proj(phoneme_probs.to(device=device, dtype=audio_emb.dtype))
        conf = confidence.to(device=device, dtype=audio_emb.dtype).clamp(
            min=float(self.phoneme_confidence_floor), max=1.0
        )
        phoneme_ctx = phoneme_ctx * conf.unsqueeze(-1)

        if audio_emb.dim() == 3:
            phoneme_ctx = phoneme_ctx.unsqueeze(1).expand(-1, audio_emb.shape[1], -1)

        conditioned = audio_emb + float(self.phoneme_stream_scale) * phoneme_ctx
        return conditioned.contiguous()

    def _compute_phoneme_alignment_metrics(
        self,
        audio_emb: torch.Tensor,
        phoneme_probs: torch.Tensor,
        phoneme_ids: torch.Tensor,
        device: torch.device,
    ) -> Optional[Dict[str, float]]:
        if audio_emb.dim() == 3:
            frame_repr = audio_emb.mean(dim=1)
        elif audio_emb.dim() == 2:
            frame_repr = audio_emb
        else:
            return None

        target_len = phoneme_probs.shape[0]
        if target_len <= 0:
            return None

        if frame_repr.shape[0] != target_len:
            frame_repr = frame_repr.transpose(0, 1).unsqueeze(0)
            frame_repr = F.interpolate(frame_repr, size=target_len, mode="linear", align_corners=False)
            frame_repr = frame_repr.squeeze(0).transpose(0, 1).contiguous()

        frame_repr = frame_repr.to(device=device, dtype=torch.float32)
        probs = phoneme_probs.to(device=device, dtype=torch.float32)
        ids = phoneme_ids.to(device=device, dtype=torch.long)

        head = self.phoneme_alignment_head.to(device=device, dtype=frame_repr.dtype)
        logits = head(frame_repr)
        log_probs = F.log_softmax(logits, dim=-1)

        kl = F.kl_div(log_probs, probs, reduction="batchmean")
        ce = F.nll_loss(log_probs, ids, reduction="mean")
        loss = 0.5 * (kl + ce)
        pred = torch.argmax(log_probs, dim=-1)
        acc = (pred == ids).float().mean()

        return {
            "loss": float(loss.item()),
            "kl": float(kl.item()),
            "ce": float(ce.item()),
            "acc": float(acc.item()),
        }

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

        phoneme_scale_tag = str(round(float(self.phoneme_stream_scale), 4)).replace(".", "p")
        enc_tag = resolve_audio_encoder_backend()
        key = (
            sha256_of_audio_array(np.ascontiguousarray(speech_array))
            + f"_fps{fps}_sr{sample_rate}_ph{int(bool(self.phoneme_enabled))}_pn{self.phoneme_num_classes}_ps{phoneme_scale_tag}_enc{enc_tag}_v3"
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

        # try to compute fused multi-stream features and persist to cache
        try:
            fused_emb = None
            try:
                # audio_emb shape may be [T, B, D] or [T, D]
                a = audio_emb
                if a.dim() == 3:
                    # [T, B, D] -> [B, T, D]
                    wav2vec_feats = a.permute(1, 0, 2).contiguous()
                elif a.dim() == 2:
                    wav2vec_feats = a.unsqueeze(0)
                else:
                    wav2vec_feats = a

                processor_in = wav2vec_feats.cpu()
                proc_out = self.audio_processor(processor_in)
                fused_emb = proc_out.get('fused_embeddings', None)
            except Exception as exc:
                loguru.logger.debug(
                    "Audio multi-stream processor step failed; fused embeddings disabled. Error: {}",
                    exc,
                )
                fused_emb = None

            if fused_emb is not None:
                audio_emb = self._apply_multistream_fusion(audio_emb, fused_emb, device=device)
        except Exception as exc:
            loguru.logger.debug("Multi-stream fusion failed; continuing with wav2vec stream only. Error: {}", exc)

        phoneme_ctx = self._extract_phoneme_timeline(
            speech_array=speech_array,
            sample_rate=sample_rate,
            target_len=audio_emb.shape[0],
            device=device,
            dtype=audio_emb.dtype,
        )
        if phoneme_ctx is not None:
            audio_emb = self._inject_phoneme_conditioning(
                audio_emb=audio_emb,
                phoneme_probs=phoneme_ctx["phoneme_probs"],
                confidence=phoneme_ctx["confidence"],
                device=device,
            )
            align_metrics = self._compute_phoneme_alignment_metrics(
                audio_emb=audio_emb,
                phoneme_probs=phoneme_ctx["phoneme_probs"],
                phoneme_ids=phoneme_ctx["phoneme_ids"],
                device=device,
            )
            if align_metrics is not None:
                self.metrics.record("phoneme_alignment_loss", align_metrics["loss"])
                self.metrics.record("phoneme_alignment_kl", align_metrics["kl"])
                self.metrics.record("phoneme_alignment_ce", align_metrics["ce"])
                self.metrics.record("phoneme_alignment_acc", align_metrics["acc"])

        try:
            payload = {
                "audio_emb": audio_emb.cpu().numpy(),
                "audio_emb_final": audio_emb.cpu().numpy(),
            }
            if phoneme_ctx is not None:
                payload["phoneme_probs"] = phoneme_ctx["phoneme_probs"].cpu().numpy()
                payload["phoneme_confidence"] = phoneme_ctx["confidence"].cpu().numpy()
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
