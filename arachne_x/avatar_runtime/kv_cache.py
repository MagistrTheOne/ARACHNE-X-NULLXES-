"""KV-cache lifecycle and temporal compression helpers for avatar inference."""
from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import torch


class KVCacheMixin:
    """Cross-chunk KV-cache helpers preserved from the avatar pipeline."""

    def _update_kv_cache_dict(self, kv_cache_dict):
        self.kv_cache_dict = kv_cache_dict

    def _compress_kv_pair_temporal(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        num_cond_latents: int,
        num_ref_latents: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compress KV cache along temporal dimension by preserving:
        1) reference latents (if present),
        2) a summarized memory of old frames,
        3) a recent exact sliding window.
        """
        if not self.temporal_memory_enabled:
            return k.contiguous(), v.contiguous(), num_cond_latents

        if num_cond_latents <= 0 or k.ndim != 4 or v.ndim != 4:
            return k.contiguous(), v.contiguous(), num_cond_latents

        if k.shape != v.shape:
            return k.contiguous(), v.contiguous(), num_cond_latents

        seq_len = k.shape[2]
        if seq_len % num_cond_latents != 0:
            # Keep cache untouched if token layout is unknown.
            return k.contiguous(), v.contiguous(), num_cond_latents

        tokens_per_frame = seq_len // num_cond_latents
        preserve_ref_frames = max(0, min(num_ref_latents or 0, num_cond_latents))
        non_ref_frames = num_cond_latents - preserve_ref_frames

        # Nothing to compress.
        if non_ref_frames <= 0:
            return k.contiguous(), v.contiguous(), num_cond_latents

        keep_recent = max(1, int(self.temporal_memory_window_frames))
        keep_recent = min(keep_recent, non_ref_frames)
        old_frames = non_ref_frames - keep_recent

        if old_frames <= 0:
            return k.contiguous(), v.contiguous(), num_cond_latents

        summary_frames = max(0, int(self.temporal_memory_summary_frames))
        summary_frames = min(summary_frames, old_frames)
        if summary_frames <= 0:
            summary_frames = 1

        B, Hh, _, Dd = k.shape
        ref_tokens = preserve_ref_frames * tokens_per_frame
        old_tokens = old_frames * tokens_per_frame

        ref_k = k[:, :, :ref_tokens, :] if ref_tokens > 0 else None
        ref_v = v[:, :, :ref_tokens, :] if ref_tokens > 0 else None

        old_k = k[:, :, ref_tokens:ref_tokens + old_tokens, :]
        old_v = v[:, :, ref_tokens:ref_tokens + old_tokens, :]
        recent_k = k[:, :, ref_tokens + old_tokens:, :]
        recent_v = v[:, :, ref_tokens + old_tokens:, :]

        old_k = old_k.view(B, Hh, old_frames, tokens_per_frame, Dd)
        old_v = old_v.view(B, Hh, old_frames, tokens_per_frame, Dd)

        k_summaries = []
        v_summaries = []
        for i in range(summary_frames):
            start = (i * old_frames) // summary_frames
            end = ((i + 1) * old_frames) // summary_frames
            if end <= start:
                end = min(old_frames, start + 1)
            if end <= start:
                continue
            k_summaries.append(old_k[:, :, start:end, :, :].mean(dim=2, keepdim=True))
            v_summaries.append(old_v[:, :, start:end, :, :].mean(dim=2, keepdim=True))

        if k_summaries:
            old_k_summary = torch.cat(k_summaries, dim=2).reshape(B, Hh, -1, Dd)
            old_v_summary = torch.cat(v_summaries, dim=2).reshape(B, Hh, -1, Dd)
            summary_frames_eff = old_k_summary.shape[2] // tokens_per_frame
        else:
            old_k_summary = k.new_empty(B, Hh, 0, Dd)
            old_v_summary = v.new_empty(B, Hh, 0, Dd)
            summary_frames_eff = 0

        parts_k = []
        parts_v = []
        if ref_k is not None:
            parts_k.append(ref_k)
            parts_v.append(ref_v)
        parts_k.append(old_k_summary)
        parts_v.append(old_v_summary)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        k_comp = torch.cat(parts_k, dim=2).contiguous()
        v_comp = torch.cat(parts_v, dim=2).contiguous()
        effective_cond_latents = preserve_ref_frames + summary_frames_eff + keep_recent

        return k_comp, v_comp, effective_cond_latents

    def _compress_kv_cache_dict_temporal(
        self,
        kv_cache_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        num_cond_latents: int,
        num_ref_latents: int,
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], int]:
        if not self.temporal_memory_enabled or not kv_cache_dict:
            return kv_cache_dict, num_cond_latents

        compressed: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        effective_latent_counts: List[int] = []

        for layer_idx, cache in kv_cache_dict.items():
            if not isinstance(cache, tuple) or len(cache) != 2:
                compressed[layer_idx] = cache
                continue
            k, v = cache
            k_comp, v_comp, eff_count = self._compress_kv_pair_temporal(
                k,
                v,
                num_cond_latents=num_cond_latents,
                num_ref_latents=num_ref_latents,
            )
            compressed[layer_idx] = (k_comp, v_comp)
            effective_latent_counts.append(eff_count)

        if not effective_latent_counts:
            return compressed, num_cond_latents

        effective_num_cond_latents = min(effective_latent_counts)
        return compressed, effective_num_cond_latents

    def _cache_clean_latents(self, cond_latents, model_max_length, offload_kv_cache, device, dtype, audio_embs, num_cond_latents, num_ref_latents, ref_img_index):
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2]).to(device=device, dtype=dtype)
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        empty_embeds = torch.zeros([cond_latents.shape[0], 1, model_max_length, self.text_encoder.config.d_model], device=device, dtype=dtype)
        _, kv_cache_dict = self.dit(
            hidden_states=cond_latents, 
            timestep=timestep, 
            encoder_hidden_states=empty_embeds,
            num_cond_latents=num_cond_latents,
            return_kv=True, 
            skip_crs_attn=True, 
            offload_kv_cache=offload_kv_cache,
            audio_embs=audio_embs,
            num_ref_latents=num_ref_latents,
            ref_img_index=ref_img_index
        )
        effective_num_cond_latents = num_cond_latents
        if self.temporal_memory_enabled:
            kv_cache_dict, effective_num_cond_latents = self._compress_kv_cache_dict_temporal(
                kv_cache_dict,
                num_cond_latents=num_cond_latents,
                num_ref_latents=num_ref_latents or 0,
            )
            if effective_num_cond_latents != num_cond_latents:
                self.metrics.record("kv_cache_cond_latents_before", num_cond_latents)
                self.metrics.record("kv_cache_cond_latents_after", effective_num_cond_latents)
        self._update_kv_cache_dict(kv_cache_dict)
        return effective_num_cond_latents
    
    def _get_kv_cache_dict(self):
        return self.kv_cache_dict
    
    def _clear_cache(self):
        self.kv_cache_dict = None
        gc.collect()
        torch.cuda.empty_cache()
