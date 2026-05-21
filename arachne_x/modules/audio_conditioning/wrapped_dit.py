"""
Frozen-base VIDEO DiT forward with optional audio-conditioning adapter injection.

When ``audio_embs is None`` or ``audio_conditioning_scale == 0``, output matches base DiT.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange

from ...context_parallel import context_parallel_util
from ..arachne_video_dit import LongCatVideoTransformer3DModel
from .adapter import AudioConditioningAdapter


class AudioConditionedVideoDiTWrapper(nn.Module):
    def __init__(
        self,
        base_dit: LongCatVideoTransformer3DModel,
        adapter: Optional[AudioConditioningAdapter] = None,
    ) -> None:
        super().__init__()
        self.base_dit = base_dit
        self.adapter = adapter

    @property
    def config(self):
        return self.base_dit.config

    @property
    def patch_size(self):
        return self.base_dit.patch_size

    @property
    def cp_split_hw(self):
        return self.base_dit.cp_split_hw

    @property
    def dtype(self):
        return next(self.base_dit.parameters()).dtype

    def _prepare_audio_hidden_states(
        self,
        audio_embs: torch.Tensor,
        n_t: int,
    ) -> torch.Tensor:
        if self.adapter is None:
            raise RuntimeError("audio_embs provided but adapter is None")
        projected = self.adapter.project_audio_embs(audio_embs)
        return projected[-n_t:]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask=None,
        num_cond_latents: int = 0,
        return_kv: bool = False,
        kv_cache_dict=None,
        skip_crs_attn: bool = False,
        offload_kv_cache: bool = False,
        audio_embs: Optional[torch.Tensor] = None,
        audio_conditioning_scale: float = 0.0,
    ):
        use_audio = (
            self.adapter is not None
            and audio_embs is not None
            and float(audio_conditioning_scale) != 0.0
            and not skip_crs_attn
        )

        if not use_audio:
            return self.base_dit(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
                kv_cache_dict=kv_cache_dict or {},
                skip_crs_attn=skip_crs_attn,
                offload_kv_cache=offload_kv_cache,
            )

        kv_cache_dict = kv_cache_dict or {}
        dit = self.base_dit
        B, _, T, H, W = hidden_states.shape
        N_t = T // dit.patch_size[0]
        N_h = H // dit.patch_size[1]
        N_w = W // dit.patch_size[2]
        latent_shape = (N_t, N_h, N_w)

        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)

        dtype = dit.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = dit.x_embedder(hidden_states)

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            t = dit.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)

        encoder_hidden_states = dit.y_embedder(encoder_hidden_states)

        if dit.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = encoder_hidden_states.squeeze(1).masked_select(
                encoder_attention_mask.unsqueeze(-1) != 0
            ).view(1, -1, hidden_states.shape[-1])
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

        if dit.cp_split_hw[0] * dit.cp_split_hw[1] > 1:
            hidden_states = rearrange(hidden_states, "B (T H W) C -> B T H W C", T=N_t, H=N_h, W=N_w)
            hidden_states = context_parallel_util.split_cp_2d(
                hidden_states, seq_dim_hw=(2, 3), split_hw=dit.cp_split_hw
            )
            hidden_states = rearrange(hidden_states, "B T H W C -> B (T H W) C")

        audio_hidden_states = self._prepare_audio_hidden_states(audio_embs, n_t=N_t)
        scale = float(audio_conditioning_scale)

        kv_cache_dict_ret: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for i, block in enumerate(dit.blocks):
            if torch.is_grad_enabled() and dit.gradient_checkpointing:
                block_outputs = dit._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    latent_shape,
                    num_cond_latents,
                    return_kv,
                    kv_cache_dict.get(i, None),
                    skip_crs_attn,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    latent_shape,
                    num_cond_latents,
                    return_kv,
                    kv_cache_dict.get(i, None),
                    skip_crs_attn,
                )

            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                hidden_states = block_outputs

            hidden_states = self.adapter.inject_block(
                i,
                hidden_states,
                audio_hidden_states,
                latent_shape,
                num_cond_latents,
                scale,
            )

        hidden_states = dit.final_layer(hidden_states, t, latent_shape)

        if dit.cp_split_hw[0] * dit.cp_split_hw[1] > 1:
            hidden_states = context_parallel_util.gather_cp_2d(
                hidden_states, shape=latent_shape, split_hw=dit.cp_split_hw
            )

        hidden_states = dit.unpatchify(hidden_states, N_t, N_h, N_w)
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        return hidden_states
