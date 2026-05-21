"""
Trainable audio-conditioning adapter for frozen VIDEO DiT.

Injection pattern: after selected base DiT blocks, residual audio cross-attention
with zero-initialized gates so scale=0 reproduces base i2v.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from ..avatar.blocks import AudioProjModel


@dataclass
class AudioConditioningAdapterConfig:
    hidden_size: int = 4096
    num_heads: int = 32
    audio_window: int = 5
    vae_scale: int = 4
    context_tokens: int = 32
    output_dim: int = 768
    intermediate_dim: int = 512
    block_indices: Tuple[int, ...] = field(
        default_factory=lambda: tuple(range(24, 48, 2))
    )
    enable_flashattn2: bool = True
    enable_flashattn3: bool = False
    enable_xformers: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["block_indices"] = list(self.block_indices)
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "AudioConditioningAdapterConfig":
        block_indices = payload.get("block_indices", cls().block_indices)
        return cls(
            hidden_size=int(payload.get("hidden_size", 4096)),
            num_heads=int(payload.get("num_heads", 32)),
            audio_window=int(payload.get("audio_window", 5)),
            vae_scale=int(payload.get("vae_scale", 4)),
            context_tokens=int(payload.get("context_tokens", 32)),
            output_dim=int(payload.get("output_dim", 768)),
            intermediate_dim=int(payload.get("intermediate_dim", 512)),
            block_indices=tuple(int(i) for i in block_indices),
            enable_flashattn2=bool(payload.get("enable_flashattn2", True)),
            enable_flashattn3=bool(payload.get("enable_flashattn3", False)),
            enable_xformers=bool(payload.get("enable_xformers", False)),
        )


class AudioInjectionBlock(nn.Module):
    """Per-block audio cross-attention residual with zero-init gate."""

    def __init__(self, config: AudioConditioningAdapterConfig) -> None:
        super().__init__()
        from ..avatar.attention import SingleStreamAttention

        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.audio_cross_attn = SingleStreamAttention(
            dim=config.hidden_size,
            encoder_hidden_states_dim=config.output_dim,
            num_heads=config.num_heads,
            qk_norm=True,
            qkv_bias=True,
            enable_flashattn2=config.enable_flashattn2,
            enable_flashattn3=config.enable_flashattn3,
            enable_xformers=config.enable_xformers,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        latent_shape: Tuple[int, int, int],
        num_cond_latents: int,
    ) -> torch.Tensor:
        x_norm = self.norm(hidden_states)
        output_cond, output_noise = self.audio_cross_attn(
            x_norm,
            audio_hidden_states,
            shape=latent_shape,
            num_cond_latents=num_cond_latents,
        )
        if output_cond is None:
            delta = output_noise
        else:
            b, n, c = hidden_states.shape
            n_t = latent_shape[0]
            num_cond_thw = num_cond_latents * (n // n_t)
            delta = torch.cat([output_cond, output_noise], dim=1)
            if delta.shape[1] != n:
                delta = delta[:, :n, :]
        return hidden_states + self.gate * delta


class AudioConditioningAdapter(nn.Module):
    """
    Projects wav2vec windows to per-frame audio tokens and injects them into
    selected frozen VIDEO DiT blocks.
    """

    def __init__(self, config: Optional[AudioConditioningAdapterConfig] = None) -> None:
        super().__init__()
        self.config = config or AudioConditioningAdapterConfig()
        self.audio_proj = AudioProjModel(
            seq_len=self.config.audio_window,
            seq_len_vf=self.config.audio_window + self.config.vae_scale - 1,
            intermediate_dim=self.config.intermediate_dim,
            output_dim=self.config.output_dim,
            context_tokens=self.config.context_tokens,
        )
        self.blocks = nn.ModuleDict(
            {
                str(idx): AudioInjectionBlock(self.config)
                for idx in self.config.block_indices
            }
        )

    @property
    def block_indices(self) -> Tuple[int, ...]:
        return self.config.block_indices

    def project_audio_embs(
        self,
        audio_embs: torch.Tensor,
        *,
        num_ref_latents: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            audio_embs: ``[B, T, W, S, C]`` wav2vec windows.
        Returns:
            ``[B*T, context_tokens, output_dim]`` tokens for cross-attention.
        """
        if audio_embs.dim() != 5:
            raise ValueError(f"audio_embs must be [B,T,W,S,C], got shape {tuple(audio_embs.shape)}")

        audio_cond = audio_embs.to(dtype=self.audio_proj.proj1.weight.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        middle_index = self.config.audio_window // 2

        latter_frame_audio_emb = rearrange(
            latter_frame_audio_emb,
            "b (n_t n) w s c -> b n_t n w s c",
            n=self.config.vae_scale,
        )
        latter_first = latter_frame_audio_emb[:, :, :1, : middle_index + 1, ...]
        latter_first = rearrange(latter_first, "b n_t n w s c -> b n_t (n w) s c")
        latter_last = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last = rearrange(latter_last, "b n_t n w s c -> b n_t (n w) s c")
        latter_middle = latter_frame_audio_emb[:, :, 1:-1, middle_index : middle_index + 1, ...]
        latter_middle = rearrange(latter_middle, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat(
            [latter_first, latter_middle, latter_last],
            dim=2,
        )
        audio_hidden_states = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)

        if num_ref_latents > 0:
            audio_start_ref = audio_hidden_states[:, [0], :, :]
            audio_hidden_states = torch.cat([audio_start_ref, audio_hidden_states], dim=1).contiguous()

        b, n_t, n_ctx, c = audio_hidden_states.shape
        return rearrange(audio_hidden_states, "b t n c -> (b t) n c")

    def inject_block(
        self,
        block_index: int,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        latent_shape: Tuple[int, int, int],
        num_cond_latents: int,
        scale: float,
    ) -> torch.Tensor:
        if scale == 0.0:
            return hidden_states
        key = str(block_index)
        if key not in self.blocks:
            return hidden_states
        block = self.blocks[key]
        delta_block = block(hidden_states, audio_hidden_states, latent_shape, num_cond_latents)
        if scale == 1.0:
            return delta_block
        return hidden_states + scale * (delta_block - hidden_states)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
