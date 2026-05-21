"""
Small head: pooled UMT5 + audio scalars → K extra cross-attn tokens (4096-dim).

Disabled by default on the pipeline (`planning_enabled=False`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PlanningTokenHead(nn.Module):
    def __init__(
        self,
        d_model: int = 4096,
        n_tokens: int = 4,
        audio_stat_dim: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_tokens = int(n_tokens)
        self.audio_stat_dim = int(audio_stat_dim)
        in_dim = self.d_model + self.audio_stat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.d_model * 2),
            nn.SiLU(),
            nn.Linear(self.d_model * 2, self.n_tokens * self.d_model),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        text_pool: torch.Tensor,
        audio_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_pool: [B, D]
            audio_stats: [B, audio_stat_dim]
        Returns:
            planning_tokens: [B, K, D]
        """
        if text_pool.dim() != 2:
            raise ValueError(f"text_pool must be [B, D], got {tuple(text_pool.shape)}")
        if audio_stats.shape[0] != text_pool.shape[0]:
            raise ValueError("audio_stats batch must match text_pool")
        x = torch.cat([text_pool, audio_stats.to(dtype=text_pool.dtype, device=text_pool.device)], dim=-1)
        out = self.mlp(x)
        return out.view(text_pool.shape[0], self.n_tokens, self.d_model)
