"""
LoRA training loss utilities for avatar DiT (flow-match scheduler).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from arachne_x.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def load_flow_match_scheduler(checkpoint_dir: str) -> FlowMatchEulerDiscreteScheduler:
    sched_dir = os.path.join(checkpoint_dir, "scheduler")
    if os.path.isdir(sched_dir):
        return FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir)
    return FlowMatchEulerDiscreteScheduler()


@torch.no_grad()
def sigmas_for_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Map training timestep scalars to scheduler sigma (flow-match noise fraction)."""
    device = timesteps.device
    schedule_t = scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)
    flat = timesteps.view(-1).to(dtype=torch.float32)
    out = []
    for t in flat:
        idx = scheduler.index_for_timestep(t, schedule_t)
        out.append(sigmas[idx])
    return torch.stack(out)


def min_snr_flow_match_weights(sigmas: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Flow-match analogue of Min-SNR: down-weight high-sigma (noisy) steps that cause grain/snow.
    snr ~ ((1 - sigma) / sigma)^2
    """
    sigmas = sigmas.clamp(min=1e-4, max=1.0 - 1e-4)
    snr = ((1.0 - sigmas) / sigmas) ** 2
    cap = torch.tensor(float(gamma), device=sigmas.device, dtype=snr.dtype)
    return torch.minimum(snr, cap) / (snr + 1e-8)


def avatar_lora_diffusion_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    min_snr_gamma: float = 5.0,
) -> torch.Tensor:
    if min_snr_gamma <= 0:
        return F.mse_loss(noise_pred.float(), noise.float())

    sigmas = sigmas_for_timesteps(scheduler, timesteps)
    weights = min_snr_flow_match_weights(sigmas, min_snr_gamma)
    err = (noise_pred.float() - noise.float()) ** 2
    if err.ndim > 1:
        err = err.mean(dim=tuple(range(1, err.ndim)))
    return (err * weights).mean()


def stabilize_audio_embs(audio_embs: torch.Tensor) -> torch.Tensor:
    """Per-token RMS normalize — reduces audio-conditioning jitter / face snow."""
    x = audio_embs.float()
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x / std).to(dtype=audio_embs.dtype)
