"""
Shared VAE latent normalize/denormalize — must match pipeline_arachne_x_video_avatar.
"""

from __future__ import annotations

from typing import Tuple

import torch

from arachne_x.modules.autoencoder_kl_wan import AutoencoderKLWan
from arachne_x.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from arachne_x.training_lora_loss import sigmas_for_timesteps


def vae_latent_mean_inv_std(
    vae: AutoencoderKLWan,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_dim = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(
        1, z_dim, 1, 1, 1
    )
    inv_std = (1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=dtype)).view(
        1, z_dim, 1, 1, 1
    )
    return mean, inv_std


def normalize_vae_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    mean, inv_std = vae_latent_mean_inv_std(vae, latents.device, latents.dtype)
    return (latents.float() - mean) * inv_std


def denormalize_vae_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize_vae_latents — matches pipe.denormalize_latents()."""
    mean, inv_std = vae_latent_mean_inv_std(vae, latents.device, latents.dtype)
    return latents.float() / inv_std + mean


def estimate_z0_from_flow_match(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
) -> torch.Tensor:
    """
    Invert flow-match scale_noise: x_t = sigma*eps + (1-sigma)*z0  =>  z0 = (x_t - sigma*eps)/(1-sigma).

    Uses DiT noise_pred (differentiable w.r.t. LoRA) rather than dataset epsilon.
    """
    sigmas = sigmas_for_timesteps(scheduler, timesteps)
    while sigmas.dim() < noisy_latents.dim():
        sigmas = sigmas.unsqueeze(-1)
    sigma = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    denom = (1.0 - sigma).clamp(min=1e-4)
    return (noisy_latents.float() - sigma * noise_pred.float()) / denom


def decode_normalized_latents_to_video(
    vae: AutoencoderKLWan,
    normalized_latents: torch.Tensor,
) -> torch.Tensor:
    """
    Decode normalized [B,C,T,H,W] latents to [B,T,C,H,W] RGB in [0,1], fp32.

    VAE weights are frozen; gradients flow to normalized_latents input.
    """
    vae.eval()
    was_tiling = bool(getattr(vae, "use_tiling", False))
    if was_tiling:
        vae.disable_tiling()
    try:
        z = denormalize_vae_latents(vae, normalized_latents)
        with torch.autocast(device_type=z.device.type, enabled=False):
            decoded = vae.decode(z, return_dict=False)[0]
        video = decoded.float().permute(0, 2, 1, 3, 4)
        return ((video + 1.0) * 0.5).clamp(0.0, 1.0)
    finally:
        if was_tiling:
            vae.enable_tiling()
