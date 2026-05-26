"""
Auxiliary avatar loss runtime for LoRA training (VAE decode + frozen identity).

GUARDRAIL — TRAINING-ONLY, EXPENSIVE PATH (read before importing):

- This module is NOT part of the production realtime avatar serving path.
  It must never be imported by ``arachne_x/runtime/avatar_serving.py``,
  ``services/arachnex-worker/``, ``src/server/``, or any WebSocket / NDJSON
  hot path. Realtime owner is
  ``arachne_x/pipeline_arachne_x_video_avatar.py``.

- :class:`AvatarAuxTrainingRuntime` is designed for ``scripts/train_lora_avatar.py``
  Phase B+ auxiliary losses (perceptual / identity / lip-sync / region).
  Every aux step can run:
    * full VAE decode of estimated ``z0`` (frozen, but still expensive),
    * frozen identity encoder forward (DINO/VGG-grade backbone),
    * perceptual backbone forward.
  These are O(decode) per batch and easily dominate VRAM/time budgets on
  H200 — explicitly out of scope for realtime constraints.

- Stage schedule (``AvatarAuxStageSchedule``) is training-step indexed and
  does not make sense at inference time. Do not call ``compute_aux_loss``
  outside the LoRA training loop.

- Phoneme importance / mouth-mask defaults here are placeholders until real
  phoneme alignment is wired in the training data path; treat them as
  training scaffolding, not as production behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arachne_x.modules.autoencoder_kl_wan import AutoencoderKLWan
from arachne_x.modules.avatar_losses import ARACHNEAvatarLossModule
from arachne_x.modules.identity_encoder import FrozenIdentityEncoder
from arachne_x.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from arachne_x.training_vae_latent import (
    decode_normalized_latents_to_video,
    estimate_z0_from_flow_match,
)


@dataclass
class AvatarAuxStageSchedule:
    stage2_step: int = 5_000
    stage3_step: int = 15_000
    stage4_step: int = 30_000

    def stage_for_step(self, step: int) -> int:
        if step >= self.stage4_step:
            return 4
        if step >= self.stage3_step:
            return 3
        if step >= self.stage2_step:
            return 2
        return 1


def pool_audio_embs_for_sync(audio_embs: torch.Tensor, target_dim: int = 768) -> torch.Tensor:
    """Reduce DiT audio_embs to [B, T, 768] for lip-sync projector."""
    x = audio_embs.float()
    while x.dim() > 3:
        x = x.mean(dim=2)
    if x.shape[-1] != target_dim:
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), target_dim).transpose(1, 2)
    return x


def extract_mouth_features(video: torch.Tensor, dim: int = 512) -> torch.Tensor:
    """Lower-half face region pooled features for lip-sync projector input."""
    b, t, c, h, w = video.shape
    mouth_crop = video[:, :, :, int(h * 0.45) :, :]
    flat = mouth_crop.reshape(b * t, c, mouth_crop.shape[-2], mouth_crop.shape[-1])
    pooled = F.adaptive_avg_pool2d(flat, (4, 4)).reshape(b * t, -1)
    if pooled.shape[-1] > dim:
        pooled = pooled[:, :dim]
    elif pooled.shape[-1] < dim:
        pooled = F.pad(pooled, (0, dim - pooled.shape[-1]))
    return pooled.view(b, t, dim)


def default_phoneme_importance(length: int, device: torch.device) -> torch.Tensor:
    """Uniform weights placeholder until phoneme alignment is wired."""
    return torch.ones(length, device=device)


def default_region_weights(
    batch: int,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    yy = torch.linspace(0, 1, height, device=device).view(1, 1, height, 1)
    eyes = torch.exp(-((yy - 0.28) ** 2) / 0.02)
    jaw = torch.exp(-((yy - 0.82) ** 2) / 0.03)
    weights = 0.85 + 0.15 * eyes + 0.10 * jaw
    return weights.expand(batch, frames, height, width).contiguous()


class AvatarAuxTrainingRuntime(nn.Module):
    """Loads frozen VAE + identity encoder + staged aux loss module."""

    def __init__(
        self,
        checkpoint_dir: str,
        reference_image_path: str,
        device: torch.device,
        *,
        training_stage: int = 1,
        perceptual_backend: str = "vgg",
        stage_schedule: Optional[AvatarAuxStageSchedule] = None,
    ):
        super().__init__()
        self.device = device
        self.stage_schedule = stage_schedule or AvatarAuxStageSchedule()
        self.vae = AutoencoderKLWan.from_pretrained(
            checkpoint_dir,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.vae.to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        if hasattr(self.vae, "disable_tiling"):
            self.vae.disable_tiling()

        self.identity_encoder = FrozenIdentityEncoder.from_env().to(device)
        self.loss_module = ARACHNEAvatarLossModule(
            training_stage=training_stage,
            perceptual_backend="dino" if perceptual_backend == "dino" else "vgg",
        ).to(device)

        ref = FrozenIdentityEncoder.load_reference_image(reference_image_path, device)
        self.register_buffer("reference_image", ref, persistent=False)
        with torch.no_grad():
            self.reference_identity = self.identity_encoder.encode_images(ref)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        if self.loss_module.loss_balancer is not None:
            yield from self.loss_module.loss_balancer.parameters()
        stage = self.loss_module.training_stage
        if stage >= 3:
            yield from self.loss_module.lip_sync_loss.parameters()

    def resolve_stage(self, global_step: int) -> int:
        stage = self.stage_schedule.stage_for_step(global_step)
        if stage != self.loss_module.training_stage:
            self.loss_module.set_training_stage(stage)
        return stage

    def compute_aux_loss(
        self,
        latents: torch.Tensor,
        audio_embs: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: FlowMatchEulerDiscreteScheduler,
        *,
        global_step: int = 0,
        compute_perceptual: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, _, t, _, _ = latents.shape
        stage = self.resolve_stage(global_step)
        if compute_perceptual is None:
            run_perceptual = torch.rand(()).item() < self.loss_module.perceptual_prob
        else:
            run_perceptual = bool(compute_perceptual)

        need_identity = stage >= 2
        need_lip = stage >= 3
        need_decode = run_perceptual or need_identity or need_lip

        if need_decode:
            z0_est = estimate_z0_from_flow_match(
                latents, noise_pred, timesteps, scheduler
            )
            video = decode_normalized_latents_to_video(self.vae, z0_est)
            latents_for_aux = z0_est
        else:
            video = torch.zeros(
                b, t, 3, 64, 64, device=latents.device, dtype=torch.float32
            )
            latents_for_aux = latents

        if need_identity or need_lip:
            gen_identity = self.identity_encoder.encode_images(video)
        else:
            gen_identity = torch.zeros(
                b,
                t,
                self.reference_identity.shape[-1],
                device=latents.device,
                dtype=torch.float32,
            )

        mouth_features = (
            extract_mouth_features(video)
            if need_lip
            else torch.zeros(b, t, 512, device=latents.device, dtype=torch.float32)
        )
        audio_features = pool_audio_embs_for_sync(audio_embs)

        phoneme_importance = (
            default_phoneme_importance(t, latents.device) if need_lip else None
        )
        region_weights = (
            default_region_weights(
                b, t, latents.shape[-2], latents.shape[-1], latents.device
            )
            if stage >= 4
            else None
        )
        mouth_mask = (
            torch.zeros(b, t, latents.shape[-2], latents.shape[-1], device=latents.device)
            if stage >= 4
            else None
        )
        if mouth_mask is not None:
            mouth_mask[:, :, int(latents.shape[-2] * 0.55) :, :] = 1.0

        aux = self.loss_module(
            audio_features=audio_features,
            mouth_features=mouth_features,
            generated_identity_embeddings=gen_identity,
            reference_identity_embedding=self.reference_identity,
            latents=latents_for_aux,
            generated_images=video,
            reference_image=self.reference_image,
            compute_perceptual=run_perceptual,
            phoneme_importance=phoneme_importance,
            region_weights=region_weights,
            mouth_mask=mouth_mask,
        )
        return aux["total"], aux
