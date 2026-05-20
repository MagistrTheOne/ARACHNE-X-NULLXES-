"""
Avatar-specific auxiliary losses for ARACHNE-X.

Production-oriented multi-objective stack:
- Uncertainty-weighted loss balancing (Kendall et al.)
- CLIP-style InfoNCE lip sync (not BCE-on-matrix)
- Frozen identity / perceptual encoders (no co-trained extractors)
- Flow-warp temporal coherence without variance anti-penalty
- Staged activation to avoid gradient warfare on LoRA
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
    from torchvision.models import VGG16_Weights
except ImportError:  # pragma: no cover - optional at import time
    models = None
    VGG16_Weights = None


LOSS_NAMES: Tuple[str, ...] = (
    "lip_sync",
    "identity",
    "temporal",
    "expression",
    "perceptual",
)

STAGE_ACTIVE_LOSSES: Dict[int, Tuple[str, ...]] = {
    1: ("perceptual",),
    2: ("perceptual", "identity"),
    3: ("perceptual", "identity", "lip_sync"),
    4: ("perceptual", "identity", "lip_sync", "temporal"),
}

DEFAULT_FIXED_WEIGHTS: Dict[str, float] = {
    "lip_sync": 0.08,
    "identity": 0.10,
    "temporal": 0.03,
    "expression": 0.0,
    "perceptual": 0.15,
}


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] or [-1, 1] RGB tensors to ImageNet-normalized VGG input."""
    if images.min() < 0:
        images = (images + 1.0) * 0.5
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (images - mean) / std


def _build_flow_warp_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert pixel-space flow [B, 2, H, W] to grid_sample grid [B, H, W, 2]."""
    b, _, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = xx.unsqueeze(0) + flow[:, 0]
    sample_y = yy.unsqueeze(0) + flow[:, 1]
    grid_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)


def _warp_frame_with_flow(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp single frame [B, C, H, W] using flow [B, 2, H, W]."""
    grid = _build_flow_warp_grid(flow)
    return F.grid_sample(
        frame,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class UncertaintyLossBalancer(nn.Module):
    """Homoscedastic uncertainty weighting for multi-task losses."""

    def __init__(self, num_losses: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(losses) != self.log_vars.numel():
            raise ValueError(
                f"Expected {self.log_vars.numel()} losses, got {len(losses)}"
            )
        total = losses[0].new_zeros(())
        for idx, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[idx])
            total = total + precision * loss + self.log_vars[idx]
        return total


class LipSyncLoss(nn.Module):
    """
    Audio-visual sync via bidirectional InfoNCE (CLIP / SyncNet style).

    Trainable projectors map audio + mouth features into a shared sync space.
    The similarity objective itself is contrastive cross-entropy, not BCE.
    """

    def __init__(self, embedding_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.audio_projector = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )
        self.mouth_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        mouth_features: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = audio_features.shape
        audio_emb = F.normalize(self.audio_projector(audio_features), p=2, dim=-1)
        mouth_emb = F.normalize(self.mouth_projector(mouth_features), p=2, dim=-1)

        logits = torch.bmm(mouth_emb, audio_emb.transpose(1, 2)) / self.temperature
        targets = torch.arange(t, device=logits.device)
        targets = targets.unsqueeze(0).expand(b, -1)

        loss_a2v = F.cross_entropy(
            logits.reshape(b * t, t),
            targets.reshape(b * t),
        )
        loss_v2a = F.cross_entropy(
            logits.transpose(1, 2).reshape(b * t, t),
            targets.reshape(b * t),
        )
        return (loss_a2v + loss_v2a) * 0.5


class IdentityPreservationLoss(nn.Module):
    """
    Identity consistency on frozen face embeddings (ArcFace / AdaFace / etc.).

    Callers must supply embeddings from a frozen encoder. This module has no
    trainable parameters — co-training an identity head with DiT destroys the signal.
    """

    def forward(
        self,
        generated_identity_embeddings: torch.Tensor,
        reference_identity_embedding: torch.Tensor,
    ) -> torch.Tensor:
        generated = F.normalize(generated_identity_embeddings, p=2, dim=-1)
        reference = F.normalize(reference_identity_embedding, p=2, dim=-1)
        reference = reference.unsqueeze(1).expand_as(generated)
        similarity = (generated * reference).sum(dim=-1)
        return 1.0 - similarity.mean()


class TemporalCoherenceLoss(nn.Module):
    """
    Temporal coherence via flow-warped latent consistency.

    Avoids raw latent L1 (frozen-face syndrome) and removes variance anti-penalty
    (which incentivizes flicker). Mouth regions can be excluded via mouth_mask.
    """

    def __init__(self, low_freq_pool: int = 4):
        super().__init__()
        self.low_freq_pool = low_freq_pool

    def _low_frequency_smoothness(self, latents: torch.Tensor) -> torch.Tensor:
        """Fallback when optical flow is unavailable: smooth only low frequencies."""
        if self.low_freq_pool <= 1:
            pooled = latents
        else:
            pooled = F.avg_pool3d(
                latents,
                kernel_size=(1, self.low_freq_pool, self.low_freq_pool),
                stride=(1, self.low_freq_pool, self.low_freq_pool),
            )
        frame_diff = pooled[:, :, 1:] - pooled[:, :, :-1]
        return torch.mean(torch.abs(frame_diff))

    def forward(
        self,
        latents: torch.Tensor,
        optical_flow: Optional[torch.Tensor] = None,
        mouth_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, t, h, w = latents.shape
        if t < 2:
            return latents.new_zeros(())

        current = latents[:, :, :-1]
        nxt = latents[:, :, 1:]

        if optical_flow is not None:
            warped_frames = []
            for frame_idx in range(t - 1):
                warped_frames.append(
                    _warp_frame_with_flow(nxt[:, :, frame_idx], optical_flow[:, :, frame_idx])
                )
            warped = torch.stack(warped_frames, dim=2)
            residual = F.smooth_l1_loss(current, warped.detach(), reduction="none")
        else:
            residual = F.smooth_l1_loss(current, nxt.detach(), reduction="none")

        if mouth_mask is not None:
            # mouth_mask: 1 = exclude from penalty (preserve lip motion)
            keep = (1.0 - mouth_mask).clamp(min=0.0)
            if keep.shape[2] != residual.shape[2]:
                keep = F.interpolate(
                    keep,
                    size=residual.shape[-2:],
                    mode="nearest",
                )
            residual = residual * keep.unsqueeze(1)

        return residual.mean()


class ExpressionControlLoss(nn.Module):
    """
    Optional AU guidance — disabled by default in staged training.

    AU targets are noisy; keep weight at 0 until prosody/latent conditioning is stable.
    """

    def __init__(self, num_aus: int = 12):
        super().__init__()
        self.num_aus = num_aus
        self.au_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_aus),
            nn.Sigmoid(),
        )

    def forward(
        self,
        face_features: torch.Tensor,
        target_emotion_logits: torch.Tensor,
        emotion_weight: float = 1.0,
    ) -> torch.Tensor:
        predicted_aus = self.au_classifier(face_features)
        target_aus = torch.sigmoid(target_emotion_logits)
        au_loss = F.mse_loss(predicted_aus, target_aus)
        au_temporal = torch.mean(torch.abs(predicted_aus[:, 1:] - predicted_aus[:, :-1]))
        return emotion_weight * (au_loss + 0.1 * au_temporal)


class FrozenVGGPerceptualLoss(nn.Module):
    """
    Multi-layer frozen VGG16 feature loss (LPIPS-style, no trainable head).

    Expects RGB images in [0, 1] or [-1, 1].
    """

    _LAYER_SLICES = (4, 9, 16, 23)

    def __init__(self):
        super().__init__()
        if models is None:
            raise RuntimeError("torchvision is required for FrozenVGGPerceptualLoss")

        try:
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            vgg = models.vgg16(pretrained=True)

        features = list(vgg.features.children())
        slices = []
        prev = 0
        for end_idx in self._LAYER_SLICES:
            block = nn.Sequential(*features[prev:end_idx]).eval()
            for param in block.parameters():
                param.requires_grad_(False)
            slices.append(block)
            prev = end_idx
        self.blocks = nn.ModuleList(slices)

    def _extract(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = _normalize_images(images)
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return tuple(feats)

    def forward(
        self,
        generated_images: torch.Tensor,
        reference_image: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c, h, w = generated_images.shape
        gen_flat = generated_images.reshape(b * t, c, h, w)
        ref_flat = reference_image.unsqueeze(1).expand(b, t, c, h, w).reshape(b * t, c, h, w)

        gen_feats = self._extract(gen_flat)
        ref_feats = self._extract(ref_flat)

        loss = gen_flat.new_zeros(())
        for gen_f, ref_f in zip(gen_feats, ref_feats):
            loss = loss + F.l1_loss(gen_f, ref_f)
        return loss / len(gen_feats)


class ARACHNEAvatarLossModule(nn.Module):
    """
    Staged avatar auxiliary loss stack with uncertainty balancing.

    Recommended training schedule:
      stage 1: perceptual only
      stage 2: + identity (frozen ArcFace embeddings)
      stage 3: + lip sync (InfoNCE)
      stage 4: + temporal flow

    Diffusion / velocity loss lives in the training loop — not here.
    """

    def __init__(
        self,
        training_stage: int = 1,
        use_uncertainty_weighting: bool = True,
        enable_expression_loss: bool = False,
        fixed_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if training_stage not in STAGE_ACTIVE_LOSSES:
            raise ValueError(f"training_stage must be 1..4, got {training_stage}")

        self.training_stage = training_stage
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.enable_expression_loss = enable_expression_loss
        self.fixed_weights = dict(DEFAULT_FIXED_WEIGHTS)
        if fixed_weights:
            self.fixed_weights.update(fixed_weights)

        self.lip_sync_loss = LipSyncLoss()
        self.identity_loss = IdentityPreservationLoss()
        self.temporal_loss = TemporalCoherenceLoss()
        self.expression_loss = ExpressionControlLoss()
        self.perceptual_loss = FrozenVGGPerceptualLoss()

        active = self._active_loss_names()
        if use_uncertainty_weighting:
            self.loss_balancer = UncertaintyLossBalancer(len(active))
        else:
            self.loss_balancer = None

    def _active_loss_names(self) -> Tuple[str, ...]:
        names = list(STAGE_ACTIVE_LOSSES[self.training_stage])
        if self.enable_expression_loss and "expression" not in names:
            names.append("expression")
        return tuple(names)

    def set_training_stage(self, stage: int) -> None:
        if stage not in STAGE_ACTIVE_LOSSES:
            raise ValueError(f"training_stage must be 1..4, got {stage}")
        self.training_stage = stage
        active = self._active_loss_names()
        if self.use_uncertainty_weighting:
            device = self.loss_balancer.log_vars.device
            old_log_vars = self.loss_balancer.log_vars.detach().clone()
            self.loss_balancer = UncertaintyLossBalancer(len(active)).to(device)
            copy_n = min(old_log_vars.numel(), len(active))
            if copy_n > 0:
                with torch.no_grad():
                    self.loss_balancer.log_vars[:copy_n].copy_(old_log_vars[:copy_n])

    def _compute_named_losses(
        self,
        audio_features: torch.Tensor,
        mouth_features: torch.Tensor,
        generated_identity_embeddings: torch.Tensor,
        reference_identity_embedding: torch.Tensor,
        latents: torch.Tensor,
        generated_images: torch.Tensor,
        reference_image: torch.Tensor,
        face_features: Optional[torch.Tensor],
        target_emotion_logits: Optional[torch.Tensor],
        optical_flow: Optional[torch.Tensor],
        mouth_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        all_losses = {
            "lip_sync": self.lip_sync_loss(audio_features, mouth_features),
            "identity": self.identity_loss(
                generated_identity_embeddings,
                reference_identity_embedding,
            ),
            "temporal": self.temporal_loss(latents, optical_flow, mouth_mask),
            "perceptual": self.perceptual_loss(generated_images, reference_image),
        }

        if face_features is not None and target_emotion_logits is not None:
            all_losses["expression"] = self.expression_loss(
                face_features,
                target_emotion_logits,
            )
        else:
            all_losses["expression"] = latents.new_zeros(())

        return all_losses

    def forward(
        self,
        audio_features: torch.Tensor,
        mouth_features: torch.Tensor,
        generated_identity_embeddings: torch.Tensor,
        reference_identity_embedding: torch.Tensor,
        latents: torch.Tensor,
        generated_images: torch.Tensor,
        reference_image: torch.Tensor,
        face_features: Optional[torch.Tensor] = None,
        target_emotion_logits: Optional[torch.Tensor] = None,
        optical_flow: Optional[torch.Tensor] = None,
        mouth_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        all_losses = self._compute_named_losses(
            audio_features=audio_features,
            mouth_features=mouth_features,
            generated_identity_embeddings=generated_identity_embeddings,
            reference_identity_embedding=reference_identity_embedding,
            latents=latents,
            generated_images=generated_images,
            reference_image=reference_image,
            face_features=face_features,
            target_emotion_logits=target_emotion_logits,
            optical_flow=optical_flow,
            mouth_mask=mouth_mask,
        )

        active_names = self._active_loss_names()
        active_tensors = [all_losses[name] for name in active_names]

        if self.use_uncertainty_weighting and self.loss_balancer is not None:
            total_loss = self.loss_balancer(active_tensors)
        else:
            total_loss = active_tensors[0].new_zeros(())
            for name in active_names:
                total_loss = total_loss + self.fixed_weights.get(name, 0.0) * all_losses[name]

        result: Dict[str, torch.Tensor] = {
            name: all_losses[name].detach() for name in LOSS_NAMES
        }
        result["active_losses"] = torch.tensor(
            [float(name in active_names) for name in LOSS_NAMES],
            device=latents.device,
        )
        result["total"] = total_loss
        return result
