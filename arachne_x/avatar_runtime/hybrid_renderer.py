"""Hybrid mouth-zone renderer helpers for avatar video decode output."""
from __future__ import annotations

from typing import Optional

import loguru
import torch
import torch.nn.functional as F


class HybridRendererMixin:
    """Mouth-zone mask preparation, blending, and renderer budget methods."""

    def _resize_and_centercrop_tensor(self, mask: torch.Tensor, target_h: int, target_w: int, resize_mode: str = 'crop'):
        """
        mask: Tensor, shape [3, H, W], dtype=float, device=gpu/cpu
        return: [3, target_h, target_w]
        """

        if resize_mode == 'default':
            mask_resized = F.interpolate(
                mask.unsqueeze(0),  # [1, 3, H, W]
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            return mask_resized

        elif resize_mode == 'crop':
            _, H, W = mask.shape
            ratio = target_w / target_h # 1
            src_ratio = W / H # > 1

            if ratio > src_ratio:
                new_w = target_w
                new_h = int(H * target_w / W)
            else:
                new_h = target_h
                new_w = int(W * target_h / H)

            mask_resized = F.interpolate(
                mask.unsqueeze(0),  # [1, 3, H, W]
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            top = (new_h - target_h) // 2
            left = (new_w - target_w) // 2

            mask_resized_cropped = mask_resized[:, top:top + target_h, left:left + target_w]
            return mask_resized_cropped
        
        else:
            raise ValueError(f"Unsupported resize_mode {resize_mode}. Use 'default' or 'crop'.")

    def _prepare_mouth_zone_mask(
        self,
        mouth_zone_masks: Optional[torch.Tensor],
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        resize_mode: str = "crop",
    ) -> Optional[torch.Tensor]:
        if mouth_zone_masks is None:
            return None

        mask = mouth_zone_masks
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=device, dtype=torch.float32)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif mask.ndim == 3:
            if mask.shape[0] in (1, 3):
                mask = mask.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
            else:
                mask = mask.unsqueeze(1)  # [B,1,H,W]
        elif mask.ndim == 4:
            if mask.shape[1] in (1, 3):
                mask = mask.mean(dim=1, keepdim=True)  # [B,1,H,W]
            else:
                return None
        else:
            return None

        if mask.shape[-2:] != (height, width):
            if mask.shape[0] == 1:
                m3 = mask[0].repeat(3, 1, 1)
                m3 = self._resize_and_centercrop_tensor(m3, height, width, resize_mode)
                mask = m3.mean(dim=0, keepdim=True).unsqueeze(0)
            else:
                resized = []
                for b in range(mask.shape[0]):
                    m3 = mask[b].repeat(3, 1, 1)
                    m3 = self._resize_and_centercrop_tensor(m3, height, width, resize_mode)
                    resized.append(m3.mean(dim=0, keepdim=True))
                mask = torch.stack(resized, dim=0)

        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1).contiguous()
        elif mask.shape[0] != batch_size:
            return None

        mask = torch.clamp(mask, 0.0, 1.0)
        for _ in range(max(1, int(self.hybrid_renderer_blur_passes))):
            mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        mask = torch.clamp(mask, 0.0, 1.0).to(dtype=dtype)
        mask = mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)  # [B,1,T,H,W]
        return mask.contiguous()

    def _compute_seam_boundary_mask(self, mouth_mask: torch.Tensor) -> torch.Tensor:
        # High value on transition ring around mouth mask.
        b, _, t, h, w = mouth_mask.shape
        flat = mouth_mask.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        eroded = F.avg_pool2d(flat, kernel_size=5, stride=1, padding=2)
        ring = torch.clamp(flat - eroded, min=0.0, max=1.0)
        ring = torch.clamp(ring * 4.0, min=0.0, max=1.0)
        ring = ring.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return ring

    def _build_mouth_controlled_branch(self, decoded_video: torch.Tensor, strength: float) -> torch.Tensor:
        # Deterministic high-frequency enhancement branch for the controlled zone.
        b, c, t, h, w = decoded_video.shape
        flat = decoded_video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        blurred = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1)
        detail = flat - blurred
        branch = flat + float(strength) * detail
        return branch.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def _temporal_stabilize_boundary(
        self,
        hybrid_video: torch.Tensor,
        boundary_mask: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        if hybrid_video.shape[2] <= 1:
            return hybrid_video

        stabilized = hybrid_video.clone()
        a = float(alpha)
        for i in range(1, stabilized.shape[2]):
            prev = stabilized[:, :, i - 1]
            curr = stabilized[:, :, i]
            seam = boundary_mask[:, :, i]
            blended = a * curr + (1.0 - a) * prev
            stabilized[:, :, i] = curr * (1.0 - seam) + blended * seam
        return stabilized

    def _validate_hybrid_renderer_budget(
        self,
        global_video: torch.Tensor,
        hybrid_video: torch.Tensor,
        mouth_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> None:
        if hybrid_video.shape[2] <= 1:
            return

        g = global_video.to(torch.float32)
        h = hybrid_video.to(torch.float32)

        dt_g = torch.abs(g[:, :, 1:] - g[:, :, :-1])
        dt_h = torch.abs(h[:, :, 1:] - h[:, :, :-1])
        seam = boundary_mask[:, :, 1:]
        seam_den = seam.mean().clamp_min(1e-6)
        flicker_g = (dt_g * seam).mean() / seam_den
        flicker_h = (dt_h * seam).mean() / seam_den
        flicker_ratio = (flicker_h / flicker_g.clamp_min(1e-6)).item()

        artifact_energy = (torch.abs(h - g) * mouth_mask).mean().item()
        self.metrics.record("hybrid_mouth_flicker_ratio", float(flicker_ratio))
        self.metrics.record("hybrid_mouth_artifact_energy", float(artifact_energy))
        self.metrics.record("hybrid_mouth_budget_ok", int(
            flicker_ratio <= float(self.hybrid_renderer_flicker_budget)
            and artifact_energy <= float(self.hybrid_renderer_artifact_budget)
        ))

        if flicker_ratio > float(self.hybrid_renderer_flicker_budget):
            loguru.logger.warning(
                "Hybrid mouth renderer flicker budget exceeded: ratio {:.4f} > {:.4f}",
                flicker_ratio,
                float(self.hybrid_renderer_flicker_budget),
            )
        if artifact_energy > float(self.hybrid_renderer_artifact_budget):
            loguru.logger.warning(
                "Hybrid mouth renderer artifact budget exceeded: {:.6f} > {:.6f}",
                artifact_energy,
                float(self.hybrid_renderer_artifact_budget),
            )

    def _apply_hybrid_mouth_renderer(
        self,
        decoded_video: torch.Tensor,
        mouth_zone_masks: Optional[torch.Tensor],
        resize_mode: str = "crop",
    ) -> torch.Tensor:
        if not self.hybrid_renderer_enabled or mouth_zone_masks is None:
            return decoded_video
        if decoded_video.ndim != 5:
            return decoded_video

        b, _, t, h, w = decoded_video.shape
        mouth_mask = self._prepare_mouth_zone_mask(
            mouth_zone_masks=mouth_zone_masks,
            batch_size=b,
            num_frames=t,
            height=h,
            width=w,
            device=decoded_video.device,
            dtype=decoded_video.dtype,
            resize_mode=resize_mode,
        )
        if mouth_mask is None:
            return decoded_video

        boundary_mask = self._compute_seam_boundary_mask(mouth_mask)
        mouth_branch = self._build_mouth_controlled_branch(
            decoded_video=decoded_video,
            strength=float(self.hybrid_renderer_mouth_strength),
        )
        hybrid = decoded_video * (1.0 - mouth_mask) + mouth_branch * mouth_mask
        hybrid = self._temporal_stabilize_boundary(
            hybrid_video=hybrid,
            boundary_mask=boundary_mask,
            alpha=float(self.hybrid_renderer_temporal_alpha),
        )
        self._validate_hybrid_renderer_budget(
            global_video=decoded_video,
            hybrid_video=hybrid,
            mouth_mask=mouth_mask,
            boundary_mask=boundary_mask,
        )
        return hybrid.contiguous()
