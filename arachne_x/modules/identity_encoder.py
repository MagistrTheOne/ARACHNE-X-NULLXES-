"""
Frozen identity embedding encoder for avatar auxiliary training.

Uses DINOv2 by default (semantic face structure). Swap to ArcFace weights later
via ``ARACHNE_IDENTITY_ENCODER=arcface`` when ONNX/torch weights are available.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


class FrozenIdentityEncoder(nn.Module):
    """Extract L2-normalized identity embeddings from RGB face images."""

    def __init__(
        self,
        backend: Literal["dino", "arcface"] = "dino",
        model_name: str = "facebook/dinov2-base",
    ):
        super().__init__()
        self.backend = backend
        self.model_name = model_name
        self._encoder = self._build_encoder(backend, model_name)
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        self.register_buffer(
            "_img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def from_env(default: str = "dino") -> "FrozenIdentityEncoder":
        backend = os.environ.get("ARACHNE_IDENTITY_ENCODER", default).strip().lower()
        if backend not in ("dino", "arcface"):
            backend = "dino"
        return FrozenIdentityEncoder(backend=backend)  # type: ignore[arg-type]

    def _build_encoder(
        self, backend: str, model_name: str
    ) -> nn.Module:
        if backend == "arcface":
            raise NotImplementedError(
                "ArcFace backend requires ARACHNE_ARCFACE_WEIGHTS; use dino for now."
            )
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name)
        model.eval()
        return model

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.min() < 0:
            images = (images + 1.0) * 0.5
        if images.shape[-1] < 224 or images.shape[-2] < 224:
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )
        mean = self._img_mean.to(device=images.device, dtype=images.dtype)
        std = self._img_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] or [B, T, C, H, W] in [0,1] or [-1,1]
        Returns:
            embeddings: [B, D] or [B, T, D], L2-normalized
        """
        if images.dim() == 5:
            b, t, c, h, w = images.shape
            flat = images.reshape(b * t, c, h, w)
            emb = self._encode_flat(flat)
            return F.normalize(emb, p=2, dim=-1).view(b, t, -1)
        return F.normalize(self._encode_flat(images), p=2, dim=-1)

    def _encode_flat(self, images: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(images)
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self._encoder(pixel_values=x.float())
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]

    @staticmethod
    def load_reference_image(path: str, device: torch.device) -> torch.Tensor:
        if Image is None:
            raise RuntimeError("Pillow is required to load reference_image")
        img = Image.open(path).convert("RGB")
        import numpy as np

        arr = np.asarray(img).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return tensor
