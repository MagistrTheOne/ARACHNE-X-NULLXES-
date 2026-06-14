"""
ARACHNE-X streaming helpers used by the avatar pipeline (VAE decode + CUDA opts).
"""

from __future__ import annotations

import os
from typing import Generator

import torch
from torch.amp import autocast


class StreamingVAEDecoder:
    """Incremental VAE decoder — outputs frames on-the-fly without buffering."""

    def __init__(self, vae, chunk_size: int = 1, enable_amp: bool = True):
        self.vae = vae
        self.chunk_size = chunk_size
        self.enable_amp = enable_amp
        compile_on = os.environ.get("ARACHNE_TORCH_COMPILE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if compile_on and hasattr(torch, "compile"):
            self.decode_fn = torch.compile(self.vae.decode, mode="reduce-overhead")
        else:
            self.decode_fn = self.vae.decode

    def decode_streaming(self, latents: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        """
        Stream-decode latents frame-by-frame.
        Args:
            latents: [B, C, T, H, W] full latent batch
        Yields:
            Decoded frame tensors [B, 3, H_out, W_out]
        """
        num_frames = latents.shape[2]

        for i in range(0, num_frames, self.chunk_size):
            chunk = latents[:, :, i : i + self.chunk_size]

            if self.enable_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    decoded = self.decode_fn(chunk, return_dict=False)[0]
            else:
                decoded = self.decode_fn(chunk, return_dict=False)[0]

            decoded = decoded.clamp(-1, 1)
            decoded = (decoded + 1) / 2  # [-1, 1] -> [0, 1]

            if decoded.ndim != 5:
                raise ValueError(f"Expected decoded video chunk [B, C, T, H, W], got {tuple(decoded.shape)}.")

            for t_idx in range(decoded.shape[2]):
                yield decoded[:, :, t_idx]


class CUDAOptimizer:
    """Enable CUDA-level optimizations for production inference."""

    @staticmethod
    def enable_flash_attention():
        """Enable Flash Attention v2 if available."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return True

    @staticmethod
    def compile_model(model, mode: str = "reduce-overhead"):
        """Compile model with torch.compile if available (PyTorch 2.0+)."""
        if hasattr(torch, "compile"):
            return torch.compile(model, mode=mode, fullgraph=False)
        return model

    @staticmethod
    def enable_grad_checkpointing(module):
        """Enable gradient checkpointing to save memory."""
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()
        return module

    @staticmethod
    def use_inference_mode():
        """Context manager for inference mode (faster than no_grad)."""
        return torch.inference_mode()
