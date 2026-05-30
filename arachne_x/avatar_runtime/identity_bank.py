"""Identity token bank helpers for the production avatar pipeline."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput


class IdentityBankMixin:
    """Identity enrollment, persistence, and prompt-token injection methods."""

    def _refresh_identity_tokens(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        identity_id: Optional[Union[int, List[int], torch.Tensor]],
        identity_strength: float,
        identity_negative_strength: float,
        batch_size: int,
        num_videos_per_prompt: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.identity_bank_enabled or identity_id is None:
            return prompt_embeds, negative_prompt_embeds
        if prompt_embeds is None or prompt_embeds.shape[2] < self.identity_tokens_per_id:
            return prompt_embeds, negative_prompt_embeds

        base_ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)
        expanded_ids: List[int] = []
        for idx in base_ids:
            expanded_ids.extend([idx] * num_videos_per_prompt)

        id_index = torch.tensor(
            expanded_ids,
            dtype=torch.long,
            device=self.identity_embedding.weight.device,
        )
        refreshed_tokens = self.identity_embedding(id_index).view(
            len(expanded_ids),
            self.identity_tokens_per_id,
            self.identity_token_dim,
        ).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype).unsqueeze(1)
        prompt_embeds[:, :, -self.identity_tokens_per_id :, :] = refreshed_tokens * float(identity_strength)

        if (
            negative_prompt_embeds is not None
            and negative_prompt_embeds.shape[2] >= self.identity_tokens_per_id
        ):
            negative_prompt_embeds[:, :, -self.identity_tokens_per_id :, :] = refreshed_tokens.to(
                device=negative_prompt_embeds.device,
                dtype=negative_prompt_embeds.dtype,
            ) * float(identity_negative_strength)

        return prompt_embeds, negative_prompt_embeds

    def _normalize_identity_ids(
        self,
        identity_id: Optional[Union[int, List[int], torch.Tensor]],
        batch_size: int,
    ) -> Optional[List[int]]:
        if identity_id is None:
            return None

        if isinstance(identity_id, int):
            ids = [identity_id] * batch_size
        elif isinstance(identity_id, torch.Tensor):
            ids = [int(x) for x in identity_id.detach().cpu().view(-1).tolist()]
        elif isinstance(identity_id, (list, tuple)):
            ids = [int(x) for x in identity_id]
        else:
            raise TypeError(
                f"`identity_id` must be int, list[int], torch.Tensor, or None. Got {type(identity_id)}."
            )

        if len(ids) == 1 and batch_size > 1:
            ids = ids * batch_size
        if len(ids) != batch_size:
            raise ValueError(
                f"`identity_id` length must be 1 or equal to batch size ({batch_size}), got {len(ids)}."
            )

        for idx in ids:
            if idx < 0 or idx >= self.identity_bank_size:
                raise ValueError(
                    f"`identity_id` {idx} is out of range [0, {self.identity_bank_size - 1}]."
                )
        return ids

    def _append_identity_tokens(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_attention_mask: Optional[torch.Tensor],
        identity_id: Optional[Union[int, List[int], torch.Tensor]],
        identity_strength: float,
        identity_negative_strength: float,
        batch_size: int,
        num_videos_per_prompt: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.identity_bank_enabled:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            )
        if identity_id is None:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            )
        if identity_strength < 0:
            raise ValueError(f"`identity_strength` must be >= 0, got {identity_strength}.")
        if identity_negative_strength < 0:
            raise ValueError(
                f"`identity_negative_strength` must be >= 0, got {identity_negative_strength}."
            )

        base_ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)
        expanded_ids: List[int] = []
        for idx in base_ids:
            expanded_ids.extend([idx] * num_videos_per_prompt)

        bank_device = self.identity_embedding.weight.device
        id_index = torch.tensor(expanded_ids, dtype=torch.long, device=bank_device)
        id_tokens = self.identity_embedding(id_index).view(
            len(expanded_ids),
            self.identity_tokens_per_id,
            self.identity_token_dim,
        )
        id_tokens = id_tokens.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        id_tokens = id_tokens.unsqueeze(1)  # [B, 1, N_id, D]

        pos_tokens = id_tokens * float(identity_strength)
        pos_mask = torch.ones(
            (pos_tokens.shape[0], pos_tokens.shape[2]),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        prompt_embeds = torch.cat([prompt_embeds, pos_tokens], dim=2)
        prompt_attention_mask = torch.cat([prompt_attention_mask, pos_mask], dim=1)

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is not None:
            neg_tokens = id_tokens.to(
                device=negative_prompt_embeds.device,
                dtype=negative_prompt_embeds.dtype,
            ) * float(identity_negative_strength)
            if identity_negative_strength > 0:
                neg_mask = torch.ones(
                    (neg_tokens.shape[0], neg_tokens.shape[2]),
                    dtype=negative_prompt_attention_mask.dtype,
                    device=negative_prompt_attention_mask.device,
                )
            else:
                neg_mask = torch.zeros(
                    (neg_tokens.shape[0], neg_tokens.shape[2]),
                    dtype=negative_prompt_attention_mask.dtype,
                    device=negative_prompt_attention_mask.device,
                )
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, neg_tokens], dim=2)
            negative_prompt_attention_mask = torch.cat([negative_prompt_attention_mask, neg_mask], dim=1)

        self.metrics.record("identity_tokens_appended", int(self.identity_tokens_per_id))
        self.metrics.record("identity_strength", float(identity_strength))
        self.metrics.record("identity_negative_strength", float(identity_negative_strength))
        self.metrics.record("identity_bank_active", 1)
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @torch.no_grad()
    def register_identity_from_latents(
        self,
        identity_id: Union[int, List[int], torch.Tensor],
        latents: torch.Tensor,
        momentum: float = 0.25,
    ) -> None:
        if not self.identity_bank_enabled:
            return
        if latents.ndim != 5:
            raise ValueError(f"`latents` must be [B, C, T, H, W], got shape {tuple(latents.shape)}.")
        if momentum < 0 or momentum > 1:
            raise ValueError(f"`momentum` must be in [0, 1], got {momentum}.")

        batch_size = latents.shape[0]
        ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)

        pooled = latents.to(torch.float32).mean(dim=(2, 3, 4))
        proj_device = next(self.identity_latent_projector.parameters()).device
        pooled = pooled.to(device=proj_device)
        projected = self.identity_latent_projector(pooled)  # [B, tokens*dim]

        with torch.no_grad():
            for b, idx in enumerate(ids):
                current = self.identity_embedding.weight[idx].detach().to(projected.dtype)
                observed = projected[b]
                updated = (1.0 - momentum) * current + momentum * observed
                cos = F.cosine_similarity(
                    current.unsqueeze(0),
                    observed.unsqueeze(0),
                    dim=-1,
                ).item()
                self.identity_embedding.weight[idx].data.copy_(
                    updated.to(self.identity_embedding.weight.dtype)
                )
                self.metrics.record("identity_bank_update_cosine", float(cos))
                self.metrics.record("identity_bank_updated_id", int(idx))

    @torch.no_grad()
    def save_identity_bank(self, path: str) -> str:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "version": 1,
            "timestamp": time.time(),
            "identity_bank_size": int(self.identity_bank_size),
            "identity_tokens_per_id": int(self.identity_tokens_per_id),
            "identity_token_dim": int(self.identity_token_dim),
            "identity_embedding": self.identity_embedding.weight.detach().cpu(),
            "identity_latent_projector": self.identity_latent_projector.state_dict(),
        }
        torch.save(payload, path)
        return path

    @torch.no_grad()
    def load_identity_bank(self, path: str, strict: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        required_keys = {
            "version",
            "identity_bank_size",
            "identity_tokens_per_id",
            "identity_token_dim",
            "identity_embedding",
        }
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"Identity bank file is missing keys: {sorted(missing)}")

        loaded_bank = payload["identity_embedding"]
        if not isinstance(loaded_bank, torch.Tensor):
            raise ValueError("`identity_embedding` must be a torch.Tensor.")

        expected_shape = (
            self.identity_bank_size,
            self.identity_tokens_per_id * self.identity_token_dim,
        )
        loaded_shape = tuple(loaded_bank.shape)
        if strict and loaded_shape != expected_shape:
            raise ValueError(
                f"Identity bank shape mismatch. Expected {expected_shape}, got {loaded_shape}."
            )

        rows = min(expected_shape[0], loaded_bank.shape[0])
        cols = min(expected_shape[1], loaded_bank.shape[1])
        self.identity_embedding.weight.data[:rows, :cols].copy_(
            loaded_bank[:rows, :cols].to(self.identity_embedding.weight.dtype)
        )

        if "identity_latent_projector" in payload:
            self.identity_latent_projector.load_state_dict(payload["identity_latent_projector"], strict=False)

        self.metrics.record("identity_bank_loaded_rows", int(rows))
        self.metrics.record("identity_bank_loaded_cols", int(cols))
        return {
            "rows_loaded": int(rows),
            "cols_loaded": int(cols),
            "strict": bool(strict),
            "source": path,
        }

    @torch.no_grad()
    def enroll_identity_from_image(
        self,
        image: PipelineImageInput,
        identity_id: Union[int, List[int], torch.Tensor],
        resolution: Literal["720p"] = "720p",
        resize_mode: str = "crop",
        momentum: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Register (or update) one-shot identity slot(s) directly from reference image(s)
        without running a full diffusion sampling loop.
        """
        if resize_mode not in ("default", "crop"):
            raise ValueError(f"Unsupported resize_mode {resize_mode}, and you can only choose from [default, crop]")
        if identity_id is None:
            raise ValueError("`identity_id` is required for identity enrollment.")

        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)

        height, width = self.get_condition_shape(
            image,
            resolution,
            scale_factor_spatial=scale_factor_spatial,
        )
        image_tensor = self.video_processor.preprocess(
            image,
            height=height,
            width=width,
            resize_mode=resize_mode,
        ).to(device=self.device, dtype=self.dit.dtype)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        batch_size = image_tensor.shape[0]
        latents = self.prepare_latents(
            image=image_tensor,
            batch_size=batch_size,
            num_channels_latents=self.dit.config.in_channels,
            height=height,
            width=width,
            num_frames=1,
            num_cond_frames=1,
            dtype=torch.float32,
            device=self.device,
            generator=None,
            latents=None,
        )
        cond_latents = latents[:, :, :1]
        self.register_identity_from_latents(
            identity_id=identity_id,
            latents=cond_latents,
            momentum=momentum,
        )
        normalized_ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)
        self.metrics.record("identity_enroll_batch_size", int(batch_size))
        self.metrics.record("identity_enroll_momentum", float(momentum))
        return {
            "identity_ids": normalized_ids,
            "batch_size": int(batch_size),
            "height": int(height),
            "width": int(width),
            "momentum": float(momentum),
        }
