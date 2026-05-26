"""Latent normalization, VAE conditioning, and scheduler helper methods."""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch

from arachne_x.utils.bukcet_config import get_bucket_config

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class LatentUtilsMixin:
    """VAE latent helpers preserved from the monolithic avatar pipeline."""

    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        num_cond_frames_added: int = 0,
        need_encode: bool = True
    ) -> torch.Tensor:
        if (image is not None) and (video is not None):
            raise ValueError("Cannot provide both `image and video` at the same time. Please provide only one.")
        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
        else:
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # Generate random noise with shape latent_shape
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        if image is not None or video is not None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
            condition_data = image if image is not None else video
            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal

            if need_encode:
                is_image = image is not None
                cond_latents = []
                for i in range(batch_size):
                    gen = generator[i] if isinstance(generator, list) else generator
                    if is_image:
                        encoded_input = condition_data[i].unsqueeze(0).unsqueeze(2)
                    else:
                        encoded_input = condition_data[i][:, -(num_cond_frames-num_cond_frames_added):].unsqueeze(0)
                    if num_cond_frames_added > 0:
                        pad_front = encoded_input[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
                        encoded_input = torch.cat([pad_front, encoded_input], dim=2)
                    assert encoded_input.shape[2] == num_cond_frames
                    latent = retrieve_latents(self.vae.encode(encoded_input), gen, sample_mode="argmax")
                    cond_latents.append(latent)

                cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
                cond_latents = self.normalize_latents(cond_latents)
            else:
                cond_latents = condition_data[:, :, -num_cond_latents:]
            
            latents[:, :, :num_cond_latents] = cond_latents

        return latents

    def get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool=False):
        if use_distill:
            distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32)
            distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()
            
            inference_indices = np.linspace(0, self.num_distill_sample_steps, num=sampling_steps, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            
            sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / self.num_timesteps
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps)
        sigmas = sigmas.to(torch.float32)
        return sigmas

    def get_condition_shape(self, condition, resolution, scale_factor_spatial=32):
        bucket_config = get_bucket_config(resolution, scale_factor_spatial=scale_factor_spatial)

        obj = condition[0] if isinstance(condition, list) and condition else condition
        try:
            height = getattr(obj, "height")
            width = getattr(obj, "width")
        except AttributeError:
            raise ValueError("Unsupported condition type")

        ratio = height / width
        # Find the closest bucket
        closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        return target_h, target_w
    
    def optimized_scale(self, positive_flat, negative_flat):
        """ from CFG-zero paper
        """
        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
        st_star = dot_product / squared_norm
        return st_star
    
    def normalize_latents(self, latents):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return (latents - latents_mean) * latents_std

    def denormalize_latents(self, latents):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return latents / latents_std + latents_mean
