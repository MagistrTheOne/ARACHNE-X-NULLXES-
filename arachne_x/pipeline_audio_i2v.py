"""
Experimental audio-conditioned I2V pipeline (lab only).

Wraps frozen base VIDEO DiT with :class:`AudioConditionedVideoDiTWrapper`.
Does not modify avatar / Elena / production runtime paths.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Literal, Optional, Union

import loguru
import torch
from diffusers.image_processor import PipelineImageInput
from tqdm import tqdm

from .modules.audio_conditioning import (
    AudioConditionedVideoDiTWrapper,
    AudioConditioningAdapter,
    AudioEncoderRuntime,
    build_windowed_audio_emb,
    encode_wav2vec_audio,
    load_audio_conditioning_adapter,
)
from .pipeline_arachne_x_video import LongCatVideoPipeline, release_modules_for_denoise, restore_modules_after_denoise, torch_gc


class AudioConditionedI2VPipeline(LongCatVideoPipeline):
    """
    Base VIDEO pipeline plus optional audio-conditioning adapter and wav2vec runtime.
    """

    def __init__(
        self,
        *args,
        audio_encoder_runtime: Optional[AudioEncoderRuntime] = None,
        audio_adapter: Optional[AudioConditioningAdapter] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.audio_encoder_runtime = audio_encoder_runtime
        self.audio_adapter = audio_adapter
        self.audio_conditioning_scale = 0.0
        self._dit_wrapper: Optional[AudioConditionedVideoDiTWrapper] = None
        self._refresh_dit_wrapper()

    def _refresh_dit_wrapper(self) -> None:
        self._dit_wrapper = AudioConditionedVideoDiTWrapper(self.dit, self.audio_adapter)

    def load_audio_conditioning_adapter(self, path: str, *, strict: bool = True) -> None:
        self.audio_adapter = load_audio_conditioning_adapter(path, device=str(self.device), strict=strict)
        self.audio_adapter.to(self.device)
        self._refresh_dit_wrapper()
        loguru.logger.info(
            "[audio-i2v] loaded adapter blocks={} trainable_params={}",
            self.audio_adapter.block_indices,
            self.audio_adapter.trainable_parameter_count(),
        )

    def build_audio_emb_from_path(
        self,
        audio_path: str,
        num_frames: int,
        *,
        sample_rate: int = 16000,
        embedding_fps: Optional[float] = None,
    ) -> torch.Tensor:
        if self.audio_encoder_runtime is None:
            raise RuntimeError("audio_encoder_runtime is not configured on pipeline")
        from .modules.audio_conditioning.audio_encode import load_audio_from_path

        speech = load_audio_from_path(audio_path, sample_rate=sample_rate)
        fps = float(embedding_fps) if embedding_fps is not None else float(16 * self.vae_scale_factor_temporal)
        full_emb = encode_wav2vec_audio(
            self.audio_encoder_runtime,
            speech,
            fps=fps,
            sample_rate=sample_rate,
        )
        audio_window = 5
        if self.audio_adapter is not None:
            audio_window = self.audio_adapter.config.audio_window
        return build_windowed_audio_emb(
            full_emb,
            num_frames,
            audio_window=audio_window,
            vae_stride=int(self.vae_scale_factor_temporal),
            device=self.device,
        )

    def generate_audio_i2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        audio_path: Optional[str] = None,
        audio_emb: Optional[torch.Tensor] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_conditioning_scale: float = 0.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        embedding_fps: Optional[float] = None,
    ):
        if float(audio_conditioning_scale) == 0.0:
            loguru.logger.info("[audio-i2v] scale=0 -> delegating to base generate_i2v")
            return self.generate_i2v(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                use_distill=use_distill,
                guidance_scale=text_guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                output_type=output_type,
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
            )

        if audio_emb is None:
            if audio_path is None:
                raise ValueError("generate_audio_i2v requires audio_path or audio_emb when scale > 0")
            audio_emb = self.build_audio_emb_from_path(
                audio_path,
                num_frames,
                embedding_fps=embedding_fps,
            )

        self.audio_conditioning_scale = float(audio_conditioning_scale)
        loguru.logger.info(
            "[audio-i2v] scale={} frames={} adapter_blocks={}",
            self.audio_conditioning_scale,
            num_frames,
            tuple(self.audio_adapter.block_indices) if self.audio_adapter else (),
        )

        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        height, width = self.get_condition_shape(image, resolution, scale_factor_spatial=scale_factor_spatial)
        self.check_inputs(prompt, negative_prompt, height, width, scale_factor_spatial)

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = text_guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        device = self.device

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        dit_dtype = self.dit.dtype

        from arachne_x.context_parallel import context_parallel_util

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_embeds = torch.zeros(
                [batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device
            )
            prompt_attention_mask = torch.zeros([batch_size, max_sequence_length], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros(
                    [batch_size, 1, max_sequence_length, caption_channels], dtype=dit_dtype, device=device
                )
                negative_prompt_attention_mask = torch.zeros(
                    [batch_size, max_sequence_length], dtype=torch.int64, device=device
                )
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        image = self.video_processor.preprocess(image, height=height, width=width)
        image = image.to(device=device, dtype=prompt_embeds.dtype)
        num_channels_latents = self.dit.config.in_channels
        latents = self.prepare_latents(
            image=image,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=1,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        audio_null = torch.zeros_like(audio_emb)
        do_audio_cfg = self.audio_conditioning_scale > 1.0

        release_modules_for_denoise(self)
        if self.audio_encoder_runtime is not None:
            self.audio_encoder_runtime.audio_encoder = self.audio_encoder_runtime.audio_encoder.to(
                "cpu", non_blocking=True
            )
            self.audio_encoder_runtime.device = "cpu"
            gc.collect()
            torch_gc()
            loguru.logger.info("[vram] released wav2vec runtime for denoise")

        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        dit_forward = self._dit_wrapper
        assert dit_forward is not None

        pos_prompt_embeds = prompt_embeds[-batch_size:] if self.do_classifier_free_guidance else prompt_embeds
        pos_prompt_mask = prompt_attention_mask[-batch_size:] if self.do_classifier_free_guidance else prompt_attention_mask
        neg_prompt_embeds = prompt_embeds[:batch_size] if self.do_classifier_free_guidance else prompt_embeds
        neg_prompt_mask = prompt_attention_mask[:batch_size] if self.do_classifier_free_guidance else prompt_attention_mask

        with tqdm(total=len(timesteps), desc="Denoising (audio_i2v)") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                timestep_base = t.expand(latents.shape[0]).to(dit_dtype)
                timestep_base = timestep_base.unsqueeze(-1).repeat(1, latents.shape[2])
                timestep_base[:, :1] = 0

                if self.do_classifier_free_guidance and do_audio_cfg:
                    noise_pred_uncond = dit_forward(
                        hidden_states=latents,
                        timestep=timestep_base,
                        encoder_hidden_states=neg_prompt_embeds,
                        encoder_attention_mask=neg_prompt_mask,
                        num_cond_latents=1,
                        audio_embs=audio_null,
                        audio_conditioning_scale=self.audio_conditioning_scale,
                    )
                    noise_pred_text = dit_forward(
                        hidden_states=latents,
                        timestep=timestep_base,
                        encoder_hidden_states=pos_prompt_embeds,
                        encoder_attention_mask=pos_prompt_mask,
                        num_cond_latents=1,
                        audio_embs=audio_null,
                        audio_conditioning_scale=self.audio_conditioning_scale,
                    )
                    noise_pred_audio = dit_forward(
                        hidden_states=latents,
                        timestep=timestep_base,
                        encoder_hidden_states=pos_prompt_embeds,
                        encoder_attention_mask=pos_prompt_mask,
                        num_cond_latents=1,
                        audio_embs=audio_emb,
                        audio_conditioning_scale=self.audio_conditioning_scale,
                    )
                    noise_pred = (
                        noise_pred_uncond
                        + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                        + self.audio_conditioning_scale * (noise_pred_audio - noise_pred_text)
                    )
                elif self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                    timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    timestep[:, :1] = 0
                    noise_pred = dit_forward(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        num_cond_latents=1,
                        audio_embs=torch.cat([audio_emb, audio_emb], dim=0),
                        audio_conditioning_scale=self.audio_conditioning_scale,
                    )
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    b = noise_pred_cond.shape[0]
                    positive = noise_pred_cond.reshape(b, -1)
                    negative = noise_pred_uncond.reshape(b, -1)
                    st_star = self.optimized_scale(positive, negative).view(b, 1, 1, 1)
                    noise_pred = noise_pred_uncond * st_star + text_guidance_scale * (
                        noise_pred_cond - noise_pred_uncond * st_star
                    )
                else:
                    noise_pred = dit_forward(
                        hidden_states=latents,
                        timestep=timestep_base,
                        encoder_hidden_states=pos_prompt_embeds,
                        encoder_attention_mask=pos_prompt_mask,
                        num_cond_latents=1,
                        audio_embs=audio_emb,
                        audio_conditioning_scale=self.audio_conditioning_scale,
                    )

                noise_pred = -noise_pred
                latents[:, :, 1:] = self.scheduler.step(
                    noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            return latents

        restore_modules_after_denoise(self)
        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self.vae.decode(latents, return_dict=False)[0]
        return self.video_processor.postprocess_video(output_video, output_type=output_type)
