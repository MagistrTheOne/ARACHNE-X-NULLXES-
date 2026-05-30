import types
from typing import Any, Dict, List, Optional, Union, Literal, Tuple

import time
import torch
import torch.nn as nn
import loguru
import numpy as np
from tqdm import tqdm 
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from transformers import AutoTokenizer, UMT5EncoderModel

from arachne_x.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from arachne_x.modules.autoencoder_kl_wan import AutoencoderKLWan
from arachne_x.modules.avatar.arachne_avatar_dit import LongCatVideoAvatarTransformer3DModel
from arachne_x.context_parallel import context_parallel_util

# -------- avatar related --------
from arachne_x.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from arachne_x.utils.monitoring import MetricsLogger
from arachne_x.streaming_inference import StreamingVAEDecoder, CUDAOptimizer
from transformers import Wav2Vec2FeatureExtractor
from diffusers.image_processor import is_valid_image, is_valid_image_imagelist
import warnings

from arachne_x.avatar_runtime.audio_conditioning import AudioConditioningMixin
from arachne_x.avatar_runtime.hybrid_renderer import HybridRendererMixin
from arachne_x.avatar_runtime.identity_bank import IdentityBankMixin
from arachne_x.avatar_runtime.kv_cache import KVCacheMixin
from arachne_x.avatar_runtime.latent_utils import LatentUtilsMixin, retrieve_latents
from arachne_x.avatar_runtime.text_conditioning import TextConditioningMixin


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class GenerationInterrupted(Exception):
    """
    Raised to abort diffusion denoising loops during realtime interruptions.

    This avoids silent partial updates and reduces wasted compute.
    """



def preprocess_video(self, video, height: Optional[int] = None, width: Optional[int] = None, resize_mode: Optional[str] = 'crop') -> torch.Tensor:
    r"""
    hack diffusers.video_processor.VideoProcessor to support the parameter of resize_mode 
    """
    if isinstance(video, list) and isinstance(video[0], np.ndarray) and video[0].ndim == 5:
        warnings.warn(
            "Passing `video` as a list of 5d np.ndarray is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 5d np.ndarray",
            FutureWarning,
        )
        video = np.concatenate(video, axis=0)
    if isinstance(video, list) and isinstance(video[0], torch.Tensor) and video[0].ndim == 5:
        warnings.warn(
            "Passing `video` as a list of 5d torch.Tensor is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 5d torch.Tensor",
            FutureWarning,
        )
        video = torch.cat(video, axis=0)

    # ensure the input is a list of videos:
    # - if it is a batch of videos (5d torch.Tensor or np.ndarray), it is converted to a list of videos (a list of 4d torch.Tensor or np.ndarray)
    # - if it is a single video, it is converted to a list of one video.
    if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim == 5:
        video = list(video)
    elif isinstance(video, list) and is_valid_image(video[0]) or is_valid_image_imagelist(video):
        video = [video]
    elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
        video = video
    else:
        raise ValueError(
            "Input is in incorrect format. Currently, we only support numpy.ndarray, torch.Tensor, PIL.Image.Image"
        )

    video = torch.stack([self.preprocess(img, height=height, width=width, resize_mode=resize_mode) for img in video], dim=0)
    video = video.permute(0, 2, 1, 3, 4)

    return video

class ArachneXVideoAvatarPipeline(
    TextConditioningMixin,
    IdentityBankMixin,
    AudioConditioningMixin,
    LatentUtilsMixin,
    KVCacheMixin,
    HybridRendererMixin,
):
    r"""
    Pipeline for text-to-video generation using LongCatVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        dit: LongCatVideoAvatarTransformer3DModel,
        audio_encoder: Wav2Vec2ModelWrapper,
        wav2vec_feature_extractor: Wav2Vec2FeatureExtractor
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.dit = dit 
        self.device = "cuda"

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8 
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.video_processor.preprocess_video = types.MethodType(preprocess_video, self.video_processor)

        self._num_timesteps = 1000
        self._num_distill_sample_steps = 50
        # Baseline runtime state. Generation calls overwrite guidance scales,
        # but mixins/properties must be safe immediately after construction.
        self._text_guidance_scale = 4.0
        self._audio_guidance_scale = 4.0
        self._emotion_guidance_scale = 0.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        self.kv_cache_dict = None

        self.audio_encoder=audio_encoder
        self.wav2vec_feature_extractor = wav2vec_feature_extractor
        # Keep avatar audio conditioning deterministic: production uses wav2vec
        # embeddings directly. Removed random multi-stream and pseudo-phoneme
        # adapters from the runtime graph.
        self.audio_processor = None
        self.multi_stream_fusion_proj = None
        self.multi_stream_fusion_scale = 0.0
        # Pseudo-phoneme conditioning was removed from the avatar runtime graph.
        # No phoneme aligner / projection / alignment head exists in prod; the CLI
        # flags (--disable_phoneme_conditioning, --phoneme_stream_scale) are no-ops
        # and runtime/inference_engine.py force-off is guarded by hasattr().
        audio_embed_dim = 768
        # Step 4: explicit emotion control channel with lip-sync safety guard.
        self.emotion_enabled = True
        self.emotion_num_classes = 8
        self.emotion_default_id = 0
        self.emotion_default_intensity = 0.0
        self.emotion_lipsync_guard_ratio = 0.35
        self.emotion_label_to_id = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "surprised": 4,
            "fearful": 5,
            "disgusted": 6,
            "calm": 7,
        }
        self.emotion_embedding = nn.Embedding(self.emotion_num_classes, audio_embed_dim)
        nn.init.normal_(self.emotion_embedding.weight, mean=0.0, std=0.02)
        self.emotion_proj = nn.Sequential(
            nn.Linear(audio_embed_dim, audio_embed_dim),
            nn.SiLU(),
            nn.Linear(audio_embed_dim, audio_embed_dim),
        )
        # Step 5: hybrid renderer for controlled mouth zone.
        self.hybrid_renderer_enabled = True
        self.hybrid_renderer_mouth_strength = 0.35
        self.hybrid_renderer_blur_passes = 2
        self.hybrid_renderer_temporal_alpha = 0.70
        self.hybrid_renderer_flicker_budget = 1.40
        self.hybrid_renderer_artifact_budget = 0.08
        # Budget validation uses .item() (GPU sync); keep off on the realtime path.
        self.hybrid_renderer_metrics_verbose = False
        self.metrics = MetricsLogger()
        self.runtime_sampling_metrics = None

        # Identity token bank (Step 2): learnable per-identity vectors injected
        # into text conditioning as extra tokens.
        self.identity_bank_enabled = True
        self.identity_bank_size = 1024
        self.identity_tokens_per_id = 4
        self.identity_token_dim = int(self.text_encoder.config.d_model)
        self.identity_embedding = nn.Embedding(
            self.identity_bank_size,
            self.identity_tokens_per_id * self.identity_token_dim,
        )
        nn.init.normal_(self.identity_embedding.weight, mean=0.0, std=0.02)

        # Phase B: optional planning tokens (before identity bank); disabled by default.
        self.planning_enabled = False
        self.planning_tokens_count = 4
        self._planning_token_head: Optional[nn.Module] = None
        latent_dim = int(getattr(self.vae.config, "z_dim", 16))
        self.identity_latent_projector = nn.Sequential(
            nn.Linear(latent_dim, self.identity_token_dim),
            nn.SiLU(),
            nn.Linear(self.identity_token_dim, self.identity_tokens_per_id * self.identity_token_dim),
        )
        self.identity_default_strength = 1.0
        self.identity_default_negative_strength = 0.0
        
        self.streaming_enabled = True
        # Temporal compression memory (Step 1): keep a recent sliding window and
        # summarize older conditioning frames inside KV-cache for long-context AVC.
        self.temporal_memory_enabled = True
        self.temporal_memory_window_frames = 8
        self.temporal_memory_summary_frames = 2
        
        # CUDA optimizations for H200
        CUDAOptimizer.enable_flash_attention()
        if hasattr(torch, 'compile'):
            self.dit = CUDAOptimizer.compile_model(self.dit, mode='reduce-overhead')
            self.vae = CUDAOptimizer.compile_model(self.vae, mode='reduce-overhead')


    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        scale_factor_spatial
    ):
        # Check height and width divisibility
        if height % scale_factor_spatial != 0 or width % scale_factor_spatial != 0:
            raise ValueError(f"`height and width` have to be divisible by {scale_factor_spatial} but are {height} and {width}.")

        # Check prompt validity
        if prompt is None:
            raise ValueError("Cannot leave `prompt` undefined.")
        
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt has to be of type str or list` but is {type(prompt)}")
        
        # Check negative prompt validity
        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt has to be of type str or list` but is {type(negative_prompt)}")
        

    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale
    
    @property
    def audio_guidance_scale(self):
        return self._audio_guidance_scale

    @property
    def emotion_guidance_scale(self):
        return self._emotion_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return (
            self._text_guidance_scale > 1.0
            or self._audio_guidance_scale > 1.0
            or self._emotion_guidance_scale > 0.0
        )

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def num_distill_sample_steps(self):
        return self._num_distill_sample_steps
    
    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs
    

    def _predict_avatar_noise(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        audio_embs: torch.Tensor,
        num_cond_latents: Optional[int] = None,
        kv_cache_dict: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_ref_latents: int = 0,
        ref_img_index: Optional[int] = None,
        mask_frame_range: Optional[int] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "audio_embs": audio_embs,
        }
        if num_cond_latents is not None:
            kwargs["num_cond_latents"] = num_cond_latents
        if kv_cache_dict is not None:
            kwargs["kv_cache_dict"] = kv_cache_dict
        if num_ref_latents:
            kwargs["num_ref_latents"] = num_ref_latents
        if ref_img_index is not None:
            kwargs["ref_img_index"] = ref_img_index
        if mask_frame_range is not None:
            kwargs["mask_frame_range"] = mask_frame_range
        if ref_target_masks is not None:
            kwargs["ref_target_masks"] = ref_target_masks
        # torch.compile + CUDAGraphs can reuse internal output buffers across
        # sequential CFG passes (uncond/text/audio), so we mark a fresh step and
        # detach the returned tensor from any reusable graph-managed storage.
        compiler_ns = getattr(torch, "compiler", None)
        if compiler_ns is not None and hasattr(compiler_ns, "cudagraph_mark_step_begin"):
            compiler_ns.cudagraph_mark_step_begin()

        rsm = getattr(self, "runtime_sampling_metrics", None)
        if rsm is not None:
            rsm.record_dit_forward(1)

        noise_pred = self.dit(**kwargs)
        if isinstance(noise_pred, torch.Tensor):
            return noise_pred.clone()
        return noise_pred





    




    @torch.no_grad()
    def generate_at2v(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # avatar related params
        audio_emb: torch.Tensor = None,
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
        resize_mode: Optional[str] = "crop",
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            prompt (`str or List[str]`):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            height (`int`, *optional*, defaults to 480):
                Height of each video frame. Must be divisible by 16.
            width (`int`, *optional*, defaults to 832):
                Width of each video frame. Must be divisible by 16.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            identity_id (`int` or `List[int]`, *optional*):
                Identity slot index (or per-sample indices) in the learnable identity token bank.
            identity_strength (`float`, *optional*, defaults to 1.0):
                Scale applied to identity tokens for conditioned branch.
            identity_negative_strength (`float`, *optional*, defaults to 0.0):
                Scale applied to identity tokens for unconditioned branch.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if emotion_guidance_scale > 0 and (emotion_id is None or emotion_intensity <= 0):
            loguru.logger.warning(
                "Emotion guidance is enabled but emotion control is missing; disabling emotion guidance for this call."
            )
            emotion_guidance_scale = 0.0

        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._emotion_guidance_scale = float(emotion_guidance_scale)
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)


        # 3. Encode inputs
        dit_dtype = self.dit.dtype
        identity_token_count = (
            self.identity_tokens_per_id
            if self.identity_bank_enabled and identity_id is not None
            else 0
        )

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
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._append_identity_tokens(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                identity_id=identity_id,
                identity_strength=identity_strength,
                identity_negative_strength=identity_negative_strength,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_seq_len = max_sequence_length + identity_token_count
            effective_batch_size = batch_size * num_videos_per_prompt
            prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_base_embs = self._prepare_audio_emb_for_dit(
            audio_emb,
            num_frames=num_frames,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_cond_embs, emotion_active = self._apply_emotion_channel(
            audio_emb=audio_base_embs,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_guidance_embs = None
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_base_embs)
            if emotion_active and self.emotion_guidance_scale > 0.0:
                audio_guidance_embs = audio_base_embs
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)

        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.dit.config.in_channels
            
        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        # 6. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        start_time = time.time()
        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    raise GenerationInterrupted()

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)

                noise_pred_cond = self._predict_avatar_noise(
                    hidden_states=latents,
                    timestep=timestep[: latents.shape[0]],
                    encoder_hidden_states=prompt_embeds[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_attention_mask,
                    audio_embs=audio_cond_embs[latents.shape[0] :] if self.do_classifier_free_guidance else audio_cond_embs,
                )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    noise_pred_uncond = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        audio_embs=audio_unond_embs,
                    )
                    noise_pred_text = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                        encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                        audio_embs=audio_unond_embs,
                    )

                    if emotion_active and self.emotion_guidance_scale > 0.0 and audio_guidance_embs is not None:
                        noise_pred_audio = self._predict_avatar_noise(
                            hidden_states=latents,
                            timestep=timestep_uncond,
                            encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                            encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                            audio_embs=audio_guidance_embs,
                        )
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_audio - noise_pred_text)
                            + self.emotion_guidance_scale * (noise_pred_cond - noise_pred_audio)
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_cond - noise_pred_text)
                        )
                else:
                    noise_pred = noise_pred_cond

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        total_time = time.time() - start_time
        try:
            self.metrics.record('denoise_seconds', total_time)
            self.metrics.record('denoise_p95', total_time)
        except Exception as exc:
            loguru.logger.debug("Metric logging failed; continuing. Error: {}", exc)

        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self.vae.decode(latents, return_dict=False)[0]
        output_video = self._apply_hybrid_mouth_renderer(
            decoded_video=output_video,
            mouth_zone_masks=mouth_zone_masks,
            resize_mode=resize_mode,
        )
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else:
            return output_video
    

    @torch.no_grad()
    def generate_ai2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # avatar related params
        audio_emb: torch.Tensor = None,
        ref_target_masks: torch.Tensor = None,
        resize_mode: Optional[str] = "crop", # "default" / "crop"
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        update_identity_bank: bool = False,
        identity_update_momentum: float = 0.25,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
        use_cfg_zero: bool = False,
        use_kv_cache: bool = False,
        reuse_kv_cache: bool = False,
        offload_kv_cache: bool = False,
        refresh_identity_tokens: bool = False,
        silence_gate: bool = True,
    ):
        r"""
        Generates video frames from an input image and text prompt using diffusion process.

        Args:
            image (`PipelineImageInput`):
                Input image for video generation.
            prompt (`str or List[str]`, *optional*):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            resolution (`Literal["480p", "720p"]`, *optional*, defaults to "480p"):
                Target video resolution. Determines output frame size.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            ref_target_masks(`torch.Tensor`, *optional*, defaults to None):
                Mask used in dual-speaker audio-driven mode.
            resize_mode(`str`, *optional*):
                Output format type. "default" for resize, "crop" for shorter-length resize and centercrop.
            identity_id (`int` or `List[int]`, *optional*):
                Identity slot index (or per-sample indices) in the learnable identity token bank.
            identity_strength (`float`, *optional*, defaults to 1.0):
                Scale applied to identity tokens for conditioned branch.
            identity_negative_strength (`float`, *optional*, defaults to 0.0):
                Scale applied to identity tokens for unconditioned branch.
            update_identity_bank (`bool`, *optional*, defaults to False):
                Update the selected identity slot(s) from current conditioning latents.
            identity_update_momentum (`float`, *optional*, defaults to 0.25):
                EMA update ratio for identity bank writes.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        height, width = self.get_condition_shape(image, resolution, scale_factor_spatial=scale_factor_spatial)
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )
        assert resize_mode in ['default', 'crop'], f"Unsupported resize_mode {resize_mode}, and you can only choose from [default, crop]"

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if emotion_guidance_scale > 0 and (emotion_id is None or emotion_intensity <= 0):
            loguru.logger.warning(
                "Emotion guidance is enabled but emotion control is missing; disabling emotion guidance for this call."
            )
            emotion_guidance_scale = 0.0


        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._emotion_guidance_scale = float(emotion_guidance_scale)
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)


        # 3. Encode inputs
        dit_dtype = self.dit.dtype
        audio_duration_sec = None
        if audio_emb is not None and hasattr(audio_emb, "shape"):
            try:
                audio_duration_sec = float(audio_emb.shape[0]) / 32.0
            except Exception:
                audio_duration_sec = None
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            _planning_token_count,
            _identity_token_count,
        ) = self._encode_prompt_with_avatar_tokens(
            prompt=prompt,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            dit_dtype=dit_dtype,
            device=device,
            identity_id=identity_id,
            identity_strength=identity_strength,
            identity_negative_strength=identity_negative_strength,
            audio_duration_sec=audio_duration_sec,
        )

        if refresh_identity_tokens and identity_id is not None and _identity_token_count > 0:
            prompt_embeds, negative_prompt_embeds = self._refresh_identity_tokens(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                identity_id=identity_id,
                identity_strength=identity_strength,
                identity_negative_strength=identity_negative_strength,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
            )

        audio_base_embs = self._prepare_audio_emb_for_dit(
            audio_emb,
            num_frames=num_frames,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        if silence_gate:
            from arachne_x.runtime.audio_motion_gate import apply_audio_motion_gate

            audio_guidance_scale, _gate_meta = apply_audio_motion_gate(
                audio_base_embs, float(audio_guidance_scale)
            )
            rsm_gate = getattr(self, "runtime_sampling_metrics", None)
            if rsm_gate is not None:
                rsm_gate.silence_ratio = _gate_meta.get("silence_ratio")
                rsm_gate.audio_guidance_scale_effective = _gate_meta.get("audio_guidance_scale_effective")

        audio_cond_embs, emotion_active = self._apply_emotion_channel(
            audio_emb=audio_base_embs,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_guidance_embs = None
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_base_embs)
            if emotion_active and self.emotion_guidance_scale > 0.0:
                audio_guidance_embs = audio_base_embs
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)
        
        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        image = self.video_processor.preprocess(image, height=height, width=width, resize_mode=resize_mode)
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

        if update_identity_bank and identity_id is not None:
            try:
                # Cond image latent sits in the first conditioned temporal slot.
                self.register_identity_from_latents(
                    identity_id=identity_id,
                    latents=latents[:, :, :1],
                    momentum=identity_update_momentum,
                )
                if _identity_token_count > 0:
                    prompt_embeds, negative_prompt_embeds = self._refresh_identity_tokens(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        identity_id=identity_id,
                        identity_strength=identity_strength,
                        identity_negative_strength=identity_negative_strength,
                        batch_size=batch_size,
                        num_videos_per_prompt=num_videos_per_prompt,
                    )
            except Exception as exc:
                loguru.logger.warning(
                    "Identity bank update (AI2V) failed; continuing without update. Error: {}",
                    exc,
                )

        # 6. Prepare ref_target_masks to latent size
        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)

        # 7. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        cache_num_cond_latents = 1
        cond_latents = None
        kv_cache_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        active_num_cond_latents = cache_num_cond_latents

        if use_kv_cache:
            if reuse_kv_cache and self.kv_cache_dict:
                kv_cache_dict = self._get_kv_cache_dict() or {}
                cond_latents = latents[:, :, :cache_num_cond_latents]
                latents = latents[:, :, cache_num_cond_latents:]
                active_num_cond_latents = int(
                    getattr(self, "_cross_chunk_kv_active_cond", None) or cache_num_cond_latents
                )
                rsm_kv = getattr(self, "runtime_sampling_metrics", None)
                if rsm_kv is not None:
                    rsm_kv.kv_cache_hits += 1
            else:
                cond_latents = latents[:, :, :cache_num_cond_latents]
                active_num_cond_latents = self._cache_clean_latents(
                    cond_latents,
                    max_sequence_length,
                    offload_kv_cache=offload_kv_cache,
                    device=device,
                    dtype=dit_dtype,
                    audio_embs=audio_base_embs,
                    num_cond_latents=cache_num_cond_latents,
                    num_ref_latents=0,
                    ref_img_index=None,
                )
                kv_cache_dict = self._get_kv_cache_dict() or {}
                latents = latents[:, :, cache_num_cond_latents:]

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    raise GenerationInterrupted()

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if not use_kv_cache:
                    timestep[:, :1] = 0
                else:
                    timestep[:, :active_num_cond_latents] = 0

                _kv = kv_cache_dict if use_kv_cache else None
                noise_pred_cond = self._predict_avatar_noise(
                    hidden_states=latents,
                    timestep=timestep[: latents.shape[0]],
                    encoder_hidden_states=prompt_embeds[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_attention_mask,
                    num_cond_latents=active_num_cond_latents,
                    kv_cache_dict=_kv,
                    audio_embs=audio_cond_embs[latents.shape[0] :] if self.do_classifier_free_guidance else audio_cond_embs,
                    ref_target_masks=ref_target_masks,
                )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    if not use_kv_cache:
                        timestep_uncond[:, :1] = 0
                    else:
                        timestep_uncond[:, :active_num_cond_latents] = 0

                    noise_pred_uncond = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=active_num_cond_latents,
                        kv_cache_dict=_kv,
                        audio_embs=audio_unond_embs,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_text = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                        encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                        num_cond_latents=active_num_cond_latents,
                        kv_cache_dict=_kv,
                        audio_embs=audio_unond_embs,
                        ref_target_masks=ref_target_masks,
                    )

                    if emotion_active and self.emotion_guidance_scale > 0.0 and audio_guidance_embs is not None:
                        noise_pred_audio = self._predict_avatar_noise(
                            hidden_states=latents,
                            timestep=timestep_uncond,
                            encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                            encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                            num_cond_latents=active_num_cond_latents,
                            kv_cache_dict=_kv,
                            audio_embs=audio_guidance_embs,
                            ref_target_masks=ref_target_masks,
                        )
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_audio - noise_pred_text)
                            + self.emotion_guidance_scale * (noise_pred_cond - noise_pred_audio)
                        )
                    elif use_cfg_zero:
                        b = noise_pred_text.shape[0]
                        st_star = self.optimized_scale(
                            noise_pred_text.reshape(b, -1),
                            noise_pred_uncond.reshape(b, -1),
                        ).view(b, 1, 1, 1)
                        noise_pred = (
                            noise_pred_uncond * st_star
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond * st_star)
                            + audio_guidance_scale * (noise_pred_cond - noise_pred_text)
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_cond - noise_pred_text)
                        )
                else:
                    noise_pred = noise_pred_cond

                noise_pred = -noise_pred

                if use_kv_cache:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents[:, :, 1:] = self.scheduler.step(
                        noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False
                    )[0]

                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if use_kv_cache and cond_latents is not None:
            latents = torch.cat([cond_latents, latents], dim=2)

        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self.vae.decode(latents, return_dict=False)[0]
        output_video = self._apply_hybrid_mouth_renderer(
            decoded_video=output_video,
            mouth_zone_masks=mouth_zone_masks,
            resize_mode=resize_mode,
        )
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else:
            return output_video
    

    @torch.no_grad()
    def generate_avc(
        self,
        video: List[Image.Image],
        video_latent: torch.Tensor,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        use_kv_cache=True,
        offload_kv_cache=False,
        enhance_hf=True,
        # avatar related params
        audio_emb: torch.Tensor = None,
        ref_latent: torch.Tensor = None,
        ref_img_index: int = None,
        mask_frame_range: int = None,
        ref_target_masks: torch.Tensor = None,
        resize_mode: Optional[str] = "crop", # "default" / "crop"
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        update_identity_bank: bool = False,
        identity_update_momentum: float = 0.25,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
    ):
        r"""
        Generates video frames from a source video and text prompt using diffusion process with spatio-temporal conditioning.

        Args:
            video (`List[Image.Image]`):
                Input video frames for conditioning.
            prompt (`str or List[str]`, *optional*):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_cond_frames (`int`, *optional*, defaults to 13):
                Number of conditioning frames from the input video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            use_kv_cache (`bool`, *optional*, defaults to True):
                Whether to use key-value cache for faster inference.
            offload_kv_cache (`bool`, *optional*, defaults to False):
                Whether to offload key-value cache to CPU to save VRAM.
            enhance_hf (`bool`, *optional*, defaults to True):
                Whether to use enhanced high-frequency denoising schedule.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            ref_latent (`torch.Tensor`):
                The latent of reference anchor image when generate long video.
            ref_img_index (`int`, *optional*, defaults to 10)
                The insertion position of the reference image relative to the noisy latent along the temporal dimension.
            mask_frame_range (`int`, *optional*, defaults to 0)
                The attention masking range for the reference image.
            ref_target_masks(`torch.Tensor`, *optional*, defaults to None):
                Mask used in dual-speaker audio-driven mode.
            resize_mode(`str`, *optional*):
                Output format type. "default" for resize, "crop" for shorter-length resize and centercrop.
            identity_id (`int` or `List[int]`, *optional*):
                Identity slot index (or per-sample indices) in the learnable identity token bank.
            identity_strength (`float`, *optional*, defaults to 1.0):
                Scale applied to identity tokens for conditioned branch.
            identity_negative_strength (`float`, *optional*, defaults to 0.0):
                Scale applied to identity tokens for unconditioned branch.
            update_identity_bank (`bool`, *optional*, defaults to False):
                Update the selected identity slot(s) from current conditioning latents.
            identity_update_momentum (`float`, *optional*, defaults to 0.25):
                EMA update ratio for identity bank writes.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        assert not (use_distill and enhance_hf), "use_distill and enhance_hf cannot both be True"
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )
        assert resize_mode in ['default', 'crop'], f"Unsupported resize_mode {resize_mode}, and you can choose from [default, crop]"
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if emotion_guidance_scale > 0 and (emotion_id is None or emotion_intensity <= 0):
            loguru.logger.warning(
                "Emotion guidance is enabled but emotion control is missing; disabling emotion guidance for this call."
            )
            emotion_guidance_scale = 0.0

        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._emotion_guidance_scale = float(emotion_guidance_scale)
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 3. Encode inputs
        dit_dtype = self.dit.dtype
        identity_token_count = (
            self.identity_tokens_per_id
            if self.identity_bank_enabled and identity_id is not None
            else 0
        )

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
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._append_identity_tokens(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                identity_id=identity_id,
                identity_strength=identity_strength,
                identity_negative_strength=identity_negative_strength,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_seq_len = max_sequence_length + identity_token_count
            effective_batch_size = batch_size * num_videos_per_prompt
            prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_base_embs = self._prepare_audio_emb_for_dit(
            audio_emb,
            num_frames=num_frames,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_cond_embs, emotion_active = self._apply_emotion_channel(
            audio_emb=audio_base_embs,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_cache_embs = audio_base_embs
        audio_guidance_embs = None
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_base_embs)
            if emotion_active and self.emotion_guidance_scale > 0.0:
                audio_guidance_embs = audio_base_embs
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)

        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        if enhance_hf:
            tail_uniform_start = 500
            tail_uniform_end = 0
            num_tail_uniform_steps = 10
            timesteps_uniform_tail = list(np.linspace(tail_uniform_start, tail_uniform_end, num_tail_uniform_steps, dtype=np.float32, endpoint=(tail_uniform_end != 0)))
            timesteps_uniform_tail = [torch.tensor(t, device=device).unsqueeze(0) for t in timesteps_uniform_tail]
            filtered_timesteps = [timestep.unsqueeze(0) for timestep in timesteps if timestep > tail_uniform_start]
            timesteps = torch.cat(filtered_timesteps + timesteps_uniform_tail)
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=timesteps.device)])

        # 5. Prepare latent variables
        video = self.video_processor.preprocess_video(video, height=height, width=width, resize_mode=resize_mode)
        video = video.to(device=device, dtype=prompt_embeds.dtype) 
        cond_videos = video[:, :, -num_cond_frames:]
        cond_videos_latents = retrieve_latents(self.vae.encode(cond_videos), generator, sample_mode="argmax")
        cond_videos_latents = self.normalize_latents(cond_videos_latents)
        if update_identity_bank and identity_id is not None:
            try:
                self.register_identity_from_latents(
                    identity_id=identity_id,
                    latents=cond_videos_latents,
                    momentum=identity_update_momentum,
                )
                if identity_token_count > 0:
                    prompt_embeds, negative_prompt_embeds = self._refresh_identity_tokens(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        identity_id=identity_id,
                        identity_strength=identity_strength,
                        identity_negative_strength=identity_negative_strength,
                        batch_size=batch_size,
                        num_videos_per_prompt=num_videos_per_prompt,
                    )
            except Exception as exc:
                loguru.logger.warning(
                    "Identity bank update (AVC) failed; continuing without update. Error: {}",
                    exc,
                )


        num_channels_latents = self.dit.config.in_channels
        latents = self.prepare_latents(
            video=video_latent,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            dtype=dit_dtype,
            device=device,
            generator=generator,
            latents=latents,
            need_encode=False
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        output_num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
        cache_num_cond_latents = output_num_cond_latents
        
        # 6. Prepare ref_target_masks from source size to latent size
        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)

        # 7. Add reference image
        num_ref_latents = 0
        if ref_latent is not None:
            cache_num_cond_latents += 1
            num_ref_latents = 1
            latents = torch.cat([ref_latent, latents], dim=2)

        # 8. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        if use_kv_cache:
            cond_latents = latents[:, :, :cache_num_cond_latents]
            kv_cache_num_cond_latents = self._cache_clean_latents(cond_latents, max_sequence_length, offload_kv_cache=offload_kv_cache, device=self.device, dtype=dit_dtype, \
                audio_embs=audio_cache_embs, num_cond_latents=cache_num_cond_latents, num_ref_latents=num_ref_latents, ref_img_index=ref_img_index)
            kv_cache_dict = self._get_kv_cache_dict()
            latents = latents[:, :, cache_num_cond_latents:]
            active_num_cond_latents = kv_cache_num_cond_latents
        else:
            kv_cache_dict = {}
            active_num_cond_latents = cache_num_cond_latents

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    raise GenerationInterrupted()

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if not use_kv_cache:
                    timestep[:, :active_num_cond_latents] = 0
                
                noise_pred_cond = self._predict_avatar_noise(
                    hidden_states=latents,
                    timestep=timestep[: latents.shape[0]],
                    encoder_hidden_states=prompt_embeds[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_attention_mask,
                    num_cond_latents=active_num_cond_latents,
                    kv_cache_dict=kv_cache_dict,
                    audio_embs=audio_cond_embs[latents.shape[0] :] if self.do_classifier_free_guidance else audio_cond_embs,
                    num_ref_latents=num_ref_latents,
                    ref_img_index=ref_img_index,
                    mask_frame_range=mask_frame_range,
                    ref_target_masks=ref_target_masks,
                )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    if not use_kv_cache:
                        timestep_uncond[:, :active_num_cond_latents] = 0

                    noise_pred_uncond = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=active_num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_unond_embs,
                        num_ref_latents=num_ref_latents, 
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_text = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                        encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                        num_cond_latents=active_num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_unond_embs,
                        num_ref_latents=num_ref_latents,
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )

                    if emotion_active and self.emotion_guidance_scale > 0.0 and audio_guidance_embs is not None:
                        noise_pred_audio = self._predict_avatar_noise(
                            hidden_states=latents,
                            timestep=timestep_uncond,
                            encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                            encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                            num_cond_latents=active_num_cond_latents,
                            kv_cache_dict=kv_cache_dict,
                            audio_embs=audio_guidance_embs,
                            num_ref_latents=num_ref_latents,
                            ref_img_index=ref_img_index,
                            mask_frame_range=mask_frame_range,
                            ref_target_masks=ref_target_masks,
                        )
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_audio - noise_pred_text)
                            + self.emotion_guidance_scale * (noise_pred_cond - noise_pred_audio)
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_cond - noise_pred_text)
                        )
                else:
                    noise_pred = noise_pred_cond
                
                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                if use_kv_cache:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents[:, :, active_num_cond_latents:] = self.scheduler.step(noise_pred[:, :, active_num_cond_latents:], t, latents[:, :, active_num_cond_latents:], return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
            
            if use_kv_cache:
                latents = torch.cat([cond_latents, latents], dim=2)
            
            if ref_latent is not None:
                latents = latents[:, :, num_ref_latents:]

            latents[:, :, :output_num_cond_latents] = cond_videos_latents

        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self.vae.decode(latents, return_dict=False)[0]
        output_video = self._apply_hybrid_mouth_renderer(
            decoded_video=output_video,
            mouth_zone_masks=mouth_zone_masks,
            resize_mode=resize_mode,
        )
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else: 
            return output_video
    
    @torch.no_grad()
    def generate_chunked_ai2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 12,
        use_distill: bool = True,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 512,
        audio_emb: torch.Tensor = None,
        resize_mode: Optional[str] = "crop",
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
        use_cfg_zero: bool = False,
        chunk_frames: int = 33,
        first_chunk_frames: Optional[int] = None,
        chunk_overlap: int = 8,
        yield_frames: bool = False,
        use_kv_cross_chunk: Optional[bool] = None,
        kv_keep_last: int = 24,
        incremental_audio: Optional[Any] = None,
    ):
        """
        Chunked ai2v: multiple ``generate_ai2v`` passes with pixel-space stitch (Sampling OS wedge).

        When ``yield_frames=True``, yields uint8 frames as each chunk completes (TTFF path).
        Otherwise returns stacked video ``[T,H,W,C]`` numpy.
        """
        import time

        from arachne_x.runtime.chunk_stitch import (
            iter_chunk_frame_ranges,
            slice_audio_emb_temporal,
            stitch_chunk_videos,
        )
        from arachne_x.inference_frames import normalize_ai2v_video_output, round_to_4n_plus_1

        if incremental_audio is None and audio_emb is None:
            raise ValueError("generate_chunked_ai2v requires pre-built `audio_emb` for full clip.")

        total_frames = int(num_frames)
        if total_frames % self.vae_scale_factor_temporal != 1:
            total_frames = total_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        total_frames = max(total_frames, 1)

        full_prepared = None
        if incremental_audio is None:
            if not torch.is_tensor(audio_emb):
                audio_emb = torch.as_tensor(audio_emb)
            if audio_emb.dim() == 3:
                full_prepared = self._prepare_audio_emb_for_dit(
                    audio_emb,
                    num_frames=total_frames,
                    batch_size=1,
                    num_videos_per_prompt=1,
                    device=self.device,
                )
            elif audio_emb.dim() >= 5:
                full_prepared = audio_emb
                if full_prepared.shape[1] < total_frames:
                    total_frames = int(full_prepared.shape[1])
            else:
                raise ValueError(f"Unsupported audio_emb shape for chunked ai2v: {tuple(audio_emb.shape)}")

        rsm = getattr(self, "runtime_sampling_metrics", None)
        if rsm is not None:
            rsm.streaming_mode = "chunked_ai2v"
            rsm.frames_total = total_frames
            rsm.cfg_passes_per_step = 4 if emotion_id and float(emotion_guidance_scale) > 0 else 3

        from arachne_x.runtime.chunk_kv import chunk_kv_enabled, seed_kv_from_chunk_tail
        from arachne_x.runtime.identity_drift_monitor import IdentityDriftMonitor

        chunk_videos: list = []
        chunk_idx = 0
        emitted_until = 0
        use_distill_flag = bool(use_distill or num_inference_steps <= 16)
        if use_kv_cross_chunk is None:
            use_kv_cross_chunk = chunk_kv_enabled()
        else:
            use_kv_cross_chunk = bool(use_kv_cross_chunk)

        drift_mon = IdentityDriftMonitor()
        next_refresh_identity = True
        next_audio_scale = float(audio_guidance_scale)

        def _iter_realtime_chunk_ranges():
            if first_chunk_frames is None or int(first_chunk_frames) <= 0:
                yield from iter_chunk_frame_ranges(total_frames, chunk_frames, chunk_overlap)
                return

            first = round_to_4n_plus_1(min(int(first_chunk_frames), total_frames))
            first = max(1, first)
            first = min(first, total_frames)
            yield 0, first, first

            if first >= total_frames:
                return

            chunk = round_to_4n_plus_1(chunk_frames)
            ov = max(0, min(int(chunk_overlap), chunk - 1))
            # Keep a small bridge into chunk 2 without making a 9-frame first
            # chunk duplicate almost entirely.
            first_overlap = min(ov, max(0, first - 1), 4)
            step = max(1, chunk - ov)
            start = max(0, first - first_overlap)
            while start < total_frames:
                end = min(start + chunk, total_frames)
                n = end - start
                if n <= 0:
                    break
                yield start, end, n
                if end >= total_frames:
                    break
                start += step

        if context_parallel_util.get_cp_rank() == 0:
            loguru.logger.info(
                "Chunked AI2V start: frames={} chunk_frames={} first_chunk_frames={} overlap={} steps={} distill={} kv_cross_chunk={} "
                "identity_id={} incremental_wav2vec={}",
                total_frames,
                int(chunk_frames),
                first_chunk_frames,
                int(chunk_overlap),
                int(num_inference_steps),
                use_distill_flag,
                use_kv_cross_chunk,
                identity_id,
                incremental_audio is not None,
            )

        for start, end, n_chunk in _iter_realtime_chunk_ranges():
            if chunk_idx == 0:
                self.kv_cache_dict = None
            n_gen = round_to_4n_plus_1(n_chunk)
            if incremental_audio is not None:
                audio_slice = incremental_audio.chunk_slice(chunk_idx, start, end, n_gen)
            else:
                audio_slice = slice_audio_emb_temporal(full_prepared, start, min(end, full_prepared.shape[1]))
            if audio_slice.shape[1] < n_gen:
                n_gen = int(audio_slice.shape[1])
                n_gen = round_to_4n_plus_1(max(1, n_gen))
            elif audio_slice.shape[1] > n_gen:
                # Tail chunk: frame budget is 4n+1 but slice may span more embedding steps.
                audio_slice = audio_slice[:, :n_gen].contiguous()

            t0 = time.perf_counter()
            reuse_kv = bool(use_kv_cross_chunk and chunk_idx > 0 and getattr(self, "kv_cache_dict", None))

            out = normalize_ai2v_video_output(
                self.generate_ai2v(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt or "",
                    resolution=resolution,
                    num_frames=n_gen,
                    num_inference_steps=num_inference_steps,
                    use_distill=use_distill_flag,
                    text_guidance_scale=text_guidance_scale,
                    audio_guidance_scale=next_audio_scale,
                    generator=generator,
                    max_sequence_length=max_sequence_length,
                    audio_emb=audio_slice,
                    resize_mode=resize_mode,
                    identity_id=identity_id,
                    identity_strength=identity_strength,
                    identity_negative_strength=identity_negative_strength,
                    emotion_id=emotion_id,
                    emotion_intensity=emotion_intensity,
                    emotion_guidance_scale=emotion_guidance_scale,
                    mouth_zone_masks=mouth_zone_masks,
                    use_cfg_zero=use_cfg_zero,
                    use_kv_cache=bool(use_kv_cross_chunk and reuse_kv),
                    reuse_kv_cache=reuse_kv,
                    refresh_identity_tokens=next_refresh_identity,
                    silence_gate=True,
                    update_identity_bank=False,
                )
            )

            if chunk_idx == 0:
                drift_mon.set_anchor_from_frame(out[0])
            cos = drift_mon.score_chunk_tail(out)
            pol = drift_mon.policy_for_next_chunk(cos)
            next_refresh_identity = bool(pol.get("refresh_identity_tokens", True))
            next_audio_scale = float(audio_guidance_scale) * float(
                pol.get("audio_guidance_scale_multiplier", 1.0)
            )

            if rsm is not None:
                chunk_elapsed = time.perf_counter() - t0
                rsm.add_denoise_elapsed(chunk_elapsed)
                if chunk_idx == 0:
                    rsm.mark_first_chunk_done(chunk_elapsed)
                rsm.chunk_count += 1
                rsm.frames_per_chunk.append(int(out.shape[0]))
                drift_dict = drift_mon.to_dict()
                rsm.identity_cosine_per_chunk = drift_dict.get("identity_cosine_per_chunk", [])
                rsm.identity_drift_min = drift_dict.get("identity_drift_min")
                rsm.corrective_actions = drift_dict.get("corrective_actions", [])
            else:
                chunk_elapsed = time.perf_counter() - t0

            if context_parallel_util.get_cp_rank() == 0:
                loguru.logger.info(
                    "Chunked AI2V chunk_done idx={} range={}..{} frames={} elapsed_sec={:.4f} "
                    "reuse_kv={} drift_cosine={:.4f} next_audio_scale={:.4f}",
                    chunk_idx,
                    start,
                    end,
                    int(out.shape[0]),
                    chunk_elapsed,
                    reuse_kv,
                    float(cos),
                    next_audio_scale,
                )

            chunk_videos.append(out)
            if use_kv_cross_chunk:
                try:
                    seeded = seed_kv_from_chunk_tail(
                        self,
                        out,
                        audio_emb_slice=audio_slice,
                        kv_keep_last=kv_keep_last,
                        max_sequence_length=max_sequence_length,
                        chunk_idx=chunk_idx,
                    )
                    if not seeded:
                        loguru.logger.warning(
                            "chunk KV seed failed (continuing): chunk_idx={} reason=empty_kv_cache",
                            chunk_idx,
                        )
                except Exception as exc:
                    loguru.logger.warning(
                        "chunk KV seed failed (continuing): chunk_idx={} error={}",
                        chunk_idx,
                        exc,
                    )
            if yield_frames:
                skip_prefix = max(0, min(int(out.shape[0]), int(emitted_until) - int(start)))
                for fi in range(skip_prefix, out.shape[0]):
                    if rsm is not None:
                        rsm.mark_first_frame_emit()
                        if fi == skip_prefix and chunk_idx == 0 and context_parallel_util.get_cp_rank() == 0:
                            loguru.logger.info("Chunked AI2V first_frame_emit metrics={}", rsm.to_dict())
                    yield out[fi]
                emitted_until = max(int(emitted_until), int(end))

            chunk_idx += 1

        if yield_frames:
            return

        stitched = stitch_chunk_videos(chunk_videos, chunk_overlap)
        return stitched

    @torch.no_grad()
    def generate_streaming_ai2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        audio_stream=None,  # Generator yielding audio chunks
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 8,  # Distilled: 8 steps instead of 50
        use_distill: Optional[bool] = None,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        audio_emb: torch.Tensor = None,
        resize_mode: Optional[str] = "crop",
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
        use_cfg_zero: bool = False,
        chunk_frames: int = 33,
        first_chunk_frames: Optional[int] = None,
        chunk_overlap: int = 8,
        use_chunked_denoise: Optional[bool] = None,
    ):
        r"""
        Streaming-like video generation (Image-to-Video).
        Operational path: chunked denoise + per-chunk emit (TTFF). Legacy: monolithic denoise + stream VAE.
        
        Args:
            image: Input image for video generation.
            prompt: Text prompt(s) for video content generation.
            audio_stream: Optional generator yielding audio chunks [sample_rate=16000].
            resolution: "480p" or "720p".
            num_frames: Number of frames to generate.
            num_inference_steps: Denoising steps (8 = distilled fast mode).
            text_guidance_scale: CFG scale for text.
            audio_guidance_scale: CFG scale for audio.
            generator: Random seed generator.
            audio_emb: Pre-computed audio embedding (alternative to audio_stream).
            resize_mode: "default" or "crop".
            identity_id: Identity slot index (or per-sample indices) in identity token bank.
            identity_strength: Scale applied to identity tokens for conditioned branch.
            identity_negative_strength: Scale applied to identity tokens for unconditioned branch.
            emotion_id: Emotion class id or label.
            emotion_intensity: Emotion intensity multiplier.
            emotion_guidance_scale: Separate CFG scale for emotion channel.
        
        Yields:
            np.ndarray: Frame as numpy array [H, W, 3] in range [0, 255].
        """
        
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        
        height, width = self.get_condition_shape(image, resolution, scale_factor_spatial=scale_factor_spatial)
        self.check_inputs(prompt, None, height, width, scale_factor_spatial)
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
        
        device = self.device
        
        incremental_audio = None
        # 1. Resolve audio embedding.
        if audio_emb is None:
            if audio_stream is None:
                raise ValueError("Either `audio_stream` or `audio_emb` must be provided.")

            from arachne_x.inference_audio import (
                IncrementalStreamingAudioEmb,
                drain_audio_stream,
                incremental_wav2vec_enabled,
            )

            if incremental_wav2vec_enabled():
                full_audio = drain_audio_stream(audio_stream)
                incremental_audio = IncrementalStreamingAudioEmb(
                    self,
                    full_audio,
                    num_frames=num_frames,
                    first_chunk_frames=first_chunk_frames,
                    device=device,
                )
                if context_parallel_util.get_cp_rank() == 0:
                    loguru.logger.info(
                        "Streaming AI2V incremental wav2vec enabled metrics={}",
                        incremental_audio.metrics_snapshot(),
                    )
            else:
                audio_chunks = []
                sample_rate = 16000
                for chunk in audio_stream:
                    if chunk is None:
                        continue
                    audio_chunks.append(np.asarray(chunk, dtype=np.float32))

                if not audio_chunks:
                    raise ValueError("`audio_stream` yielded no chunks.")

                full_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32, copy=False)
                audio_stride = max(int(self.vae_scale_factor_temporal), 1)
                emb_fps = getattr(self, "inference_embedding_fps", None)
                if emb_fps is None:
                    emb_fps = 16 * audio_stride
                else:
                    emb_fps = float(emb_fps)
                full_audio_emb = self.get_audio_embedding(
                    full_audio,
                    fps=emb_fps,
                    device=device,
                    sample_rate=sample_rate,
                )
                audio_emb = self._build_windowed_audio_embedding(
                    full_audio_emb,
                    num_frames=num_frames,
                    device=device,
                )
        else:
            audio_emb = self._prepare_audio_emb_for_dit(
                audio_emb,
                num_frames=num_frames,
                batch_size=1,
                num_videos_per_prompt=1,
                device=device,
            )

        import os

        legacy_streaming = os.environ.get("ARACHNE_LEGACY_STREAMING", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        use_chunked = not legacy_streaming
        if use_chunked_denoise is not None:
            use_chunked = bool(use_chunked_denoise)
        if use_distill is not None:
            use_distill_flag = bool(use_distill)
        else:
            use_distill_flag = bool(num_inference_steps <= 16)

        if use_chunked and int(num_frames) > int(chunk_frames):
            rsm_stream = getattr(self, "runtime_sampling_metrics", None)
            if rsm_stream is not None:
                rsm_stream.streaming_mode = "chunked_ai2v"
            if context_parallel_util.get_cp_rank() == 0:
                loguru.logger.info(
                    "Streaming AI2V selected chunked path frames={} chunk_frames={} first_chunk_frames={} overlap={} steps={} distill={}",
                    int(num_frames),
                    int(chunk_frames),
                    first_chunk_frames,
                    int(chunk_overlap),
                    int(num_inference_steps),
                    use_distill_flag,
                )
            for frame_np in self.generate_chunked_ai2v(
                image=image,
                prompt=prompt,
                negative_prompt="",
                resolution=resolution,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                use_distill=use_distill_flag,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                generator=generator,
                max_sequence_length=max_sequence_length,
                audio_emb=audio_emb,
                resize_mode=resize_mode,
                identity_id=identity_id,
                identity_strength=identity_strength,
                identity_negative_strength=identity_negative_strength,
                emotion_id=emotion_id,
                emotion_intensity=emotion_intensity,
                emotion_guidance_scale=emotion_guidance_scale,
                mouth_zone_masks=mouth_zone_masks,
                use_cfg_zero=use_cfg_zero,
                chunk_frames=chunk_frames,
                first_chunk_frames=first_chunk_frames,
                chunk_overlap=chunk_overlap,
                yield_frames=True,
                incremental_audio=incremental_audio,
            ):
                yield frame_np
            return

        # Legacy: monolithic denoise then stream VAE decode
        if incremental_audio is not None:
            incremental_audio._build_full()
            audio_emb = incremental_audio._full_prepared
        rsm_stream = getattr(self, "runtime_sampling_metrics", None)
        if rsm_stream is not None:
            rsm_stream.streaming_mode = "legacy_monolithic"
        if context_parallel_util.get_cp_rank() == 0:
            loguru.logger.warning(
                "Streaming AI2V selected legacy monolithic path frames={} chunk_frames={} chunked={} legacy_env={} "
                "steps={} distill={}; TTFF waits for full denoise before first frame.",
                int(num_frames),
                int(chunk_frames),
                use_chunked,
                legacy_streaming,
                int(num_inference_steps),
                use_distill_flag,
            )
        latents = self.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt="",
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_distill=use_distill_flag,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type="latent",
            max_sequence_length=max_sequence_length,
            audio_emb=audio_emb,
            resize_mode=resize_mode,
            identity_id=identity_id,
            identity_strength=identity_strength,
            identity_negative_strength=identity_negative_strength,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            emotion_guidance_scale=emotion_guidance_scale,
            mouth_zone_masks=mouth_zone_masks,
            use_cfg_zero=use_cfg_zero,
        )

        # Stream decode: denormalize, decode frame-by-frame, yield
        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        vae_decoder = StreamingVAEDecoder(self.vae, chunk_size=1, enable_amp=True)
        frame_times = []
        stream_mouth_mask = None
        stream_boundary_mask = None
        prev_stabilized = None
        prev_stabilized_for_flicker = None
        hybrid_artifacts = []
        hybrid_boundary_diffs = []
        hybrid_global_diffs = []
        for frame_idx, decoded in enumerate(vae_decoder.decode_streaming(latents)):
            frame_time = time.time()
            frame_tensor = decoded
            if mouth_zone_masks is not None and self.hybrid_renderer_enabled:
                if stream_mouth_mask is None:
                    prepared = self._prepare_mouth_zone_mask(
                        mouth_zone_masks=mouth_zone_masks,
                        batch_size=decoded.shape[0],
                        num_frames=1,
                        height=decoded.shape[-2],
                        width=decoded.shape[-1],
                        device=decoded.device,
                        dtype=decoded.dtype,
                        resize_mode=resize_mode,
                    )
                    if prepared is not None:
                        stream_mouth_mask = prepared[:, :, 0]
                        stream_boundary_mask = self._compute_seam_boundary_mask(prepared)[:, :, 0]

                if stream_mouth_mask is not None and stream_boundary_mask is not None:
                    branch = self._build_mouth_controlled_branch(
                        decoded.unsqueeze(2),
                        strength=float(self.hybrid_renderer_mouth_strength),
                    )[:, :, 0]
                    blended = decoded * (1.0 - stream_mouth_mask) + branch * stream_mouth_mask
                    if prev_stabilized is not None:
                        a = float(self.hybrid_renderer_temporal_alpha)
                        blended = (
                            blended * (1.0 - stream_boundary_mask)
                            + (a * blended + (1.0 - a) * prev_stabilized) * stream_boundary_mask
                        )
                    prev_stabilized = blended.detach()
                    frame_tensor = blended

                    hybrid_artifacts.append(float((torch.abs(frame_tensor - decoded) * stream_mouth_mask).mean().item()))
                    if prev_stabilized_for_flicker is not None:
                        diff = torch.abs(frame_tensor - prev_stabilized_for_flicker)
                        hybrid_boundary_diffs.append(float((diff * stream_boundary_mask).mean().item()))
                        hybrid_global_diffs.append(float(diff.mean().item()))
                    prev_stabilized_for_flicker = frame_tensor.detach()

            frame_np = (frame_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame_times.append(time.time() - frame_time)
            rsm_stream = getattr(self, "runtime_sampling_metrics", None)
            if rsm_stream is not None:
                rsm_stream.mark_first_frame_emit()
            yield frame_np
        
        # 4. Log performance
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            sorted_times = sorted(frame_times)
            p95_latency = sorted_times[int(len(sorted_times) * 0.95)] * 1000 if sorted_times else 0
            self.metrics.record('streaming_fps', fps)
            self.metrics.record('streaming_p95_latency_ms', p95_latency)
            if hybrid_artifacts:
                artifact_mean = float(sum(hybrid_artifacts) / len(hybrid_artifacts))
                self.metrics.record("hybrid_stream_artifact_energy", artifact_mean)
            if hybrid_boundary_diffs and hybrid_global_diffs:
                boundary_mean = float(sum(hybrid_boundary_diffs) / len(hybrid_boundary_diffs))
                global_mean = float(sum(hybrid_global_diffs) / len(hybrid_global_diffs))
                ratio = boundary_mean / max(global_mean, 1e-6)
                self.metrics.record("hybrid_stream_flicker_ratio", ratio)
                self.metrics.record("hybrid_stream_budget_ok", int(
                    ratio <= float(self.hybrid_renderer_flicker_budget)
                ))
            if context_parallel_util.get_cp_rank() == 0:
                loguru.logger.info(f"Streaming complete: {fps:.1f} FPS, P95 latency: {p95_latency:.1f}ms")

    

    def to(self, device: str | torch.device):
        """
        Move pipeline to specified device.

        Args:
            device: Target device string

        Returns:
            Self
        """
        self.device = device
        if self.dit is not None:
            self.dit = self.dit.to(device, non_blocking=True)
            if hasattr(self.dit, 'lora_dict') and self.dit.lora_dict:
                for lora_key, lora_network in self.dit.lora_dict.items():
                    for lora in lora_network.loras:
                        lora.to(device, non_blocking=True)
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device, non_blocking=True)
        if self.vae is not None:
            self.vae = self.vae.to(device, non_blocking=True)
        if self.identity_embedding is not None:
            self.identity_embedding = self.identity_embedding.to(device, non_blocking=True)
        if self.identity_latent_projector is not None:
            self.identity_latent_projector = self.identity_latent_projector.to(device, non_blocking=True)
        if self.emotion_embedding is not None:
            self.emotion_embedding = self.emotion_embedding.to(device, non_blocking=True)
        if self.emotion_proj is not None:
            self.emotion_proj = self.emotion_proj.to(device, non_blocking=True)
        return self
    

LongCatVideoAvatarPipeline = ArachneXVideoAvatarPipeline
