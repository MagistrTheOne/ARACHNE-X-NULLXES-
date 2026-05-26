"""Text, prompt, and optional planning-token helpers for avatar generation."""
from __future__ import annotations

import html
import os
from typing import List, Optional, Tuple, Union

import ftfy
import loguru
import regex as re
import torch
import torch.nn as nn

from arachne_x.context_parallel import context_parallel_util

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class TextConditioningMixin:
    """UMT5 prompt encoding and avatar token assembly methods."""

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        mask = mask.to(device=device)
        if num_videos_per_prompt > 1:
            mask = mask.repeat_interleave(num_videos_per_prompt, dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)

        return prompt_embeds, mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
            
        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def _get_planning_token_head(self) -> nn.Module:
        if self._planning_token_head is None:
            from arachne_x.planning.planning_token_head import PlanningTokenHead

            self._planning_token_head = PlanningTokenHead(
                d_model=self.identity_token_dim,
                n_tokens=self.planning_tokens_count,
            ).to(device=self.device, dtype=self.dit.dtype)
        return self._planning_token_head

    def try_load_planning_head(self, checkpoint_dir: str) -> bool:
        """Load ``planning/planning_head.safetensors`` when present; enables planning."""
        path = os.path.join(checkpoint_dir, "planning", "planning_head.safetensors")
        if not os.path.isfile(path):
            return False
        try:
            from safetensors.torch import load_file

            head = self._get_planning_token_head()
            state = load_file(path, device=str(self.device))
            head.load_state_dict(state, strict=False)
            self.planning_enabled = True
            loguru.logger.info("Planning token head loaded from {}", path)
            return True
        except Exception as exc:
            loguru.logger.warning("Failed to load planning head from {}: {}", path, exc)
            return False

    def _append_planning_tokens(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_attention_mask: Optional[torch.Tensor],
        *,
        audio_duration_sec: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.planning_enabled or self.planning_tokens_count <= 0:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            )

        pooled = prompt_embeds.mean(dim=2).squeeze(1)
        bsz = pooled.shape[0]
        dur = float(audio_duration_sec or 0.0)
        audio_stats = torch.tensor(
            [[dur, dur / max(bsz, 1), 0.0, 0.0]] * bsz,
            device=pooled.device,
            dtype=pooled.dtype,
        )
        head = self._get_planning_token_head()
        plan_tokens = head(pooled, audio_stats).unsqueeze(1)
        plan_mask = torch.ones(
            (plan_tokens.shape[0], plan_tokens.shape[2]),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        prompt_embeds = torch.cat([plan_tokens, prompt_embeds], dim=2)
        prompt_attention_mask = torch.cat([plan_mask, prompt_attention_mask], dim=1)

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is not None:
            neg_tokens = torch.zeros_like(plan_tokens)
            neg_mask = torch.zeros_like(plan_mask)
            negative_prompt_embeds = torch.cat([neg_tokens, negative_prompt_embeds], dim=2)
            negative_prompt_attention_mask = torch.cat([neg_mask, negative_prompt_attention_mask], dim=1)

        self.metrics.record("planning_tokens_appended", int(self.planning_tokens_count))
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def _encode_prompt_with_avatar_tokens(
        self,
        *,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]],
        batch_size: int,
        num_videos_per_prompt: int,
        max_sequence_length: int,
        dit_dtype: torch.dtype,
        device: torch.device,
        identity_id: Optional[Union[int, List[int], torch.Tensor]],
        identity_strength: float,
        identity_negative_strength: float,
        audio_duration_sec: Optional[float] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        int,
        int,
    ]:
        """UMT5 encode → planning tokens (optional) → identity tokens (optional)."""
        identity_token_count = (
            self.identity_tokens_per_id
            if self.identity_bank_enabled and identity_id is not None
            else 0
        )
        planning_token_count = (
            self.planning_tokens_count if self.planning_enabled else 0
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
            ) = self._append_planning_tokens(
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                audio_duration_sec=audio_duration_sec,
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
            prompt_seq_len = max_sequence_length + planning_token_count + identity_token_count
            effective_batch_size = batch_size * num_videos_per_prompt
            prompt_embeds = torch.zeros(
                [effective_batch_size, 1, prompt_seq_len, caption_channels],
                dtype=dit_dtype,
                device=device,
            )
            prompt_attention_mask = torch.zeros(
                [effective_batch_size, prompt_seq_len],
                dtype=torch.int64,
                device=device,
            )
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros(
                    [effective_batch_size, 1, prompt_seq_len, caption_channels],
                    dtype=dit_dtype,
                    device=device,
                )
                negative_prompt_attention_mask = torch.zeros(
                    [effective_batch_size, prompt_seq_len],
                    dtype=torch.int64,
                    device=device,
                )
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        else:
            raise RuntimeError("Unexpected context-parallel rank layout")

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            planning_token_count,
            identity_token_count,
        )
