from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor

from .modules.autoencoder_kl_wan import AutoencoderKLWan
from .modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .modules.arachne_video_dit import LongCatVideoTransformer3DModel
from .modules.avatar.arachne_avatar_dit import LongCatVideoAvatarTransformer3DModel
from .audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from .pipeline_arachne_x_video import ArachneXVideoPipeline
from .pipeline_arachne_x_video_avatar import ArachneXVideoAvatarPipeline
from .pipeline_audio_i2v import AudioConditionedI2VPipeline
from .modules.audio_conditioning import AudioConditioningAdapter
from .modules.audio_conditioning.audio_encode import AudioEncoderRuntime


@dataclass(frozen=True)
class WeightsLayout:
    tokenizer: str = "tokenizer"
    text_encoder: str = "text_encoder"
    vae: str = "vae"
    scheduler: str = "scheduler"
    dit: str = "dit"
    avatar_single: str = "avatar_single"
    avatar_multi: str = "avatar_multi"
    audio_dir: str = "audio"
    wav2vec2: str = "audio/wav2vec2"
    vocal_separator: str = "audio/vocal_separator/Kim_Vocal_2.onnx"


logger = logging.getLogger(__name__)


def _p(root: str, subpath: str) -> str:
    return str(Path(root) / subpath)


def _has_wav2vec_weights(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    return (path / "pytorch_model.bin").is_file() or (path / "model.safetensors").is_file()


def resolve_wav2vec_checkpoint_path(
    checkpoint_dir: str,
    layout: WeightsLayout = WeightsLayout(),
) -> str:
    """
    Resolve wav2vec weights for VIDEO/AVATAR runtime trees.

    VIDEO checkpoints often omit ``audio/wav2vec2``; AVATAR snapshots store weights under
    ``chinese-wav2vec2-base``. Falls back to ``ARACHNE_AVATAR_CKPT`` / ``AVATAR_CKPT``.
    """
    root = Path(checkpoint_dir)
    candidates: list[Path] = [
        root / layout.wav2vec2,
        root / "chinese-wav2vec2-base",
    ]
    for env_key in ("ARACHNE_AVATAR_CKPT", "AVATAR_CKPT"):
        avatar_ckpt = (os.environ.get(env_key) or "").strip()
        if not avatar_ckpt:
            continue
        avatar_root = Path(avatar_ckpt)
        candidates.extend(
            [
                avatar_root / "audio" / "wav2vec2",
                avatar_root / "chinese-wav2vec2-base",
            ]
        )

    seen: set[str] = set()
    for cand in candidates:
        probe = cand.resolve() if cand.exists() else cand
        key = str(probe)
        if key in seen:
            continue
        seen.add(key)
        if _has_wav2vec_weights(probe):
            resolved = str(probe)
            logger.info(
                "wav2vec_checkpoint resolved checkpoint_dir=%s path=%s",
                checkpoint_dir,
                resolved,
            )
            return resolved

    default = str(root / layout.wav2vec2)
    logger.warning(
        "wav2vec_checkpoint missing checkpoint_dir=%s candidates=%s fallback=%s",
        checkpoint_dir,
        [str(c) for c in candidates],
        default,
    )
    return default


def load_base_pipeline(
    checkpoint_dir: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[Tuple[int, int]] = None,
    layout: WeightsLayout = WeightsLayout(),
) -> ArachneXVideoPipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        _p(checkpoint_dir, layout.tokenizer),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        _p(checkpoint_dir, layout.text_encoder),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        _p(checkpoint_dir, layout.vae),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        _p(checkpoint_dir, layout.scheduler),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        _p(checkpoint_dir, layout.dit),
        cp_split_hw=cp_split_hw,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    pipe = ArachneXVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(device)
    return pipe


def load_avatar_pipeline(
    checkpoint_dir: str,
    variant: str = "single",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[Tuple[int, int]] = None,
    layout: WeightsLayout = WeightsLayout(),
) -> ArachneXVideoAvatarPipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        _p(checkpoint_dir, layout.tokenizer),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        _p(checkpoint_dir, layout.text_encoder),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        _p(checkpoint_dir, layout.vae),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        _p(checkpoint_dir, layout.scheduler),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    if variant not in ("single", "multi"):
        raise ValueError(f"Unknown avatar variant: {variant}")
    avatar_subdir = layout.avatar_single if variant == "single" else layout.avatar_multi
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        _p(checkpoint_dir, avatar_subdir),
        cp_split_hw=cp_split_hw,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    wav2vec_path = resolve_wav2vec_checkpoint_path(checkpoint_dir, layout)
    audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(device)
    audio_encoder.feature_extractor._freeze_parameters()

    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_path,
        local_files_only=True,
    )

    pipe = ArachneXVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=audio_encoder,
        wav2vec_feature_extractor=wav2vec_feature_extractor,
    )
    pipe.to(device)
    from arachne_x.infer_attention import configure_infer_bsa

    configure_infer_bsa(pipe.dit)
    if hasattr(pipe, "try_load_planning_head"):
        pipe.try_load_planning_head(checkpoint_dir)
    return pipe


def load_audio_i2v_pipeline(
    checkpoint_dir: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[Tuple[int, int]] = None,
    layout: WeightsLayout = WeightsLayout(),
    audio_adapter: Optional[AudioConditioningAdapter] = None,
    audio_adapter_path: Optional[str] = None,
) -> AudioConditionedI2VPipeline:
    """
    Experimental audio-conditioned I2V over frozen VIDEO checkpoint.

    Requires wav2vec weights in the runtime tree or ``ARACHNE_AVATAR_CKPT`` fallback.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        _p(checkpoint_dir, layout.tokenizer),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        _p(checkpoint_dir, layout.text_encoder),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        _p(checkpoint_dir, layout.vae),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        _p(checkpoint_dir, layout.scheduler),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        _p(checkpoint_dir, layout.dit),
        cp_split_hw=cp_split_hw,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    for param in dit.parameters():
        param.requires_grad = False

    wav2vec_path = resolve_wav2vec_checkpoint_path(checkpoint_dir, layout)
    audio_runtime = AudioEncoderRuntime.from_checkpoint(wav2vec_path, device=device)

    adapter = audio_adapter
    if adapter is None and audio_adapter_path:
        from .modules.audio_conditioning import load_audio_conditioning_adapter

        adapter = load_audio_conditioning_adapter(audio_adapter_path, device=device)
    if adapter is None:
        adapter = AudioConditioningAdapter()
    adapter.to(device)

    pipe = AudioConditionedI2VPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder_runtime=audio_runtime,
        audio_adapter=adapter,
    )
    pipe.to(device)
    return pipe


def get_vocal_separator_path(checkpoint_dir: str, layout: WeightsLayout = WeightsLayout()) -> str:
    return _p(checkpoint_dir, layout.vocal_separator)
