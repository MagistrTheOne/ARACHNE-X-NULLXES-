"""
Fine-tune avatar DiT with LoRA only (frozen base weights).

Expects the same precomputed tensors as scripts/train.py (LatentDataset):
latents, prompt_embeds, prompt_mask, timesteps, noise, audio_embs.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import sys
import time
from glob import glob
from typing import Iterable
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file, save_file

from arachne_x.modules.avatar.arachne_avatar_dit import LongCatVideoAvatarTransformer3DModel
from arachne_x.training_latent_common import (
    collate_latent_samples,
    normalize_prompt_embeds_batch,
    squeeze_collated_singleton_batch_dim,
    validate_latent_sample,
)
from arachne_x.training_wds import LatentWebDataset
from arachne_x.modules.lora_utils import (
    avatar_attention_only_lora_filter,
    build_initial_lora_state_dict,
    create_lora_network,
    default_avatar_train_lora_filter,
)
from arachne_x.training_lora_loss import (
    avatar_lora_diffusion_loss,
    load_flow_match_scheduler,
    stabilize_audio_embs,
)
from arachne_x.training_avatar_aux import AvatarAuxStageSchedule, AvatarAuxTrainingRuntime
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root
from Demo.training_config_h200 import H200TrainingConfig


def _phase(msg: str) -> None:
    print(f"[train_lora_avatar] {msg}", flush=True)


class LatentDataset(Dataset):
    """Same contract as scripts/train.LatentDataset."""

    def __init__(self, dataset_dir: str):
        self.files = sorted(
            glob(os.path.join(dataset_dir, "*.pt")) + glob(os.path.join(dataset_dir, "*.npz"))
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt or .npz files found in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        if path.endswith(".pt"):
            sample = torch.load(path, map_location="cpu")
        else:
            data = np.load(path)
            sample = {k: torch.from_numpy(data[k]) for k in data.files}

        validate_latent_sample(sample, require_audio=True, source=f"{path}: ")
        return sample


def _to_device(batch: Dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if k in {"prompt_mask", "timesteps"}:
            out[k] = v.to(device)
        else:
            out[k] = v.to(device=device, dtype=dtype)
    return out


def main():
    parser = argparse.ArgumentParser(description="ARACHNE-X avatar DiT LoRA training")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Root with avatar_single/")
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument("--dataset_dir", type=str, default=None, help="Folder with *.pt / *.npz")
    ds.add_argument(
        "--wds_shards",
        type=str,
        default=None,
        help='WebDataset shards, e.g. "/data/shard_{000000..000009}.tar"',
    )
    parser.add_argument("--output_dir", type=str, default="./outputs_train_lora_avatar")
    parser.add_argument("--lora_key", type=str, default="train", help="Key in dit.lora_dict and enable_loras")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank; if omitted: from --config JSON (lora_rank), else 128.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha; if omitted: from --config JSON (lora_alpha), else 64.",
    )
    parser.add_argument(
        "--lora_prefixes",
        type=str,
        default="",
        help="Comma-separated name prefixes for Linear modules (e.g. blocks.,audio_proj.). Empty = default filter.",
    )
    parser.add_argument(
        "--lora_scope",
        type=str,
        choices=("default", "attention"),
        default="attention",
        help="default=blocks+audio_proj+final_layer; attention=attn/cross_attn/audio_cross_attn only (less snow).",
    )
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=5.0,
        help="Min-SNR gamma for flow-match loss (0=plain MSE). Default 5.0 reduces high-noise grain.",
    )
    parser.add_argument(
        "--normalize_audio_embs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Per-token RMS normalize audio_embs before DiT forward.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9995,
        help="EMA decay for LoRA weights (0=off). Default 0.9995; use 0.9997 for 500k+ steps.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument(
        "--resume_lora_path",
        type=str,
        default=None,
        help="Load LoRA weights from a prior checkpoint (e.g. lora_step_15.safetensors).",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help="Global step to continue from (use 16 after saving lora_step_15).",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional H200TrainingConfig JSON")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="AdamW weight decay; if omitted, use H200TrainingConfig.weight_decay.",
    )
    parser.add_argument("--wds_shuffle", type=int, default=5000, help="WebDataset shuffle buffer (0=off).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable DiT activation checkpointing (needs ~100GB+ extra VRAM on 720p latents).",
    )
    parser.add_argument(
        "--enable_aux_losses",
        action="store_true",
        help="Enable staged avatar auxiliary losses (VAE decode + identity/perceptual).",
    )
    parser.add_argument(
        "--reference_image",
        type=str,
        default=None,
        help="Reference face image for identity/perceptual aux losses (required if --enable_aux_losses).",
    )
    parser.add_argument(
        "--aux_training_stage",
        type=int,
        default=1,
        choices=(1, 2, 3, 4),
        help="Initial aux loss stage (auto-advances with --aux_stage*_step).",
    )
    parser.add_argument(
        "--aux_stage2_step",
        type=int,
        default=5000,
        help="Global step to enable identity aux loss (stage 2).",
    )
    parser.add_argument(
        "--aux_stage3_step",
        type=int,
        default=15000,
        help="Global step to enable lip-sync aux loss (stage 3).",
    )
    parser.add_argument(
        "--aux_stage4_step",
        type=int,
        default=30000,
        help="Global step to enable temporal aux loss (stage 4).",
    )
    parser.add_argument(
        "--aux_loss_weight",
        type=float,
        default=0.15,
        help="Scale factor for auxiliary loss total added to diffusion loss.",
    )
    parser.add_argument(
        "--perceptual_backend",
        type=str,
        choices=("vgg", "dino"),
        default="vgg",
        help="Perceptual backend for aux losses (dino = DINOv2, stage 4+ quality).",
    )
    add_resolve_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config_path = args.config
    if config_path is not None:
        cfg = H200TrainingConfig.from_json(config_path)
    else:
        cfg = H200TrainingConfig()

    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.weight_decay

    lora_rank = (
        args.lora_rank
        if args.lora_rank is not None
        else (cfg.lora_rank if config_path is not None else 128)
    )
    lora_alpha = (
        args.lora_alpha
        if args.lora_alpha is not None
        else (float(cfg.lora_alpha) if config_path is not None else 64.0)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    fw_dtype = torch.float32 if device == "cpu" else dtype

    checkpoint_dir = resolve_weights_root(
        args.checkpoint_dir,
        allow_hub=args.allow_hub_download,
        cache_dir=args.weights_cache_dir,
    )

    if args.dataset_dir:
        dataset = LatentDataset(args.dataset_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
    else:
        iterable = LatentWebDataset(
            args.wds_shards,
            require_audio=True,
            shuffle=max(0, args.wds_shuffle),
        )
        loader = DataLoader(
            iterable,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_latent_samples,
            drop_last=True,
        )

    batch_iter: Iterable = itertools.cycle(loader) if args.dataset_dir else loader

    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    _phase(f"loading DiT from {checkpoint_dir}/avatar_single …")
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        checkpoint_dir,
        subfolder="avatar_single",
        torch_dtype=dtype,
    )
    _phase(f"DiT shards loaded ({time.time() - t0:.1f}s), moving to {device} …")
    dit.to(device)
    _phase(f"DiT on {device} ({time.time() - t0:.1f}s total)")
    dit.requires_grad_(False)
    dit.train()

    if hasattr(dit, "disable_bsa"):
        dit.disable_bsa()
        _phase("BSA disabled for LoRA train (dense flash-attn parity)")

    if not args.no_gradient_checkpointing and hasattr(dit, "enable_gradient_checkpointing"):
        dit.enable_gradient_checkpointing()
        _phase("gradient_checkpointing on")
    elif hasattr(dit, "disable_gradient_checkpointing"):
        dit.disable_gradient_checkpointing()

    prefixes = None
    if args.lora_prefixes.strip():
        prefixes = tuple(p.strip() for p in args.lora_prefixes.split(",") if p.strip())

    scope_filter = (
        avatar_attention_only_lora_filter
        if args.lora_scope == "attention"
        else default_avatar_train_lora_filter
    )

    def name_filter(name: str, mod: torch.nn.Linear) -> bool:
        return scope_filter(name, mod, include_prefixes=prefixes)

    train_scheduler = load_flow_match_scheduler(checkpoint_dir)
    _phase(f"flow-match scheduler loaded (min_snr_gamma={args.min_snr_gamma})")

    init_state = build_initial_lora_state_dict(
        dit,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        name_filter=name_filter,
        dtype=torch.float32,
    )

    lora_network = create_lora_network(
        dit,
        init_state,
        multiplier=1.0,
        network_dim=args.lora_rank,
        network_alpha=args.lora_alpha,
    )
    incon = lora_network.load_state_dict(init_state, strict=True)
    assert not incon.missing_keys and not incon.unexpected_keys, (incon.missing_keys, incon.unexpected_keys)
    if args.resume_lora_path:
        resume_path = os.path.abspath(args.resume_lora_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume LoRA not found: {resume_path}")
        resume_state = load_file(resume_path, device="cpu")
        resume_incon = lora_network.load_state_dict(resume_state, strict=True)
        assert not resume_incon.missing_keys and not resume_incon.unexpected_keys, (
            resume_incon.missing_keys,
            resume_incon.unexpected_keys,
        )
        _phase(f"resumed LoRA weights from {resume_path}")

    lora_network.to(device=device, dtype=dtype)
    lora_network.train()
    dit.lora_dict[args.lora_key] = lora_network
    dit.enable_loras([args.lora_key])

    aux_runtime: AvatarAuxTrainingRuntime | None = None
    if args.enable_aux_losses:
        if not args.reference_image or not os.path.isfile(args.reference_image):
            raise FileNotFoundError(
                "--enable_aux_losses requires existing --reference_image"
            )
        aux_runtime = AvatarAuxTrainingRuntime(
            checkpoint_dir,
            args.reference_image,
            torch.device(device),
            training_stage=args.aux_training_stage,
            perceptual_backend=args.perceptual_backend,
            stage_schedule=AvatarAuxStageSchedule(
                stage2_step=args.aux_stage2_step,
                stage3_step=args.aux_stage3_step,
                stage4_step=args.aux_stage4_step,
            ),
        )
        _phase(
            f"aux losses enabled stage={args.aux_training_stage} "
            f"backend={args.perceptual_backend} ref={args.reference_image}"
        )

    opt_params = list(lora_network.prepare_optimizer_params(args.lr))
    if aux_runtime is not None:
        opt_params.extend(list(aux_runtime.trainable_parameters()))
    optimizer = torch.optim.AdamW(opt_params, weight_decay=weight_decay)

    ema_decay = float(args.ema_decay)
    ema_state: Dict[str, torch.Tensor] | None = None
    if ema_decay > 0.0:
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        ema_state = {k: v.detach().cpu().clone() for k, v in lora_network.state_dict().items()}
        _phase(f"EMA enabled decay={ema_decay}")

    meta = {
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_key": args.lora_key,
        "lora_scope": args.lora_scope,
        "lora_prefixes": list(prefixes) if prefixes else args.lora_scope,
        "min_snr_gamma": args.min_snr_gamma,
        "normalize_audio_embs": args.normalize_audio_embs,
        "ema_decay": ema_decay,
        "layer_count": len(lora_network.loras),
        "checkpoint_dir": checkpoint_dir,
        "resume_lora_path": args.resume_lora_path,
        "start_step": args.start_step,
        "enable_aux_losses": args.enable_aux_losses,
        "aux_training_stage": args.aux_training_stage,
        "aux_stage2_step": args.aux_stage2_step,
        "aux_stage3_step": args.aux_stage3_step,
        "aux_stage4_step": args.aux_stage4_step,
        "aux_loss_weight": args.aux_loss_weight,
        "perceptual_backend": args.perceptual_backend,
        "reference_image": args.reference_image,
    }
    with open(os.path.join(args.output_dir, "lora_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    step = max(0, int(args.start_step))
    if step >= args.max_steps:
        raise ValueError(f"start_step ({step}) must be < max_steps ({args.max_steps})")
    if step > 0:
        _phase(f"continuing from step {step} → max_steps {args.max_steps}")
    _phase("training loop start")

    for batch in batch_iter:
        batch = _to_device(batch, device, fw_dtype)
        latents = squeeze_collated_singleton_batch_dim(batch["latents"])
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        noise = squeeze_collated_singleton_batch_dim(batch["noise"])
        if noise.ndim == 4:
            noise = noise.unsqueeze(0)
        prompt_embeds = normalize_prompt_embeds_batch(batch["prompt_embeds"])
        prompt_mask = batch["prompt_mask"]
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask.unsqueeze(0)
        while prompt_mask.dim() > 2 and prompt_mask.size(1) == 1:
            prompt_mask = prompt_mask.squeeze(1)
        timesteps = batch["timesteps"].view(-1)
        audio_embs = squeeze_collated_singleton_batch_dim(batch["audio_embs"])
        if audio_embs.ndim == 4:
            audio_embs = audio_embs.unsqueeze(0)
        if args.normalize_audio_embs:
            audio_embs = stabilize_audio_embs(audio_embs)

        # Gradient checkpointing + bf16: forward under autocast; MSE in float32 for backward.
        amp_cm = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" and fw_dtype == torch.bfloat16
            else contextlib.nullcontext()
        )
        with amp_cm:
            noise_pred = dit(
                hidden_states=latents,
                timestep=timesteps.to(dtype=fw_dtype),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                audio_embs=audio_embs,
            )

        loss = avatar_lora_diffusion_loss(
            noise_pred,
            noise,
            timesteps,
            train_scheduler,
            min_snr_gamma=args.min_snr_gamma,
        )

        aux_log: dict | None = None
        if aux_runtime is not None:
            aux_total, aux_log = aux_runtime.compute_aux_loss(
                latents=latents,
                audio_embs=audio_embs,
                global_step=step,
            )
            loss = loss + args.aux_loss_weight * aux_total

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_network.parameters(), cfg.max_grad_norm)
        if aux_runtime is not None:
            aux_params = list(aux_runtime.trainable_parameters())
            if aux_params:
                torch.nn.utils.clip_grad_norm_(aux_params, cfg.max_grad_norm)
        optimizer.step()

        if ema_state is not None:
            with torch.no_grad():
                for k, v in lora_network.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v.detach().cpu(), alpha=1.0 - ema_decay)

        if step % 50 == 0 or (step > 0 and step % args.save_every == 0):
            msg = f"step {step} loss={loss.item():.6f}"
            if aux_log is not None:
                msg += (
                    f" aux={aux_log['total'].item():.6f}"
                    f" stage={aux_runtime.loss_module.training_stage if aux_runtime else '-'}"
                )
                if "raw_log_vars" in aux_log:
                    msg += f" raw_log_vars={aux_log['raw_log_vars'].tolist()}"
                if "compute_perceptual" in aux_log:
                    msg += f" perceptual={bool(aux_log['compute_perceptual'].item())}"
            _phase(msg)

        if step > 0 and step % args.save_every == 0:
            sub = os.path.join(args.output_dir, f"{args.lora_key}_step_{step}.safetensors")
            save_file({k: v.detach().cpu() for k, v in lora_network.state_dict().items()}, sub)
            _phase(f"saved {sub}")
            if ema_state is not None:
                ema_sub = os.path.join(args.output_dir, f"{args.lora_key}_step_{step}_ema.safetensors")
                save_file(ema_state, ema_sub)
                _phase(f"saved {ema_sub}")

        step += 1
        if step >= args.max_steps:
            break

    final_path = os.path.join(args.output_dir, f"{args.lora_key}_final.safetensors")
    save_file({k: v.detach().cpu() for k, v in lora_network.state_dict().items()}, final_path)
    _phase(f"training complete → {final_path}")
    if ema_state is not None:
        ema_final = os.path.join(args.output_dir, f"{args.lora_key}_final_ema.safetensors")
        save_file(ema_state, ema_final)
        _phase(f"training complete EMA → {ema_final}")


if __name__ == "__main__":
    main()
