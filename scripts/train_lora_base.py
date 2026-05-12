"""
Fine-tune the base text-to-video DiT with LoRA only (frozen base weights).

Input samples are the precomputed tensors exported by
``scripts/export_latent_training_sample_base.py``:
latents, prompt_embeds, prompt_mask, timesteps, noise.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Union

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Demo.training_config_h200 import H200TrainingConfig
from arachne_x.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from arachne_x.modules.lora_utils import build_initial_lora_state_dict, create_lora_network
from arachne_x.training_latent_common import (
    collate_latent_samples,
    normalize_prompt_embeds_batch,
    squeeze_collated_singleton_batch_dim,
    validate_latent_sample,
)
from arachne_x.training_wds import LatentWebDataset
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root


class LatentDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.files = sorted(glob(os.path.join(dataset_dir, "*.pt")) + glob(os.path.join(dataset_dir, "*.npz")))
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
        validate_latent_sample(sample, require_audio=False, source=f"{path}: ")
        return sample


def default_base_train_lora_filter(
    name: str,
    mod: torch.nn.Linear,
    include_prefixes: Optional[Union[Sequence[str], Set[str]]] = None,
) -> bool:
    """
    Conservative base DiT LoRA scope for visual style/identity tests.

    Keep the patch stem and text/timestep embedders frozen; adapt transformer blocks
    and final projection only.
    """
    del mod
    if name.startswith("x_embedder"):
        return False
    if include_prefixes is not None:
        return any(name.startswith(p) for p in include_prefixes)
    return name.startswith("blocks.") or name.startswith("final_layer.")


def _to_device(batch: Dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if k in {"prompt_mask", "timesteps"}:
            out[k] = v.to(device)
        else:
            out[k] = v.to(device=device, dtype=dtype)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ARACHNE-X base DiT LoRA training")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Root with dit/")
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument("--dataset_dir", type=str, default=None, help="Folder with *.pt / *.npz")
    ds.add_argument(
        "--wds_shards",
        type=str,
        default=None,
        help='WebDataset shards, e.g. "/data/shard_{000000..000009}.tar"',
    )
    parser.add_argument("--output_dir", type=str, default="./outputs_train_lora_base")
    parser.add_argument("--lora_key", type=str, default="train")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument(
        "--lora_prefixes",
        type=str,
        default="",
        help="Comma-separated Linear module prefixes. Empty = blocks.*, final_layer.*.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--wds_shuffle", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    add_resolve_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = H200TrainingConfig()
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.weight_decay

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

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
            require_audio=False,
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

    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir,
        subfolder="dit",
        torch_dtype=dtype,
    )
    dit.to(device)
    dit.requires_grad_(False)
    dit.train()

    if not args.no_gradient_checkpointing and hasattr(dit, "enable_gradient_checkpointing"):
        dit.enable_gradient_checkpointing()
        print("[train_lora_base] gradient_checkpointing on")

    prefixes = None
    if args.lora_prefixes.strip():
        prefixes = tuple(p.strip() for p in args.lora_prefixes.split(",") if p.strip())

    def name_filter(name: str, mod: torch.nn.Linear) -> bool:
        return default_base_train_lora_filter(name, mod, include_prefixes=prefixes)

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
    incompatible = lora_network.load_state_dict(init_state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, (
        incompatible.missing_keys,
        incompatible.unexpected_keys,
    )

    lora_network.to(device=device, dtype=dtype)
    lora_network.train()
    dit.lora_dict[args.lora_key] = lora_network
    dit.enable_loras([args.lora_key])

    optimizer = torch.optim.AdamW(
        lora_network.prepare_optimizer_params(args.lr),
        weight_decay=weight_decay,
        foreach=False,
    )

    meta = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_key": args.lora_key,
        "lora_prefixes": list(prefixes) if prefixes else "default_base_train_lora_filter",
        "layer_count": len(lora_network.loras),
        "checkpoint_dir": checkpoint_dir,
        "lr": args.lr,
        "max_steps": args.max_steps,
    }
    with open(os.path.join(args.output_dir, "lora_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    amp_cm_factory = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda" and dtype == torch.bfloat16
        else contextlib.nullcontext
    )

    step = 0
    for batch in batch_iter:
        batch = _to_device(batch, device, dtype)
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

        with amp_cm_factory():
            noise_pred = dit(
                hidden_states=latents,
                timestep=timesteps.to(dtype=dtype),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_network.parameters(), cfg.max_grad_norm)
        optimizer.step()

        if step % 25 == 0:
            print(f"[step {step}] loss={loss.item():.6f}")

        if step > 0 and step % args.save_every == 0:
            sub = os.path.join(args.output_dir, f"lora_step_{step}.safetensors")
            save_file({k: v.detach().cpu() for k, v in lora_network.state_dict().items()}, sub)
            print(f"saved {sub}")

        step += 1
        if step >= args.max_steps:
            break

    final_path = os.path.join(args.output_dir, "lora_final.safetensors")
    save_file({k: v.detach().cpu() for k, v in lora_network.state_dict().items()}, final_path)
    print(f"Training complete. LoRA saved to {final_path}")


if __name__ == "__main__":
    main()
