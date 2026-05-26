import argparse
import contextlib
import itertools
import json
import os
from glob import glob
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from arachne_x.modules.arachne_video_dit import LongCatVideoTransformer3DModel
from arachne_x.modules.avatar.arachne_avatar_dit import LongCatVideoAvatarTransformer3DModel
from arachne_x.training_latent_common import (
    collate_latent_samples,
    normalize_prompt_embeds_batch,
    squeeze_collated_singleton_batch_dim,
    validate_latent_sample,
)
from arachne_x.training_wds import LatentWebDataset
from arachne_x.weights_resolve import add_resolve_args, resolve_weights_root
from arachne_x.training_config_h200 import H200TrainingConfig


class LatentDataset(Dataset):
    """
    Dataset of precomputed training tensors.

    Expected keys per sample (.pt or .npz):
    - latents: [C, T, H, W] or [1, C, T, H, W]
    - prompt_embeds: [1, 1, S, D] or [1, S, D]
    - prompt_mask: [S] or [1, S]
    - timesteps: [1] or scalar
    - noise: same shape as latents
    - audio_embs (avatar only): [T, W, S2, D] or [1, T, W, S2, D]
    """

    def __init__(self, dataset_dir: str, *, require_audio: bool = False):
        self.files = sorted(glob(os.path.join(dataset_dir, "*.pt")) + glob(os.path.join(dataset_dir, "*.npz")))
        self.require_audio = require_audio
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

        validate_latent_sample(sample, require_audio=self.require_audio, source=f"{path}: ")

        return sample


def _to_device(batch: Dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if k in {"prompt_mask", "timesteps"}:
            out[k] = v.to(device)
        else:
            out[k] = v.to(device=device, dtype=dtype)
    return out


def _load_config(path: str) -> H200TrainingConfig:
    if path.endswith(".json"):
        return H200TrainingConfig.from_json(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return H200TrainingConfig(**cfg)


def _make_optimizer(model: torch.nn.Module, args: argparse.Namespace, weight_decay: float, device: str):
    lr = args.lr
    # Avatar full fine-tune on one GPU: FP32 Adam states OOM; use 8-bit optimizer by default when bnb is available.
    want_8bit = False
    if device == "cuda":
        want_8bit = bool(args.adam8bit) or (args.mode == "avatar" and not args.no_adam8bit)

    if want_8bit:
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError:
            if args.adam8bit:
                raise ImportError("Install bitsandbytes for --adam8bit: pip install bitsandbytes") from None
            print(
                "[train] WARNING: bitsandbytes not installed; using AdamW(foreach=False). "
                "Avatar full fine-tune usually OOMs at optimizer.step — pip install bitsandbytes"
            )
        else:
            print("[train] optimizer=AdamW8bit (bitsandbytes)")
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)

    opt_kw: dict = {"lr": lr, "weight_decay": weight_decay, "fused": False}
    if device == "cuda":
        opt_kw["foreach"] = bool(args.adam_foreach)
    print(f"[train] optimizer=AdamW(foreach={opt_kw.get('foreach', 'n/a')})")
    return torch.optim.AdamW(model.parameters(), **opt_kw)


def _output_dir_arg(value: str) -> str:
    """Treat empty / whitespace as missing (e.g. ``--output_dir "$OUT"`` when OUT is unset)."""
    s = (value or "").strip()
    if not s:
        print("[train] --output_dir empty; using ./outputs_train (set OUT or pass e.g. --output_dir /workspace/out)")
        return "./outputs_train"
    return s


def main():
    parser = argparse.ArgumentParser(description="ARACHNE-X training entrypoint")
    parser.add_argument("--mode", type=str, choices=["base", "avatar"], required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    ds = parser.add_mutually_exclusive_group(required=True)
    ds.add_argument("--dataset_dir", type=str, default=None, help="Folder with *.pt / *.npz (LatentDataset)")
    ds.add_argument(
        "--wds_shards",
        type=str,
        default=None,
        help='WebDataset shards URL or brace glob, e.g. "/data/shard_{000000..000099}.tar"',
    )
    parser.add_argument(
        "--output_dir",
        type=_output_dir_arg,
        default="./outputs_train",
        help="Checkpoint output directory (default: ./outputs_train). Empty string falls back to default.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument(
        "--wds_shuffle",
        type=int,
        default=5000,
        help="WebDataset shuffle buffer (0 = off). Ignored for --dataset_dir.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers (map-style: shuffle each epoch; WDS uses worker split).",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable DiT activation checkpointing (faster steps, much higher VRAM). Default: on when "
        "H200TrainingConfig.use_gradient_checkpointing is True.",
    )
    parser.add_argument(
        "--adam8bit",
        action="store_true",
        help="Force bitsandbytes AdamW8bit (also default for --mode avatar on CUDA unless --no_adam8bit).",
    )
    parser.add_argument(
        "--no_adam8bit",
        action="store_true",
        help="Do not use AdamW8bit for avatar on CUDA (plain AdamW; needs huge VRAM for optimizer state).",
    )
    parser.add_argument(
        "--adam_foreach",
        action="store_true",
        help="AdamW multi-tensor foreach (faster optimizer.step, higher peak VRAM). Default: off.",
    )
    add_resolve_args(parser)
    args = parser.parse_args()
    # Again in case anything assigned Namespace after parse (defensive)
    args.output_dir = _output_dir_arg(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    config = H200TrainingConfig() if args.config is None else _load_config(args.config)

    checkpoint_dir = resolve_weights_root(
        args.checkpoint_dir,
        allow_hub=args.allow_hub_download,
        cache_dir=args.weights_cache_dir,
    )

    if args.dataset_dir:
        dataset = LatentDataset(args.dataset_dir, require_audio=(args.mode == "avatar"))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
    else:
        use_audio = args.mode == "avatar"
        iterable = LatentWebDataset(
            args.wds_shards,
            require_audio=use_audio,
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

    # Map-style loader exhausts after one epoch; cycle so max_steps works with few .pt files.
    batch_iter: Iterable = itertools.cycle(loader) if args.dataset_dir else loader

    if args.mode == "base":
        model = LongCatVideoTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder="dit",
            torch_dtype=dtype,
        )
        use_audio = False
    else:
        model = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder="avatar_single",
            torch_dtype=dtype,
        )
        use_audio = True

    model.to(device)
    model.train()

    use_gc = bool(getattr(config, "use_gradient_checkpointing", True)) and not args.no_gradient_checkpointing
    # Diffusers ModelMixin: must call enable_gradient_checkpointing() so _gradient_checkpointing_func is set;
    # setting gradient_checkpointing=True alone leaves _gradient_checkpointing_func=None and crashes in forward.
    if use_gc and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        print("[train] gradient_checkpointing on (enable_gradient_checkpointing(); --no_gradient_checkpointing to disable)")
    elif hasattr(model, "disable_gradient_checkpointing"):
        model.disable_gradient_checkpointing()

    optimizer = _make_optimizer(model, args, config.weight_decay, device)

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

        # Gradient checkpointing + bf16: backward recomputation must see the same autocast as forward
        # or PyTorch can raise "Found dtype BFloat16 but expected Float" during loss.backward().
        amp_cm = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" and dtype == torch.bfloat16
            else contextlib.nullcontext()
        )
        with amp_cm:
            if use_audio:
                if "audio_embs" not in batch:
                    raise KeyError("avatar training requires audio_embs in dataset")
                audio_embs = squeeze_collated_singleton_batch_dim(batch["audio_embs"])
                if audio_embs.ndim == 4:
                    audio_embs = audio_embs.unsqueeze(0)
                noise_pred = model(
                    hidden_states=latents,
                    timestep=timesteps.to(dtype=dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    audio_embs=audio_embs,
                )
            else:
                noise_pred = model(
                    hidden_states=latents,
                    timestep=timesteps.to(dtype=dtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                )

        loss = F.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if device == "cuda":
            torch.cuda.empty_cache()
        optimizer.step()

        if step % 50 == 0:
            print(f"[step {step}] loss={loss.item():.6f}")

        if step > 0 and step % args.save_every == 0:
            save_dir = os.path.join(args.output_dir, f"checkpoint_{step}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)

        step += 1
        if step >= args.max_steps:
            break

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Training complete. Saved to {final_dir}")


if __name__ == "__main__":
    main()
