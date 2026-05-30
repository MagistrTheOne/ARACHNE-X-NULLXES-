#!/usr/bin/env python3
"""
ARACHNE-X DiT trainer — continued pretrain / LoRA over precomputed flow-match latents.

This is the missing training *driver* referenced by
``scripts/prepare_foundation_train_pack.py``. It turns the existing building
blocks (latent export → flow-match target → Min-SNR loss) into actual gradient
steps, so the depth-expanded foundation weights stop being initialization-only.

Why no VAE / text encoder here
------------------------------
Training samples (``training_latent_export*``) already bake ``latents`` (noisy
z_t), ``noise`` (eps target), ``timesteps`` (t), and ``prompt_embeds`` /
``prompt_mask``. So the trainer loads **only the DiT + a flow-match scheduler** —
this is exactly why the foundation repo (DiT-only safetensors at root) is
trainable without the full runtime tree.

Targets
-------
- ``--model foundation``  : 50B depth-expanded backbone (``ARACHNE_FOUNDATION_CKPT``,
  DiT safetensors at repo root).
- ``--model base13b``     : our 13.6B base video DiT (``$NULLXES_CHECKPOINT_DIR/dit``).
- ``--dit_dir PATH``      : explicit override.

Modes
-----
- ``--mode lora``  : freeze base, train attention-only LoRA (NULLXES policy). Fits a
  single GPU; cheap smoke that proves the loop learns.
- ``--mode full``  : train all DiT params. 50B requires sharding — launch under
  ``torchrun`` and the trainer wraps the model in FSDP (full-shard, bf16).

Launch recipes
--------------
  # 13B LoRA smoke on one H200
  python scripts/train_arachne_dit.py --model base13b --mode lora \\
      --latents_dir /workspace/datasets/arachne-foundation-smoke/latents \\
      --out /workspace/runs/base13b-lora --micro_bsz 1 --grad_accum 8 --max_steps 500

  # 50B foundation continued-pretrain, full-shard across N GPUs
  torchrun --standalone --nproc_per_node=8 scripts/train_arachne_dit.py \\
      --model foundation --mode full --grad_checkpointing \\
      --latents_dir /workspace/datasets/arachne-foundation-smoke/latents \\
      --out /workspace/runs/foundation-cpt --micro_bsz 1 --grad_accum 16 --max_steps 20000

NOTE: each exported ``.pt`` bakes a single random timestep per clip. For serious
training, re-export with multiple t per clip (more diverse sigma coverage).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arachne_x.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from arachne_x.training_latent_common import collate_latent_samples, validate_latent_sample
from arachne_x.training_lora_loss import avatar_lora_diffusion_loss


# --------------------------------------------------------------------------- #
# distributed
# --------------------------------------------------------------------------- #
class DistContext:
    def __init__(self) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.enabled = self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def setup(self) -> torch.device:
        if self.enabled:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            return torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def barrier(self) -> None:
        if self.enabled:
            import torch.distributed as dist

            dist.barrier()

    def teardown(self) -> None:
        if self.enabled:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class LatentPtDataset(torch.utils.data.Dataset):
    """Map-style dataset over flat ``.pt`` latent training samples."""

    def __init__(self, paths: List[Path], *, require_audio: bool = False) -> None:
        if not paths:
            raise ValueError("no latent .pt samples found")
        self.paths = paths
        self.require_audio = require_audio

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        return validate_latent_sample(obj, require_audio=self.require_audio, source=f"{path}: ")


def _collect_latent_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        root = Path(args.manifest).resolve().parent
        for s in manifest.get("samples", []):
            rel = s.get("latent")
            if rel:
                paths.append((root / rel).resolve())
    if args.latents_dir:
        paths.extend(sorted(Path(args.latents_dir).glob("*.pt")))
    # de-dup, keep existing
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key in seen or not p.is_file():
            continue
        seen.add(key)
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def resolve_dit_dir(args: argparse.Namespace) -> str:
    if args.dit_dir:
        return args.dit_dir
    if args.model == "foundation":
        ckpt = (os.environ.get("ARACHNE_FOUNDATION_CKPT") or "").strip()
        if not ckpt:
            raise ValueError("set --dit_dir or ARACHNE_FOUNDATION_CKPT for --model foundation")
        return ckpt
    if args.model == "base13b":
        root = (os.environ.get("NULLXES_CHECKPOINT_DIR") or os.environ.get("ARACHNE_CHECKPOINT_DIR") or "").strip()
        if not root:
            raise ValueError("set --dit_dir or NULLXES_CHECKPOINT_DIR for --model base13b")
        return str(Path(root) / "dit")
    raise ValueError(f"unknown --model {args.model!r}")


def build_dit(args: argparse.Namespace, dtype: torch.dtype):
    from arachne_x.modules.arachne_video_dit import LongCatVideoTransformer3DModel

    dit_dir = resolve_dit_dir(args)
    logger.info("loading DiT model={} dir={} dtype={}", args.model, dit_dir, dtype)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        dit_dir,
        cp_split_hw=None,
        torch_dtype=dtype,
        local_files_only=True,
    )
    if args.grad_checkpointing:
        dit.gradient_checkpointing = True
        logger.info("gradient checkpointing enabled")
    return dit, dit_dir


_LORA_TRAIN_KEY = "train"


def attach_lora(dit, args: argparse.Namespace):
    """Freeze base, build attention-only LoRA, patch forwards, return (network, trainable_params).

    Uses the model's own LoRA lifecycle (``lora_dict`` + ``enable_loras``) so the
    forward routing is identical to inference. LoRA master weights are kept in
    fp32 for stable optimization while the base stays bf16.
    """
    from arachne_x.modules.lora_utils import (
        avatar_attention_only_lora_filter,
        build_initial_lora_state_dict,
        create_lora_network,
    )

    for p in dit.parameters():
        p.requires_grad_(False)

    init_sd = build_initial_lora_state_dict(
        dit,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        name_filter=avatar_attention_only_lora_filter,
        dtype=torch.float32,
    )
    network = create_lora_network(
        dit,
        init_sd,
        multiplier=1.0,
        network_dim=args.lora_rank,
        network_alpha=args.lora_alpha,
    )
    network.load_state_dict(init_sd, strict=True)

    if getattr(dit, "lora_dict", None) is None:
        dit.lora_dict = {}
    if getattr(dit, "active_loras", None) is None:
        dit.active_loras = []
    dit.lora_dict[_LORA_TRAIN_KEY] = network
    dit.enable_loras([_LORA_TRAIN_KEY])  # monkeypatch module.forward + move loras to device

    trainable: List[torch.nn.Parameter] = []
    for lora in network.loras:
        lora.lora_down.float()
        lora.lora_up.float()
        for p in (*lora.lora_down.parameters(), *lora.lora_up.parameters()):
            p.requires_grad_(True)
            trainable.append(p)
    logger.info("LoRA attached: {} modules, {} trainable tensors (fp32 master)", len(network.loras), len(trainable))
    return network, trainable


def resolve_scheduler_dir(args: argparse.Namespace, dit_dir: str) -> Optional[str]:
    """Latents were exported with the BASE video scheduler — load the same one."""
    candidates: List[Path] = []
    if args.scheduler_dir:
        candidates.append(Path(args.scheduler_dir))
    ckpt = (os.environ.get("NULLXES_CHECKPOINT_DIR") or os.environ.get("ARACHNE_CHECKPOINT_DIR") or "").strip()
    if ckpt:
        candidates.append(Path(ckpt) / "scheduler")
    candidates.append(Path(dit_dir) / "scheduler")
    candidates.append(Path(dit_dir).parent / "scheduler")
    for c in candidates:
        if c.is_dir():
            return str(c)
    return None


def maybe_fsdp(dit, dist: DistContext, dtype: torch.dtype):
    """Wrap DiT in FSDP full-shard (full mode, multi-GPU)."""
    from functools import partial

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    block_cls = type(dit.blocks[0])
    mp = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=dtype)
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={block_cls})
    wrapped = FSDP(
        dit,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=dist.local_rank if torch.cuda.is_available() else None,
        use_orig_params=True,
    )
    logger.info("FSDP full-shard wrap done (block_cls={})", block_cls.__name__)
    return wrapped


# --------------------------------------------------------------------------- #
# lr schedule
# --------------------------------------------------------------------------- #
def lr_factor(step: int, *, warmup: int, total: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(total - warmup, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cos


# --------------------------------------------------------------------------- #
# train step
# --------------------------------------------------------------------------- #
def forward_loss(dit, batch: Dict[str, torch.Tensor], scheduler, device, dtype, min_snr_gamma: float):
    latents = batch["latents"].to(device=device, dtype=dtype)
    noise = batch["noise"].to(device=device, dtype=torch.float32)
    timesteps = batch["timesteps"].reshape(latents.shape[0]).to(device=device)
    prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)
    prompt_mask = batch["prompt_mask"].to(device=device)

    noise_pred = dit(
        hidden_states=latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask,
        return_kv=False,
    )
    return avatar_lora_diffusion_loss(noise_pred, noise, timesteps, scheduler, min_snr_gamma=min_snr_gamma)


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def save_lora(network, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in network.state_dict().items()}
    tmp = out_dir / f".lora_step{step}.pt.tmp{os.getpid()}"
    final = out_dir / f"lora_step{step}.pt"
    torch.save(sd, tmp)
    os.replace(tmp, final)
    logger.info("saved LoRA checkpoint {}", final)


def save_full_fsdp(dit, dist: DistContext, out_dir: Path, step: int) -> None:
    import torch.distributed as dist_mod
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(dit, StateDictType.FULL_STATE_DICT, cfg):
        sd = dit.state_dict()
    if dist.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".dit_step{step}.pt.tmp{os.getpid()}"
        final = out_dir / f"dit_step{step}.pt"
        torch.save(sd, tmp)
        os.replace(tmp, final)
        logger.info("saved full DiT checkpoint {}", final)
    if dist.enabled:
        dist_mod.barrier()


def save_full_single(dit, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in dit.state_dict().items()}
    tmp = out_dir / f".dit_step{step}.pt.tmp{os.getpid()}"
    final = out_dir / f"dit_step{step}.pt"
    torch.save(sd, tmp)
    os.replace(tmp, final)
    logger.info("saved full DiT checkpoint {}", final)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARACHNE-X DiT trainer (flow-match continued pretrain / LoRA)")
    # model
    p.add_argument("--model", choices=["foundation", "base13b"], default="base13b")
    p.add_argument("--dit_dir", default=None, help="explicit DiT dir (overrides --model)")
    p.add_argument("--mode", choices=["lora", "full"], default="lora")
    # data
    p.add_argument("--latents_dir", default=None, help="dir of flat .pt latent samples")
    p.add_argument("--manifest", default=None, help="manifest.json with samples[].latent")
    p.add_argument("--scheduler_dir", default=None, help="flow-match scheduler dir (defaults to $NULLXES_CHECKPOINT_DIR/scheduler)")
    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--micro_bsz", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--min_snr_gamma", type=float, default=5.0)
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    # lora
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    # runtime
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output run dir")
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def build_optimizer(params, args: argparse.Namespace):
    if args.optim == "adamw8bit":
        try:
            import bitsandbytes as bnb

            logger.info("using bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        except ImportError:
            logger.warning("bitsandbytes not available; falling back to torch AdamW")
    return torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)


def main() -> None:
    args = parse_args()
    dist = DistContext()
    device = dist.setup()
    torch.manual_seed(args.seed + dist.rank)

    if not dist.is_main:
        logger.remove()
    logger.info(
        "trainer start model={} mode={} world_size={} device={} bf16=True",
        args.model, args.mode, dist.world_size, device,
    )

    dtype = torch.bfloat16 if (torch.cuda.is_available()) else torch.float32

    # --- data ---
    paths = _collect_latent_paths(args)
    if dist.is_main:
        logger.info("dataset: {} latent samples", len(paths))
    dataset = LatentPtDataset(paths, require_audio=False)
    sampler = None
    if dist.enabled:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.micro_bsz,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_latent_samples,
        pin_memory=torch.cuda.is_available(),
    )

    # --- model ---
    dit, dit_dir = build_dit(args, dtype)
    sched_dir = resolve_scheduler_dir(args, dit_dir)
    if sched_dir:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir, local_files_only=True)
        if dist.is_main:
            logger.info("flow-match scheduler loaded from {}", sched_dir)
    else:
        scheduler = FlowMatchEulerDiscreteScheduler()
        if dist.is_main:
            logger.warning(
                "no scheduler dir found (set --scheduler_dir or NULLXES_CHECKPOINT_DIR); "
                "using default FlowMatchEulerDiscreteScheduler — Min-SNR timestep mapping may drift. "
                "Use --min_snr_gamma 0 for plain MSE if t indices mismatch."
            )

    lora_net = None
    if args.mode == "lora":
        dit.to(device)
        lora_net, trainable = attach_lora(dit, args)
        for t in trainable:
            t.requires_grad_(True)
        opt_params = trainable
        train_module = dit  # base frozen, lora trainable
    else:  # full
        if dist.enabled:
            dit = maybe_fsdp(dit, dist, dtype)
        else:
            dit.to(device)
        opt_params = [p for p in dit.parameters() if p.requires_grad]
        train_module = dit

    optimizer = build_optimizer(opt_params, args)
    dit.train()

    # --- loop ---
    micro_step = 0
    opt_step = 0
    accum = max(args.grad_accum, 1)
    loss_accum = 0.0
    t_window = time.perf_counter()
    out_dir = Path(args.out)
    data_iter = _infinite(loader, sampler)

    optimizer.zero_grad(set_to_none=True)
    while opt_step < args.max_steps:
        batch = next(data_iter)
        loss = forward_loss(train_module, batch, scheduler, device, dtype, args.min_snr_gamma)
        (loss / accum).backward()
        loss_accum += float(loss.detach())
        micro_step += 1

        if micro_step % accum != 0:
            continue

        # optimizer step boundary
        if args.mode == "full" and dist.enabled:
            grad_norm = dit.clip_grad_norm_(args.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(opt_params, args.grad_clip)

        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_factor(opt_step, warmup=args.warmup_steps, total=args.max_steps, min_ratio=args.min_lr_ratio)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1

        if dist.is_main and (opt_step % args.log_every == 0 or opt_step == 1):
            dt = time.perf_counter() - t_window
            sps = (args.log_every * accum * args.micro_bsz * max(dist.world_size, 1)) / max(dt, 1e-6)
            mem = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
            logger.info(
                "step={} loss={:.5f} lr={:.2e} grad_norm={:.3f} samples/s={:.2f} max_mem_gb={:.1f}",
                opt_step,
                loss_accum / accum,
                optimizer.param_groups[0]["lr"],
                float(grad_norm),
                sps,
                mem,
            )
            t_window = time.perf_counter()
        loss_accum = 0.0

        if opt_step % args.save_every == 0:
            if args.mode == "lora":
                if dist.is_main:
                    save_lora(lora_net, out_dir, opt_step)
            elif dist.enabled:
                save_full_fsdp(dit, dist, out_dir, opt_step)
            elif dist.is_main:
                save_full_single(dit, out_dir, opt_step)

    # final save
    if args.mode == "lora":
        if dist.is_main:
            save_lora(lora_net, out_dir, opt_step)
    elif dist.enabled:
        save_full_fsdp(dit, dist, out_dir, opt_step)
    elif dist.is_main:
        save_full_single(dit, out_dir, opt_step)

    if dist.is_main:
        logger.info("training complete: {} optimizer steps", opt_step)
    dist.teardown()


def _infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


if __name__ == "__main__":
    main()
