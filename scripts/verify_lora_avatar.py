"""
Smoke tests for avatar LoRA key format and save/load roundtrip.

1) Toy nn.Module: build_initial_lora_state_dict -> create_lora_network ->
   safetensors -> fresh model + load_state_dict (tensor-equal spot check).
2) Optional: real DiT if --checkpoint_dir/avatar_single exists: save LoRA, load_lora + enable_loras.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetensors.torch import load_file, save_file

from arachne_x.modules.lora_utils import (
    build_initial_lora_state_dict,
    create_lora_network,
    default_avatar_train_lora_filter,
)


class _Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)


class ToyAvatarLikeDiT(nn.Module):
    """Minimal paths: blocks.* (x_embedder skipped by filter)."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.x_embedder = nn.Identity()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(2)])


def _run_toy() -> None:
    d = 32
    m1 = ToyAvatarLikeDiT(d)
    st = build_initial_lora_state_dict(
        m1,
        rank=4,
        alpha=8.0,
        name_filter=lambda name, mod: default_avatar_train_lora_filter(name, mod),
    )
    net = create_lora_network(m1, st, 1.0, 4, 8.0)
    inc = net.load_state_dict(st, strict=True)
    assert not inc.missing_keys and not inc.unexpected_keys

    path = tempfile.mktemp(suffix=".safetensors")
    try:
        save_file({k: v.cpu() for k, v in net.state_dict().items()}, path)
        loaded = load_file(path, device="cpu")
        m2 = ToyAvatarLikeDiT(d)
        net2 = create_lora_network(m2, loaded, 1.0, 4, 8.0)
        net2.load_state_dict(loaded, strict=False)
        k = next(iter(st))
        assert torch.allclose(net2.state_dict()[k].float(), st[k].float())
    finally:
        if os.path.isfile(path):
            os.remove(path)

    print("toy lora roundtrip OK")


def _run_real_checkpoint(checkpoint_dir: str) -> None:
    from arachne_x.modules.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="avatar_single", torch_dtype=dtype
    )
    dit.to(device)
    dit.requires_grad_(False)

    st = build_initial_lora_state_dict(
        dit,
        rank=8,
        alpha=16.0,
        name_filter=lambda n, m: default_avatar_train_lora_filter(n, m),
        dtype=torch.float32,
    )
    net = create_lora_network(dit, st, 1.0, 8, 16.0)
    net.load_state_dict(st, strict=True)
    net.to(device=device, dtype=dtype)

    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    try:
        save_file({k: v.cpu() for k, v in net.state_dict().items()}, path)

        dit.lora_dict.clear()
        dit.active_loras.clear()
        dit.disable_all_loras()
        dit.load_lora(path, "verify", multiplier=1.0, lora_network_dim=8, lora_network_alpha=16.0)
        dit.enable_loras(["verify"])
        print(f"real checkpoint LoRA load+enable OK ({len(dit.lora_dict['verify'].loras)} modules)")
    finally:
        dit.disable_all_loras()
        if os.path.isfile(path):
            os.remove(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="If set and .../avatar_single exists, run load_lora on real DiT.",
    )
    args = parser.parse_args()

    _run_toy()

    sub = (
        os.path.join(args.checkpoint_dir, "avatar_single")
        if args.checkpoint_dir
        else None
    )
    if sub and os.path.isdir(sub):
        _run_real_checkpoint(args.checkpoint_dir)
    elif args.checkpoint_dir:
        print(f"skip real checkpoint: missing {sub}")


if __name__ == "__main__":
    main()
