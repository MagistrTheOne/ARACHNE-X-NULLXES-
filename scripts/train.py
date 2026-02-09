import argparse
import json
import os
from dataclasses import asdict
from glob import glob
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from arachne_x.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from arachne_x.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from Demo.training_config_h200 import H200TrainingConfig


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

        required = {"latents", "prompt_embeds", "prompt_mask", "timesteps", "noise"}
        missing = required - set(sample.keys())
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")

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


def main():
    parser = argparse.ArgumentParser(description="ARACHNE-X training entrypoint")
    parser.add_argument("--mode", type=str, choices=["base", "avatar"], required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_train")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    config = H200TrainingConfig() if args.config is None else _load_config(args.config)

    dataset = LatentDataset(args.dataset_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    if args.mode == "base":
        model = LongCatVideoTransformer3DModel.from_pretrained(
            args.checkpoint_dir,
            subfolder="dit",
            torch_dtype=dtype,
        )
        use_audio = False
    else:
        model = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            args.checkpoint_dir,
            subfolder="avatar_single",
            torch_dtype=dtype,
        )
        use_audio = True

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)

    step = 0
    for batch in loader:
        batch = _to_device(batch, device, dtype)
        latents = batch["latents"]
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        noise = batch["noise"]
        if noise.ndim == 4:
            noise = noise.unsqueeze(0)
        prompt_embeds = batch["prompt_embeds"]
        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_mask = batch["prompt_mask"]
        if prompt_mask.ndim == 1:
            prompt_mask = prompt_mask.unsqueeze(0)
        timesteps = batch["timesteps"].view(-1)

        if use_audio:
            if "audio_embs" not in batch:
                raise KeyError("avatar training requires audio_embs in dataset")
            audio_embs = batch["audio_embs"]
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

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
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
