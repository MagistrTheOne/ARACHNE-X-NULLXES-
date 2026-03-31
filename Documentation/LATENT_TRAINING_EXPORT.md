# Latent export for training (design backlog)

Training scripts [`scripts/train.py`](../scripts/train.py) and [`scripts/train_lora_avatar.py`](../scripts/train_lora_avatar.py) expect **pre-baked** tensors per sample (`.pt`/`.npz`): `latents`, `prompt_embeds`, `prompt_mask`, `timesteps`, `noise`, and for avatar runs `audio_embs`. There is **no** first-party exporter in this repository yet.

## Target contract

Align with `LatentDataset` in `scripts/train.py`: shapes must match what `LongCatVideoAvatarTransformer3DModel` expects in the training forward (see docstrings on `LatentDataset`).

## Building blocks already in-tree

- **VAE + text:** avatar pipeline encodes images/video and runs the text encoder inside `generate_*` paths; reuse the same `UMT5` + `AutoencoderKLWan` calls rather than reimplementing.
- **Audio embeddings:** `ArachneXVideoAvatarPipeline.get_audio_embedding` and related helpers in [`arachne_x/pipeline_arachne_x_video_avatar.py`](../arachne_x/pipeline_arachne_x_video_avatar.py).
- **Noise and timestep:** must follow the same schedule / distribution as the training loss (MSE on noise prediction): typically sample `timesteps` from the flow-match scheduler used at train time and sample Gaussian `noise` matching `latents` shape.

## Suggested future script

A minimal `scripts/export_latent_training_sample.py` could: load `load_avatar_pipeline`, take `--image`, `--audio`, `--prompt`, `--output sample.pt`, encode once, pack tensors, and save. Multi-sample batching and disk layout are product decisions.

This document tracks scope only; implementation is a separate epic.
