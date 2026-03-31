# Latent export for training

Training scripts [`scripts/train.py`](../scripts/train.py) and [`scripts/train_lora_avatar.py`](../scripts/train_lora_avatar.py) expect pre-baked tensors per sample (`.pt`/`.npz`): `latents`, `prompt_embeds`, `prompt_mask`, `timesteps`, `noise`, and for avatar runs `audio_embs`.

## Implemented exporter (single sample)

[`scripts/export_latent_training_sample.py`](../scripts/export_latent_training_sample.py) builds **one** `.pt` aligned with [`LatentDataset`](../scripts/train.py):

- Text: `encode_prompt(..., do_classifier_free_guidance=False)` — single branch (no CFG doubling, no identity bank).
- Latents: `prepare_latents` from the reference image, then **flow-matching noise** via `scheduler.scale_noise(z0, t, eps)` so `latents` in the file is the noisy input at `t` and `noise` is `eps` (MSE target in the current train loop).
- Audio: same windowing as [`scripts/infer.py`](../scripts/infer.py) via [`arachne_x/inference_audio.py`](../arachne_x/inference_audio.py) and `pipe._prepare_audio_emb_for_dit`.

CLI (weights resolution matches other scripts; optional Hub):

```bash
python scripts/export_latent_training_sample.py \
  --checkpoint_dir /path/to/weights \
  --image ref.png \
  --audio speech.wav \
  --prompt "..." \
  --output sample.pt
```

Use `--allow_hub_download` for an `org/model` id when weights are not local.

## Limitations

- One sample per run; no batch dataset builder.
- Identity tokens and emotion/CFG-doubled layouts are out of scope for this exporter.

## Building blocks (reference)

- VAE + text: avatar pipeline components (`UMT5`, `AutoencoderKLWan`).
- Audio: `ArachneXVideoAvatarPipeline.get_audio_embedding` in [`arachne_x/pipeline_arachne_x_video_avatar.py`](../arachne_x/pipeline_arachne_x_video_avatar.py).
