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

## Batch from URLs (вставить только ссылки)

1. Скопируйте [`examples/avatar_latent_url_manifest.example.json`](../examples/avatar_latent_url_manifest.example.json) в свой файл (массив объектов или `{ "samples": [ ... ] }`).
2. Подставьте рабочие `image_url` и `audio_url` (https к jpg/png и wav и т.д.; с Hugging Face удобны [постоянные ссылки на raw-файлы](https://huggingface.co/docs/hub/main/storage-backends#raw-github-style-urls) или ваш CDN).
3. Один запуск (пайплайн грузится **один раз**):

```bash
python scripts/export_latent_batch_from_urls.py \
  --checkpoint_dir /path/to/weights \
  --manifest my_manifest.json \
  --output_dir /path/to/latent_dataset \
  --resolution 480p
```

На выходе: `output_dir/{id}.pt` для `LatentDataset` / `--mode avatar`.

## Pod launcher (`arachne_x_train`)

Для одного запуска на поде с переменными окружения: [`scripts/arachne_x_train.py`](../scripts/arachne_x_train.py) (обёртка над `train.py`). См. docstring в файле: `ARACHNE_CHECKPOINT_DIR`, `ARACHNE_DATASET_DIR`, `ARACHNE_TRAIN_MODE`, `ARACHNE_MERGE_INTO`, и т.д.

## Limitations

- One sample per run; no batch dataset builder.
- Identity tokens and emotion/CFG-doubled layouts are out of scope for this exporter.

## Building blocks (reference)

- VAE + text: avatar pipeline components (`UMT5`, `AutoencoderKLWan`).
- Audio: `ArachneXVideoAvatarPipeline.get_audio_embedding` in [`arachne_x/pipeline_arachne_x_video_avatar.py`](../arachne_x/pipeline_arachne_x_video_avatar.py).
