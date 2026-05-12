# ARACHNE-X GTM — VAE / latent ABI (контракт совместимости)

Источник правды в коде: [`arachne_x/modules/autoencoder_kl_wan.py`](../../arachne_x/modules/autoencoder_kl_wan.py) (`AutoencoderKLWan`), [`vae/config.json`](../../vae/config.json) (часть полей), пайплайны [`arachne_x/pipeline_arachne_x_video_avatar.py`](../../arachne_x/pipeline_arachne_x_video_avatar.py) и [`arachne_x/pipeline_arachne_x_video.py`](../../arachne_x/pipeline_arachne_x_video.py).

## Зафиксированный ABI (текущий Wan-style VAE)

| Параметр | Значение | Где используется |
|----------|----------|------------------|
| `z_dim` | **16** | Каналы латента DiT `in_channels` / `out_channels` (по умолчанию 16) |
| `scale_factor_temporal` | **4** | `vae_scale_factor_temporal`, правило кадров `(num_frames - 1) % 4 == 0`, audio stride |
| `scale_factor_spatial` | **8** | `vae_scale_factor_spatial`, делимость H/W |
| `latents_mean`, `latents_std` | длина **16** каждый | `normalize_latents` / `denormalize_latents` в пайплайнах |
| RGB I/O | `in_channels=3`, `out_channels=3` | encode/decode видео |
| `temperal_downsample` | `[false, true, true]` | конфиг энкодера (типичный Wan layout) |

Любой новый VAE **должен** либо:

1. Совпадать по всем полям выше (или эквивалентно давать тот же tensor shape после encode), либо  
2. Явно ломать ABI и тогда обновляются: DiT `in_channels`/`out_channels`, projection слои, `latents_mean/std`, логика `prepare_latents`, экспорт латентов (`scripts/export_latent_*`, `training_latent_*`), `build_avatar_windowed_audio_emb` (stride от temporal factor).

## Критерии приёмки нового VAE

- **Shape:** `encode(video)` → `[B, z_dim, T_lat, H_lat, W_lat]` с теми же отношениями к входу, что ожидает DiT, или задокументирован mapping через adapter.
- **Нормализация:** либо те же `latents_mean/std`, либо новые + переобучение DiT на новой статистике.
- **Temporal:** сохранение `(F-1) % temporal_factor == 0` или обновление всех мест с `num_frames` / continuation.
- **Реконструкция:** PSNR/LPIPS/temporal flicker на hold-out до merge в production weights.
- **Совместимость весов:** старые чекпоинты DiT без адаптера работают только при полном ABI-match.

## Ссылка на реализацию

- Нормализация латентов: `LongCatVideoAvatarPipeline.normalize_latents` / `denormalize_latents` (тот же паттерн в base pipeline).
