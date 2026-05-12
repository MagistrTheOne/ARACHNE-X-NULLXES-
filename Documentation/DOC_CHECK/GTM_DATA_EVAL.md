# ARACHNE-X GTM — данные и eval (качество до enterprise)

## Цели качества

1. **Video:** temporal consistency, motion naturalness, text alignment (T2V/I2V/VC).
2. **Avatar:** lip-sync, identity stability, mouth/background artifacts, long-segment drift.
3. **Realtime (продукт):** first-frame latency, sustained FPS, A/V sync — отдельный SLO от offline MOS.

## Данные (репозиторий уже ссылается)

См. [`data/datasets/README.md`](../../data/datasets/README.md):

- `tinytigerpan/tiger200k_preview`
- `TempoFunk/hdvila-100M` (subset)
- `Owen777/HQ-OpenHumanVid` (subset)

Дополнительно для **talking-head / digital employee** GTM:

| Направление | Примеры источников | Заметки |
|-------------|-------------------|---------|
| Human-centric video + audio | OpenHumanVid / HQ-OpenHumanVid | Капшены, motion; проверить лицензии и consent |
| Talking face | HDTF-style corpora на HF, VoxCeleb (идентичность, не обязательно full-body) | Следить за deepfake policy |
| Long-form stability | hdvila / internal high-quality interview footage | Сегменты + continuation |
| Dyadic / conversational** | SpeakerVid-5M (arxiv) | Если продукт — диалог агент↔кандидат |

**Если готового датасета нет:** собрать **proprietary** пакет (согласия, watermark metadata, tenant_id), затем:

1. Автоматический фильтр: blur, exposure, face size, SNR, cut detection.  
2. Pseudo-labels: ASR → текст; опционально emotion bucket.  
3. Экспорт в latent shards: `scripts/export_latent_*` + `pack_latent_shards_wds.py` → `scripts/train.py --wds_shards`.

## Eval gates (merge в production weights)

Минимальный набор **до** merge:

| Метрика | Назначение | Порог |
|---------|------------|-------|
| Reconstruction (VAE change) | rFID / LPIPS на hold-out кадрах | регрессия vs baseline |
| Temporal variance | optical flow / flicker proxy | не хуже baseline |
| Lip-sync proxy | sync confidence / landmark DTW (если есть GT audio) | не ниже baseline |
| Identity | ArcFace cosine на фиксированном сете идентичностей | не ниже baseline |
| Human MOS / internal panel | субъективно | заранее N≥K raters |

Автоматизация: nightly job на фиксированном `eval_manifest.json` (пути к клипам + промпты + аудио).

## Связь с обучением в репо

- Latent training contract: [`arachne_x/training_latent_common.py`](../../arachne_x/training_latent_common.py) (`validate_latent_sample`).
- Экспорт одного сэмпла: [`scripts/export_latent_training_sample.py`](../../scripts/export_latent_training_sample.py).

Любая смена VAE → **перегенерация** latent shards или версионирование bucket `latents_v2/`.
