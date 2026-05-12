# ARACHNE-X-ULTRA V2 — «Ночная Фурия» — данные, eval gates и CI

Система: **proprietary NULLXES**, **independently trained**, **production checkpoints**. Публикация весов: [ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO), [ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR). Матрица готовности V2: [`GTM_ULTRA_V2_NIGHT_FURY.md`](GTM_ULTRA_V2_NIGHT_FURY.md).

Инференс в проде: **`arachne_x.runtime`** + **Inference Worker** (`services/longcat-worker/`); см. `GTM_PRODUCTION_CONTRACT.md`.

---

## 1. Цели качества (scope eval)

| Направление | Что измеряем |
|-------------|--------------|
| **VIDEO** | Временная согласованность, естественность движения, соответствие тексту (T2V / I2V / VC). |
| **AVATAR** | Lip-sync, стабильность идентичности, артефакты рта/фона, дрейф на длинных сегментах. |
| **Latency (realtime worker)** | Задержка до первого кадра, устойчивый throughput кадров — **отдельные SLO**, не смешивать с offline MOS без явной метки среды. |

---

## 2. Данные

См. [`data/datasets/README.md`](../../data/datasets/README.md) — контракт сырых данных и публичные наборы для внутренней подготовки (лицензии и consent — обязательная юридическая проверка перед продом).

Для **talking-head / digital employee**:

- Human-centric video + audio; long-form stability; при необходимости **proprietary** корпус с согласиями, метаданными watermark / tenant.
- Пайплайн: автоматические фильтры (blur, exposure, face size, SNR, cut) → при необходимости ASR для подписей → экспорт латентов `scripts/export_latent_*` + `pack_latent_shards_wds.py` / `train.py --wds_shards` по текущему контракту `training_latent_common.validate_latent_sample`.

Любая смена VAE → перегенерация latent shards или версионирование bucket (`latents_v2/` и т.д.); см. `GTM_VAE_ABI.md`.

---

## 3. Eval gates (merge в production weights)

**Gate** — обязательное условие для merge; результат — **pass / fail** по порогам относительно зафиксированного **baseline** (тег чекпоинта + hash манифеста eval).

### 3.1 Обязательные метрики (автоматизируемые)

| ID | Метрика | Назначение | Порог (тип) |
|----|---------|------------|-------------|
| **E-VAE-REC** | Reconstruction (при изменении VAE): rFID или LPIPS на hold-out кадрах | Регрессия качества декодера | ≤ baseline + ε (ε задаётся релизом) |
| **E-TEMP** | Temporal variance / flicker proxy (optical flow magnitude variance или эквивалент в репозитории) | Стабильность во времени | не хуже baseline |
| **E-LIPS** | Lip-sync proxy (sync confidence, landmark DTW при наличии GT audio) | Аватар + аудио | не ниже baseline |
| **E-ID** | Identity: cosine по фиксированному **внутреннему** эмбеддеру лица на фиксированном наборе идентичностей | Сохранение лица | не ниже baseline |
| **E-T2V** | Внутренний text–video alignment score (если зафиксирован в tooling репозитория) | Соответствие промпту | не ниже baseline |

### 3.2 Gate человека (не автоматизируется в CI полностью)

| ID | Метрика | Порог |
|----|---------|-------|
| **E-MOS** | Internal panel MOS (шкала фиксирована протоколом) | N≥K raters, медиана ≥ порога релиза |

---

## 4. Формат CI проверки

**Артефакты:**

- `eval_manifest.json` — список клипов: `clip_id`, пути к входам (изображение / видео / аудио), `prompt`, `negative_prompt`, ожидаемый `mode`, опционально `reference_face_id`.
- `eval_baseline.json` — численные пороги и версия baseline-чекпоинта.

**Шаги job (логический pipeline, без привязки к CI-платформе):**

1. **Setup:** загрузка **production checkpoint** (VIDEO или AVATAR) в кэш; фиксация `CUDA_VISIBLE_DEVICES` и seed policy в конфиге job.
2. **Generate:** вызов **`arachne_x.runtime`** (или HTTP worker с тем же чекпоинтом — явно указать в конфиге job один путь, см. `GTM_PRODUCTION_CONTRACT.md`).
3. **Score:** скрипты метрик E-*; запись `eval_report.json` (все числа + pass/fail по gate).
4. **Compare:** каждый gate E-* против `eval_baseline.json`; E-MOS — вне обязательного nightly, по расписанию релиза.

**Критерий merge PR с весами:** все E-* из секции 3.1 = **pass**; нарушение любого = **fail**; E-MOS = **release gate** на ветке релиза.

**Частота:** nightly на фиксированном манифесте; на каждый PR с изменением `arachne_x/modules` (VAE/DiT/attention) — полный eval VIDEO+AVATAR по матрице затронутых режимов.

---

## 5. Связь с кодом обучения

- Контракт латентного сэмпла: [`arachne_x/training_latent_common.py`](../../arachne_x/training_latent_common.py) (`validate_latent_sample`).
- Экспорт одного сэмпла: [`scripts/export_latent_training_sample.py`](../../scripts/export_latent_training_sample.py).

---

## 6. Покрытие режимов (матрица минимум)

| Режим runtime / CLI | Обязательный в nightly |
|---------------------|-------------------------|
| `t2v`, `i2v`, `vc` | да (VIDEO checkpoint) |
| `ai2v`, `streaming_ai2v` | да (AVATAR checkpoint) |
| `at2v`, `avc` | да, если в релизной матрице продуктов |
| `enroll_identity` | по флагу продукта |

---

## 7. Realtime / NIGHT FURY V2 (NDJSON worker)

**Цель:** те же gates **E-LIPS**, **E-ID**, **E-TEMP** плюс **E-MOS** (человек) на коротком **stress-стриме** (несколько минут аудио чанков через Inference Worker), без смешения с offline MOS без пометки среды.

**Расширение `eval_manifest.json` (опциональные поля):**

| Поле | Назначение |
|------|------------|
| `stream_profile` | `ndjson_worker` — вызов HTTP стрима вместо только CLI |
| `pcm_chunks_wav` | Путь к WAV с последовательностью чанков как в проде |
| `session_duration_sec` | Длительность стресс-сессии |

**Процедура:** после генерации NDJSON-лога кадров — прогон тех же скореров E-* где применимо (E-TEMP по последовательности кадров; E-LIPS при наличии референс-аудио); **fail любого gate = fail merge** для веток, затрагивающих realtime-путь (`arachne_x/runtime/avatar_serving.py`, worker, streaming в пайплайне).

См. также: [`GTM_NIGHT_FURY_V2_DIRECTIVE.md`](GTM_NIGHT_FURY_V2_DIRECTIVE.md), [`GTM_V2_RUNTIME_AUDIT.md`](GTM_V2_RUNTIME_AUDIT.md).

---

**NULLXES** · GTM eval spec · синхронизировано с `arachne_x.runtime` и production contract.
