# ARACHNE-X-ULTRA V2 — «Ночная Фурия» — архитектура репозитория и спецификация (NULLXES)

**Документ:** статический обзор по дереву исходников и внутренней документации репозитория.  
**Веса:** в данной рабочей копии **не хранятся**; полный бандл скачивается отдельно (см. раздел 2). Веса — **proprietary NULLXES**, **independently trained**, **production checkpoints**.  
**Ограничение:** обзор по дереву и контрактам; smoke-тесты Python — `tests/` и [`Documentation/REQUIREMENTS.md`](../REQUIREMENTS.md). Полный GPU infer — только Linux CUDA (RunPod).

---

## 1. Краткое саммари (NULLXES)

**ARACHNE-X-ULTRA V2 («Ночная Фурия»):** целевое состояние продукта — **stateful realtime avatar system**; техническая база репозитория — см. [`GTM_ULTRA_V2_NIGHT_FURY.md`](GTM_ULTRA_V2_NIGHT_FURY.md).

**ARACHNE-X** — платформа гиперреалистичных digital-human / talking-head (стриминг по аудио, lip-sync, идентичность) и **базового видео** (T2V, I2V, video continuation). Ядро — **3D Diffusion Transformer** (порядка **~13.6B** параметров в текущей линейке), **VAE**, **flow-match scheduler**, опционально **context parallel** и **block-sparse / FlashAttention** пути. Архитектура может следовать общим идеям 3D DiT; **все публикуемые веса NULLXES переобучены независимо**.

Четыре логических уровня репозитория:

1. **Библиотека `arachne_x/`** (включая **`runtime/`**) — загрузка весов (`loader.py`), базовый и аватарный пайплайны, **программный инференс** (`InferenceEngine`, `execute_infer`), стриминговый движок, модули DiT/VAE/scheduler, аватарные блоки, обучение/экспорт латентов (скрипты в `scripts/`, демо-конфиги в `Demo/`).
2. **`scripts/infer.py`** — тонкая CLI-обёртка над `arachne_x.runtime`.
3. **`services/arachnex-worker/`** — **Inference Worker**: FastAPI, **in-process** загрузка аватар-пайплайна и генерация (см. `gpu_avatar_runtime.py`). Канон исполнения DiT/VAE на GPU для HTTP-контрактов воркера. Переменные окружения: `NULLXES_CHECKPOINT_DIR` / `ARACHNE_CHECKPOINT_DIR` и др.
4. **`src/server/`** — **internal**: оркестрация продукта, вызов Inference Worker по URL (`NULLXES_AVATAR_INFERENCE_URL` и связанные ключи); не дублировать второй полноценный GPU-процесс с теми же весами в целевой топологии. Детали маршрутов — фактическое дерево `src/server/*.py`.

Публичный API пакета (по `arachne_x/__init__.py`): `WeightsLayout`, `load_base_pipeline`, `load_avatar_pipeline`, `get_vocal_separator_path`. Дополнительно для интеграций: **`arachne_x.runtime`**.

---

## 2. Веса и Hugging Face

| Источник | Назначение |
|----------|------------|
| [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) | **Production checkpoint** VIDEO: T2V, I2V, video-continuation, long-form и интерактивные режимы, заявленные на карточке. |
| [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) | **Production checkpoint** AVATAR: говорящий аватар, аудио-кондиционирование. |
| Локальный каталог после загрузки с Hub | Ожидаемый layout: `tokenizer/`, `vae/` и остальные подпапки по `WeightsLayout` / `loader.py` (`arachne_x/weights_resolve.py`). |

В git могут лежать **только метаданные** чекпоинта (индексы, `config.json`, scheduler) — **без** многогигабайтных шардов. Полный снапшот — на диске рантайма или загрузка с Hub (`weights_resolve.resolve_weights_root`, `allow_hub`).

---

## 3. Архитектура по каталогам

### 3.1 `arachne_x/` — ядро модели и пайплайнов

| Область | Файлы / папки | Роль |
|---------|----------------|------|
| Загрузка | `loader.py`, `weights_resolve.py` | Единая точка загрузки компонентов для инференса; опционально HF Hub → локальный root. |
| Runtime | `runtime/inference_engine.py` | Программный инференс (контракт CLI без дублирования). |
| Пайплайны | `pipeline_arachne_x_video.py`, `pipeline_arachne_x_video_avatar.py` | Базовое видео (T2V/I2V/VC) и аватар (ai2v/at2v/avc, streaming). |
| Стриминг | `streaming_inference.py` | KV-cache, буферы, real-time параметры (см. ARCHITECTURE.md → Runtime). |
| 3D DiT (видео) | `modules/arachne_video_dit.py` (legacy shim: `longcat_video_dit.py`), `attention.py`, `blocks.py`, `rope_3d.py` | Трансформер 3D + внимание. Класс публичного ABI: `LongCatVideoTransformer3DModel` (имя сохранено для чекпоинтов). |
| Аватар DiT | `modules/avatar/arachne_avatar_dit.py` (legacy shim: `longcat_video_dit_avatar.py`), `modules/avatar_losses.py` | Аудио-условие, anchors, потери. ABI: `LongCatVideoAvatarTransformer3DModel`. |
| VAE / scheduler | `modules/autoencoder_kl_wan.py`, `modules/scheduling_flow_match_euler_discrete.py` | Латенты, шаги диффузии. |
| Аудио | `audio_process/*`, `inference_audio.py` | Реальный пайплайн: **librosa → Wav2Vec2 → embedding**; при `speak_text`: **TTS → WAV → тот же пайплайн**; допустим fallback single-stream и отключение optional веток при ошибках. |
| Параллелизм | `context_parallel/*`, `block_sparse_attention/*` | CP, sparse attention. |
| Речь / TTS | `speech/*`, `tts/*` | Интеграции для сценариев с синтезом речи перед эмбеддингом. |
| Обучение | `training_*.py`, `scripts/train.py`, `scripts/train_lora_avatar.py` | Обучение на латентах; целевой вынос — см. `GTM_PRODUCTION_CONTRACT.md` §6. |

### 3.2 `src/server/` — internal оркестрация

- Сессии, HTTP/WebSocket поверх продукта, вызов Inference Worker для кадров аватара.
- Конфиг: `config/pipeline_config.defaults.json` и переменные `NULLXES_AVATAR_*`.

### 3.3 `services/arachnex-worker/` — Inference Worker (prod GPU serving)

- FastAPI; аватар: **in-process** `load_avatar_pipeline` → `avatar_serving` (lazy CUDA).
- Эндпоинты: `GET /health`, `POST /v1/realtime/avatar_frames`, `POST /v1/arachne/generate` (legacy `/v1/longcat/generate`).
- Запуск: см. [`services/arachnex-worker/README.md`](../../services/arachnex-worker/README.md), [`ARCHITECTURE.md`](../../ARCHITECTURE.md) §5, [`RUNPOD_H200_AVATAR_SETUP.md`](../../RUNPOD_H200_AVATAR_SETUP.md).
- VIDEO tasks в HTTP могут быть ограничены — полный VIDEO infer через **`arachne_x.runtime`** + `ARACHNE-X-ULTRA-VIDEO` ckpt.

### 3.3.1 Lineage vs production weights

| | |
|--|--|
| **Prod weights** | [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR), [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) |
| **Reference only** | [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video) — архитектурный класс / отчёт; **не** runtime checkpoint в NULLXES prod |
| **ABI** | Имена классов `LongCatVideo*Transformer3DModel` сохранены для совместимости config/shards |

### 3.4 Прочее

| Путь | Назначение |
|------|------------|
| `scripts/` | CLI train/export; **`infer.py`** — thin wrapper над `arachne_x.runtime`. |
| `Demo/` | Внутренние демо, не prod-only path (`GTM_FOLDER_AUDIT.md`). |
| `tests/` | Юнит-тесты. |
| `Documentation/` | GTM, контракты, деплой. |
| `assets/avatar/` | Примеры манифестов сцен. |

---

## 4. Спецификация продукта (целевые показатели, NULLXES README)

Значения — **как в проектной документации** (`README.md`); на железе зависят от сборки и чекпоинта.

| Категория | Параметр |
|-----------|-----------|
| Инференс (цель) | ~30 FPS streaming |
| Задержка кадра (цель) | порядка **< 33 ms** на кадр |
| Lip-sync | заявлено **> 95%** |
| Идентичность лица | заявлено **> 0.92** cosine (внутренняя метрика в стиле face-embedding) |
| VRAM (цель) | **~110–120 GB** (класс H200/H100/A100) |
| Базовая модель | **~13.6B** параметров DiT |
| Модальности | аудио, текст, изображение, видео |
| Параллелизм | Context Parallel (Ulysses) |

**Мультипоток аудио (концепт):** полосы для артикуляции, просодии, движения головы — по внутренней спецификации модулей аватара.

---

## 5. Публикация весов NULLXES

Официальные карточки **production checkpoints**:

- [ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO)
- [ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR)

Условия лицензирования и ограничения — **только по тексту карточек NULLXES на Hub**. Этот репозиторий — **код и оркестрация**; шардов весов в git нет.

---

## 6. Ограничения данного документа

- Не выполнялись команды окружения, не проверялись хэши весов и не валидировался HF snapshot.
- При расхождении с другими внутренними MD приоритет у **фактического дерева** и **`GTM_PRODUCTION_CONTRACT.md`**.

---

## 7. GTM architecture notes (2026)

- [GTM_ULTRA_V2_NIGHT_FURY.md](GTM_ULTRA_V2_NIGHT_FURY.md) — **V2 stateful realtime**, RunPod, аудит стека, дубли, чек-лист.
- [GTM_PRODUCTION_CONTRACT.md](GTM_PRODUCTION_CONTRACT.md) — **единый production-контур**, typed API (спека), train engine layout, Docker-роли.
- [GTM_FOLDER_AUDIT.md](GTM_FOLDER_AUDIT.md) — keep / refactor / archive.
- [GTM_VAE_ABI.md](GTM_VAE_ABI.md) — латентный ABI.
- [GTM_VAE_STRATEGY.md](GTM_VAE_STRATEGY.md) — смена VAE внутри NULLXES.
- [GTM_DATA_EVAL.md](GTM_DATA_EVAL.md) — eval gates и CI.
- Программный инференс: **`arachne_x.runtime`**, CLI — `scripts/infer.py`.

---

**NULLXES LLC** · внутренний обзор для инженеров и партнёров · может дополняться ссылками на `FULL_SYSTEM_OVERVIEW.md` и `ARACHNE_X_API_DOCUMENTATION.md`.
