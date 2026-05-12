# ARACHNE-X — архитектура репозитория и спецификация (NULLXES)

**Документ:** статический обзор по дереву исходников и внутренней документации репозитория.  
**Веса:** в данной рабочей копии **не хранятся**; полный бандл скачивается отдельно (см. раздел 2).  
**Ограничение:** без запуска окружения, без импорта Python и без проверки рантайма — только структура и контракты.

---

## 1. Краткое саммари (NULLXES)

**ARACHNE-X** — платформа гиперреалистичных digital-human / talking-head с упором на **real-time** (стриминг по аудио, lip-sync, идентичность) и на **базовое видео** (T2V, I2V, video continuation). Ядро — **Diffusion Transformer (~13.6B)** в духе LongCat-Video, **VAE**, **flow-match scheduler**, опционально **context parallel** и **block-sparse / FlashAttention** пути.

В репозитории три логических слоя:

1. **Библиотека `arachne_x/`** — загрузка весов (`loader.py`), базовый и аватарный пайплайны, стриминговый движок, модули DiT/VAE/scheduler, аватарные блоки (аудио, anchors, losses), обучение/экспорт латентов (скрипты в `scripts/`, конфиги в `Demo/`).
2. **Оркестрация real-time продукта `src/server/`** — aiohttp-приложение (сессии, WebSocket, health, OpenAPI, webhooks, превью/бутстрап аватара, интеграция ASR/LLM/TTS по `config/pipeline_config*.json`).
3. **GPU sidecar `services/longcat-worker/`** — HTTP-обёртка над upstream LongCat-Video (`torchrun`), для сценариев, где генерация вынесена на отдельный GPU-сервис (`NULLXES_AVATAR_INFERENCE_URL`).

Публичный API пакета (по `arachne_x/__init__.py`): `WeightsLayout`, `load_base_pipeline`, `load_avatar_pipeline`, `get_vocal_separator_path`.

---

## 2. Веса и Hugging Face

| Источник | Назначение |
|----------|------------|
| [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) | Публикация **ARACHNE-X-ULTRA**: T2V, I2V, video-continuation, long video, интерактив; карточка ссылается на техотчёт и upstream-репо для установки PyTorch/flash-attn и запуска демо. |
| Локальный каталог после `snapshot_download` / `huggingface-cli download` | Ожидаемый layout: как минимум `tokenizer/`, `vae/` и остальные подпапки согласно `WeightsLayout` / `loader.py` (см. `arachne_x/weights_resolve.py` — эвристика по `tokenizer` + `vae`). |

В корне репозитория могут присутствовать **только метаданные** чекпоинта (например `dit/diffusion_pytorch_model.safetensors.index.json`, `dit/config.json`, `scheduler/`, `vae/config.json`) — **без** многогигабайтных `.safetensors` шардов. Для инференса нужен полный снапшот весов на диске или разрешённая загрузка с Hub (`weights_resolve.resolve_weights_root`, флаг `allow_hub`).

---

## 3. Архитектура по каталогам

### 3.1 `arachne_x/` — ядро модели и пайплайнов

| Область | Файлы / папки | Роль |
|---------|----------------|------|
| Загрузка | `loader.py`, `weights_resolve.py` | Единая точка загрузки компонентов для инференса; опционально HF Hub → локальный root. |
| Пайплайны | `pipeline_arachne_x_video.py`, `pipeline_arachne_x_video_avatar.py` | Базовое видео (T2V/I2V/VC) и аватар (ai2v/at2v/avc, streaming). |
| Стриминг | `streaming_inference.py`, `config_realtime.py` | KV-cache, буферы, real-time параметры. |
| Базовый DiT | `modules/longcat_video_dit.py`, `attention.py`, `blocks.py`, `rope_3d.py` | Трансформер 3D + внимание. |
| Аватар DiT | `modules/avatar/*`, `modules/avatar_losses.py` | Условие по аудио, anchors, обучающие/регуляризационные потери. |
| VAE / scheduler | `modules/autoencoder_kl_wan.py`, `modules/scheduling_flow_match_euler_discrete.py` | Декод латентов, шаги диффузии. |
| Аудио | `audio_process/*` | Мультипоток (lip / prosody / head), wav2vec, фонемы. |
| Параллелизм | `context_parallel/*`, `block_sparse_attention/*` | Ulysses-style CP, sparse attention интерфейсы. |
| Речь / TTS (интеграция) | `speech/*`, `tts/*` | Провайдеры и real-time обвязка для оркестратора. |
| Обучение | `training_*.py`, скрипты в `scripts/train.py`, `train_lora_avatar.py` | Обучение DiT / LoRA на предаугментированных латентах. |

### 3.2 `src/server/` — production-скелет медиа-слоя

- `webrtc_server.py` — маршруты: health, OpenAPI, webhooks, session start/stop/status, media patch, realtime token, chat, avatar preview/bootstrap, WebSocket.
- Вспомогательные модули: `session_manager.py`, `media_layer.py`, `realtime_api.py`, `realtime_avatar_loop.py`, ASR/TTS runners, webhook security.

Конфиг верхнего уровня: `config/pipeline_config.defaults.json` (пример: Faster-Whisper, Qwen2.5-7B, Edge TTS, аватар через `longcat_worker_http` и переменные `NULLXES_AVATAR_*`).

### 3.3 `services/longcat-worker/` — HTTP GPU worker

- FastAPI + вызов `torchrun` для задач `text-to-video` | `image-to-video` | `video-continuation`.
- Переменные: `LONGCAT_VIDEO_REPO`, `LONGCAT_CHECKPOINT_DIR`, опционально mock MP4 для dev без GPU.

### 3.4 Прочее

| Путь | Назначение |
|------|------------|
| `scripts/` | CLI: `infer.py`, `train.py`, экспорт латентов, `run_webrtc_server.py`. |
| `Demo/` | Streamlit и run_demo_* для T2V/I2V/VC/аватар/long/interactive. |
| `tests/` | Юнит-тесты (конфиг, webhook, realtime API, LoRA roundtrip). |
| `Documentation/` | Контракты SaaS, деплой, API, чеклисты ULTRA/avatar, enterprise RAG/digital employee. |
| `assets/avatar/` | Примеры JSON-манифестов сцен аватара. |

---

## 4. Спецификация продукта (целевые показатели, NULLXES README)

Значения ниже — **как заявлено в проектной документации** (`README.md`); фактическая величина на конкретном железе зависит от сборки и чекпоинта.

| Категория | Параметр |
|-----------|-----------|
| Инференс (цель) | ~30 FPS streaming |
| Задержка кадра (цель) | порядка **< 33 ms** на кадр |
| Lip-sync | заявлено **> 95%** |
| Идентичность лица | заявлено **> 0.92** cosine (ArcFace-стиль метрики) |
| VRAM (цель) | **~110–120 GB** на инференс (класс H200/H100/A100) |
| Базовая модель | **~13.6B** параметров DiT |
| Модальности | аудио, текст, изображение, видео |
| Параллелизм | Context Parallel (Ulysses) |

**Мультипоток аудио (концепт):** отдельные полосы для артикуляции губ (~18–24 Hz), просодии/эмоции (~4–6 Hz), движения головы (~1–2 Hz).

---

## 5. Связь с ARACHNE-X-ULTRA (внешняя карточка)

По [карточке HF ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO):

- Единая архитектура под **T2V / I2V / video-continuation**, long-form и интерактивные режимы.
- Отчёт: **LongCat-Video / ARACHNE-X-ULTRA Technical Report**, arXiv **2510.22200** (как указано на странице модели).
- Лицензия весов на карточке: **MIT** (с оговорками по товарным знакам/патентам — см. карточку и LICENSE в upstream).

Этот git-репозиторий — **код и оркестрация**; **канонический путь получения весов** для ULTRA-линейки — загрузка с Hub (или зеркало), не коммит шардов в git.

---

## 6. Ограничения данного документа

- Не выполнялись `python`, `pip`, `torchrun`, не проверялись хэши весов и не валидировался HF snapshot.
- Устаревшие фразы в старых внутренних MD (например «нет HTTP-сервера в репо») могут расходиться с текущим `src/server/` — приоритет у **фактического дерева** `src/server/*.py`.

---

## 7. GTM architecture notes (2026)

- [GTM_FOLDER_AUDIT.md](GTM_FOLDER_AUDIT.md) — keep / refactor / archive по каталогам.
- [GTM_VAE_ABI.md](GTM_VAE_ABI.md) — контракт латентного пространства VAE.
- [GTM_VAE_STRATEGY.md](GTM_VAE_STRATEGY.md) — совместимая замена VAE vs breaking tokenizer.
- [GTM_DATA_EVAL.md](GTM_DATA_EVAL.md) — данные и eval gates.
- Программный инференс без дублирования логики CLI: `arachne_x.runtime` (`execute_infer`, `InferenceEngine`), обёртка — `scripts/infer.py`.

---

**NULLXES LLC** · внутренний обзор для инженеров и партнёров · может дополняться ссылками на `FULL_SYSTEM_OVERVIEW.md` и `ARACHNE_X_API_DOCUMENTATION.md`.
