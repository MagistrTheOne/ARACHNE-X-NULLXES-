# ARACHNE-X — что есть в полной системе

**Дата:** 20.02.2026  
**Версия:** 1.0

---

## 1. Назначение системы

- **Гиперреалистичные AI-аватары** (talking head) в реальном времени.
- **Базовое видео:** Text-to-Video, Image-to-Video, Video Continuation.
- **Аватары:** Audio-to-Video (один/несколько человек), стриминг по аудио.
- **Цель по железу:** NVIDIA H200/H100/A100, 30 FPS, ~110–120 GB VRAM на инференс.

---

## 2. Состав по папкам и файлам

### 2.1 Ядро: `arachne_x/`

| Путь | Назначение |
|------|------------|
| **`__init__.py`** | Публичный API: `WeightsLayout`, `load_base_pipeline`, `load_avatar_pipeline`, `get_vocal_separator_path` |
| **`loader.py`** | Единственная точка загрузки весов для инференса: tokenizer, text_encoder, vae, scheduler, dit, audio (avatar). |
| **`pipeline_arachne_x_video.py`** | Базовый пайплайн: T2V, I2V, VC. |
| **`pipeline_arachne_x_video_avatar.py`** | Аватар-пайплайн: ai2v, at2v, avc, **generate_streaming_ai2v**, identity/emotion/phoneme, hybrid mouth, temporal memory. |
| **`streaming_inference.py`** | Движок стриминга: KV-cache, circular buffer, optical flow, distilled scheduler. |
| **`config_realtime.py`** | Конфиг real-time (буферы, FPS, память). |
| **`model_adapter.py`** | Адаптер для подключения внешних/других чекпоинтов к пайплайну. |

**Модели (`arachne_x/modules/`):**

| Файл | Роль |
|------|------|
| `longcat_video_dit.py` | Базовый DiT (LongCatVideoTransformer3DModel). |
| `autoencoder_kl_wan.py` | VAE для видео (Wan-style). |
| `scheduling_flow_match_euler_discrete.py` | Scheduler (Flow Match, Euler). |
| `attention.py`, `blocks.py`, `rope_3d.py` | Блоки и внимание базового DiT. |
| **`avatar/`** | |
| `longcat_video_dit_avatar.py` | Аватар-DiT (условие по аудио, identity и т.д.). |
| `blocks.py`, `attention.py`, `rope_3d.py` | Блоки аватар-модели. |
| `facial_anchors.py` | 68-point anchoring, landmark constrainer. |
| `avatar_losses.py` | Потери: lip-sync, identity, temporal, expression, perceptual. |
| `lora_utils.py` | LoRA для дообучения. |

**Аудио (`arachne_x/audio_process/`):**

| Файл | Роль |
|------|------|
| `multi_stream_processor.py` | Три потока: lip-sync, prosody, head motion → fusion 1024-dim. |
| `wav2vec2.py` | Обёртка Wav2Vec2 для аудио-эмбеддингов. |
| `phoneme_aligner.py` | Фонемная привязка для губ. |
| `torch_utils.py` | Утилиты (в т.ч. `save_video_ffmpeg`). |

**Оптимизация и утилиты:**

| Путь | Роль |
|------|------|
| `context_parallel/context_parallel_util.py` | Context parallel (Ulysses), разбиение по GPU. |
| `context_parallel/ulysses_wrapper.py` | Обёртка Ulysses. |
| `block_sparse_attention/*` | Block-sparse attention + FlashAttention. |
| `utils/prompt_enhancer.py` | Улучшение промптов. |
| `utils/monitoring.py` | Логирование и мониторинг. |
| `utils/bukcet_config.py` | Конфиг бакетов (разрешение/длина). |

---

### 2.2 Точки входа: `scripts/`

| Файл | Режим | Описание |
|------|--------|----------|
| **`infer.py`** | CLI инференс | Режимы: `t2v`, `i2v`, `vc`, `ai2v`, `at2v`, `avc`, `streaming_ai2v`. Загружает пайплайн через `loader`, генерирует видео в файл. |
| **`train.py`** | CLI обучение | Режимы: `base`, `avatar`. Загружает только DiT из `checkpoint_dir` (subfolder `dit` / `avatar_single`), обучает на предаугментированных латентах (`.pt`/`.npz`), сохраняет чекпоинты. |

---

### 2.3 Демо и конфиг обучения: `Demo/`

| Файл | Назначение |
|------|------------|
| **`run_streamlit.py`** | Streamlit UI: T2V, I2V, VC (базовый пайплайн + LoRA). |
| **`run_demo_streaming_realtime.py`** | Демо real-time стриминга аватара (image + audio → поток кадров). |
| **`run_demo_avatar_single_audio_to_video.py`** | Один аватар, один аудио → видео. |
| **`run_demo_avatar_multi_audio_to_video.py`** | Несколько персон/аудио → одно видео. |
| **`run_demo_text_to_video.py`** | T2V демо. |
| **`run_demo_image_to_video.py`** | I2V демо. |
| **`run_demo_video_continuation.py`** | VC демо. |
| **`run_demo_long_video.py`** | Длинное видео. |
| **`run_demo_interactive_video.py`** | Интерактивное видео. |
| **`training_config_h200.py`** | Конфиг обучения: `H200TrainingConfig`, профили (SINGLE/DUAL/POD/MEGA), LoRA, данные. |

---

### 2.4 Бенчмарк и зависимости

| Файл | Назначение |
|------|------------|
| **`benchmark_realtime.py`** | Замер FPS, latency, памяти для аватар-пайплайна на H200. |
| **`requirements.txt`** | Основные зависимости (PyTorch, diffusers, transformers, streamlit и т.д.). |
| **`requirements_avatar.txt`** | Доп. зависимости для аватаров. |

---

### 2.5 Документация: `Documentation/`

| Файл | Содержание |
|------|------------|
| **`ARACHNE-X_IMPLEMENTATION_SUMMARY.md`** | Facial anchors, multi-stream audio, avatar losses, streaming engine, H200 training, метрики. |
| **`ARACHNE-X-ULTRA_NULLXES_12-02-2026_CHECKLIST.md`** | Чеклист ULTRA: архитектура, память, identity, phoneme, emotion, hybrid mouth, качество, production. |
| **`AVATAR_UPGRADE_CHECKLIST.md`** | Чеклист доработок аватара. |

---

### 2.6 Ресурсы и прочее

| Путь | Содержание |
|------|------------|
| **`assets/avatar/`** | Примеры конфигов: `single_example_1.json`, `multi_example_1.json`, `multi_example_2.json` (prompt, image, audio, bbox). |
| **`README.md`** | Обзор, возможности, бенчмарки, системные требования, Quick Start. |
| **`ARACHNE_X_API_DOCUMENTATION.md`** | API для деплоя: загрузка, методы генерации, REST/WebSocket примеры. |
| **`PARTNER_COST_ESTIMATE.md`** | Смета для партнёра (JobAI): GPU, пакеты, договор. |
| **`LICENSE`** | Лицензия. |

---

## 3. Потоки данных (кратко)

### Инференс

1. **Загрузка весов:** только в **`arachne_x/loader.py`** (`load_base_pipeline` / `load_avatar_pipeline`).
2. **Запрос:** через `scripts/infer.py` или напрямую Python API (`pipe.generate_*`).
3. **Аватар стриминг:** `pipe.generate_streaming_ai2v(image, prompt, audio_stream, ...)` → итерация по кадрам.

### Обучение

1. **Полное дообучение DiT:** **`scripts/train.py`** — `--mode base|avatar`, `from_pretrained(..., subfolder="dit"|"avatar_single")`.
2. **LoRA только на аватарном DiT:** **`scripts/train_lora_avatar.py`** — базовые веса заморожены, учатся адаптеры в формате `arachne_x/modules/lora_utils.py` (ключи `lora___lorahyphen___...`), выход — `lora_final.safetensors` и `lora_train_meta.json`. Данные те же, что для `--mode avatar` (латенты + `audio_embs`). Смоки: `python scripts/verify_lora_avatar.py` (игрушечная модель; опционально `--checkpoint_dir` с `avatar_single/`).
3. **Данные:** предаугментированные латенты в `dataset_dir` (`.pt`/`.npz`: latents, prompt_embeds, prompt_mask, timesteps, noise, для avatar — audio_embs).
4. **Конфиг:** `Demo/training_config_h200.py` (H200TrainingConfig; какие поля реально читают `train.py` / `train_lora_avatar.py`, см. docstring класса в файле).
5. **Регрессия LoRA без полного импорта:** `python -m unittest tests.test_lora_init_roundtrip -v`.
6. **Экспорт латентов из сырья** — не реализован; черновик требований: `Documentation/LATENT_TRAINING_EXPORT.md`.

---

## 4. Что в итоге есть в полной системе

| Категория | Что есть |
|-----------|----------|
| **Загрузка весов (инференс)** | `arachne_x/loader.py` — единственная точка. |
| **Загрузка весов (обучение)** | `scripts/train.py` — полный DiT; `scripts/train_lora_avatar.py` — LoRA поверх `avatar_single`. |
| **Базовое видео** | T2V, I2V, VC: пайплайн в `pipeline_arachne_x_video.py`, вызов из `infer.py` и Streamlit. |
| **Аватары** | Single/Multi, ai2v/at2v/avc, стриминг: `pipeline_arachne_x_video_avatar.py`, демо в `Demo/`. |
| **Real-time стриминг** | `generate_streaming_ai2v` + движок в `streaming_inference.py`, конфиг в `config_realtime.py`. |
| **Аудио** | Multi-stream (lip/prosody/head), Wav2Vec2, фонемы — в `audio_process/` и пайплайне. |
| **Лицо и идентичность** | Facial anchors, identity bank, emotion, hybrid mouth — в пайплайне и `modules/`. |
| **Обучение** | `scripts/train.py` (full), `scripts/train_lora_avatar.py` (LoRA), конфиг H200 в `Demo/training_config_h200.py`, датасет — предаугментированные латенты (вне репо). |
| **Деплой** | API описан в `ARACHNE_X_API_DOCUMENTATION.md`; готового HTTP-сервера в репо нет — только примеры. |
| **Документация** | README, Implementation Summary, чеклисты ULTRA/AVATAR, смета партнёра. |

Итого: в полной системе есть **инференс (включая стриминг)**, **обучение по предаугментированным латентам**, **все модели и пайплайны**, **демо и бенчмарки**, **конфиги и доки**. Нет в репо: сырой пайплайн подготовки латентов из видео/аудио и готового production HTTP/WebSocket сервера — их нужно поднимать отдельно.
