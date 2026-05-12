# ARACHNE-X-ULTRA V2 — «Ночная Фурия» — production contract (NULLXES)

Документ фиксирует **единый продуктовый контур** инференса и состав обязательных компонентов. Целевое позиционирование V2: **stateful realtime avatar system** (сессионное состояние, непрерывный поток аудио→эмбеддинг→кадр), поверх существующего **inference pipeline**. Архитектура модели **не меняется**; веса — **proprietary**, **independently trained**, **production checkpoints**.

Полная матрица готовности RunPod / стек / чек-лист: [`GTM_ULTRA_V2_NIGHT_FURY.md`](GTM_ULTRA_V2_NIGHT_FURY.md).

---

## 1. Публикация весов (канон)

| Линейка | Hugging Face (официальная публикация NULLXES) |
|---------|-----------------------------------------------|
| VIDEO (T2V / I2V / continuation и смежные режимы в карточке) | [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) |
| AVATAR (говорящий аватар, аудио-кондиционирование) | [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) |

Локальный layout после загрузки снимка — по `WeightsLayout` / `arachne_x/loader.py` / `arachne_x/weights_resolve.py`.

---

## 2. Единый канонический контур инференса

| Слой | Роль | Статус |
|------|------|--------|
| **`arachne_x.runtime`** | Программный API: `InferenceEngine`, `execute_infer` — тот же контракт, что тонкий CLI `scripts/infer.py`. Офлайн/пакетный сценарий, интеграции, тесты. | **Production (library)** |
| **`scripts/infer.py`** | Только `argparse` + вызов `execute_infer`. | **Production (thin wrapper)** |
| **HTTP GPU worker** — реализация в репозитории: каталог `services/longcat-worker/` (имя каталога — **legacy internal identifier**, внешне — **NULLXES Inference Worker**) | FastAPI: NDJSON-стрим (`/v1/realtime/avatar_frames`), MP4 по job API; **in-process** `load_avatar_pipeline` и вызовы методов пайплайна (`gpu_avatar_runtime.py`). Поведение должно оставаться **семантически согласованным** с `arachne_x.runtime` (те же чекпоинты и режимы); дедупликация кода — отдельный refactor-тикет. | **Production (serving)** |
| **`src/server/`** | Вызовы HTTP к воркеру по конфигу (`NULLXES_AVATAR_*` и др.); сессии и маршрутизация продукта. **Не** второй экземпляр загрузки DiT на GPU в целевой топологии «один inference-процесс на GPU». | **Internal / orchestration** (не дублировать ядро инференса) |

**Решение:** канон **исполнения** нейросети на GPU в проде — **Inference Worker** (`services/longcat-worker/`). Канон **программного** контракта пакета — **`arachne_x.runtime`**. **`src/server/`** — **internal** относительно владения весами DiT/VAE на GPU.

---

## 3. Обязательные компоненты чекпоинта (минимальный состав)

Для **VIDEO** и **AVATAR** набор поддиректорий задаётся загрузчиком; минимум для работы пайплайнов:

| Компонент | Назначение |
|-----------|------------|
| **Tokenizer** (текст) | Условие по тексту для DiT. |
| **VAE** | Латентное пространство видео; ABI см. `GTM_VAE_ABI.md`. |
| **DiT** (базовый и/или аватарный вариант в layout) | Диффузия в латентном пространстве. |
| **Scheduler** | Шаги сэмплинга (flow-match и т.д. в репозитории). |
| **Аудио-ветка (аватар)** | Wav2Vec2-совместимый энкодер в layout `audio/`; цепочка **librosa → Wav2Vec2 → embedding**; при `speak_text`: **TTS → WAV → та же цепочка** (не фейк; деталь и риски — `GTM_ULTRA_V2_NIGHT_FURY.md` §4). |

Допустимые упрощения в рантайме: fallback на single-stream wav2vec; отключение optional модулей при ошибках (логирование, без подмены нулями «тихого» conditioning).

---

## 4. Аудио (фиксация для GTM)

- Вход: моно **16 kHz** float (или PCM16 в HTTP контракте воркера).
- **Реальный** пайплайн: загрузка/ресэмплинг → **Wav2Vec2** (веса в чекпоинте) → тензор для DiT.
- **TTS перед эмбеддингом:** синтез в WAV, затем **тот же** энкодер.
- Не является симуляцией: опциональные ветки (fusion, phoneme) могут отключаться — остаётся wav2vec-поток.

---

## 5. Typed inference API (спецификация без реализации кода)

Единый контракт для сервисов и обёрток (поля — логические; реализация в Python позже в `arachne_x.runtime` или отдельном `arachne_x.contracts` по решению команды):

**`InferenceJob`**

- `mode`: enum — `t2v` | `i2v` | `vc` | `ai2v` | `at2v` | `avc` | `streaming_ai2v` | `enroll_identity` (зеркало CLI).
- `checkpoint_dir`: корень весов (VIDEO или AVATAR layout).
- `prompt`, `negative_prompt`: строки.
- `image_path` | `video_path` | `audio_path`: опциональные пути к файлам.
- `speak_text` + блок **`TtsSpec`** (provider, model_id, device, язык, спикер, параметры провайдера): если задан — приоритет ниже явного `audio_path` по правилам текущего `resolve_avatar_wav_path`.
- `resolution`, `num_frames`, `num_inference_steps`, guidance scales.
- `output_path` / sink для стрима.
- `identity_*` / `emotion_*` / `lora_*`: как в текущем CLI.

**`InferenceResult`**

- Для файловых режимов: путь(и) к артефакту, exit metadata.
- Для стрима: итератор кадров или делегирование воркеру по HTTP.

Сервисный слой воркера уже отражает подмножество (JSON body → вызов пайплайна); со временем **свести** поля HTTP к полям `InferenceJob` (один schema truth).

---

## 6. Train engine (структура каталогов, без кода)

Цель: вынести логику из `scripts/train.py` и `scripts/train_lora_avatar.py` в библиотеку.

Предлагаемая структура:

```
arachne_x/
  train/
    __init__.py          # публичные фасады
    engine.py            # общий цикл: dataloader → step → ckpt
    specs.py             # dataclass TrainingJobSpec, LoRAJobSpec
    latent_dataset.py    # обёртка над WDS / validate_latent_sample
```

`scripts/train.py` → **thin**: парсинг → `TrainingJobSpec` → `train.engine.run()`.  
Тот же паттерн для LoRA-аватара.

---

## 7. Docker-образы (три роли, без команд)

| Образ | Содержимое | Назначение |
|-------|------------|------------|
| **`arachne-x-train`** | PyTorch + зависимости обучения, инструменты экспорта латентов, **без** обязательного GPU в сборке образа (GPU на рантайме job). | Обучение / LoRA / упаковка шардов. |
| **`arachne-x-inference`** | PyTorch CUDA, `arachne_x`, веса монтируются томом; entrypoint — worker или `python -m` runtime. | Пакетный инференс, CI eval, батч. |
| **`arachne-x-realtime`** | То же + минимальный HTTP сервер воркера (Inference Worker). | Низкая задержка, один процесс на GPU. |

Версии CUDA / драйвер / **Triton** (если используется BSA-путь) — одна матрица совместимости на команду; фиксируется в `docker/` README или отдельном `GTM_RUNTIME_MATRIX.md` при появлении.

---

## 8. Связанные GTM-документы

- `GTM_NIGHT_FURY_V2_DIRECTIVE.md` — **directive V2**: матрица слоёв, MVP single-Pod, employee packs, контур инференса.
- `GTM_ULTRA_V2_NIGHT_FURY.md` — V2 RunPod, стек, дубли, чек-лист.
- `GTM_FOLDER_AUDIT.md` — keep / refactor / archive.
- `GTM_VAE_ABI.md` — латентный ABI.
- `GTM_VAE_STRATEGY.md` — смена VAE внутри NULLXES.
- `GTM_DATA_EVAL.md` — метрики и CI gates.
- `ARACHNE-X_ARCHITECTURE_SPEC_NULLXES.md` — обзор дерева.

---

**NULLXES** · ARACHNE-X-ULTRA V2 «Ночная Фурия» · внутренний GTM-контракт · не изменяет исходный код без отдельного запроса.
