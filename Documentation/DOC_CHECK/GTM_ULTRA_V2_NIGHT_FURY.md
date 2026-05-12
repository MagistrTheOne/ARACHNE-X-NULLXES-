# ARACHNE-X-ULTRA V2 — «Ночная Фурия»

**NULLXES · production readiness · GPU-first (RunPod)**

| | |
|--|--|
| **Позиционирование** | Переход от *stateless inference pipeline* к **stateful realtime avatar system**: сессия, непрерывный поток аудио→латент→кадр, сохраняемый контекст (идентичность, буферы, KV/streaming state по спецификации продукта). |
| **Веса** | **Proprietary**, **independently trained**, **production checkpoints** — только NULLXES. |
| **Инфра** | **RunPod**, целевые GPU **H100 / H200 / B200**. CPU — не целевой путь исполнения DiT/VAE. |
| **Архитектура модели** | **Без изменений:** DiT, VAE, аудио-цепочка до кондиционирования — зафиксированы. |

**Публикация весов**

- VIDEO: [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO)
- AVATAR: [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR)

---

## 1. Связь с базовым GTM

- Полный **directive V2** (матрица слоёв, MVP Pod, employee packs): [`GTM_NIGHT_FURY_V2_DIRECTIVE.md`](GTM_NIGHT_FURY_V2_DIRECTIVE.md)
- Контракт инференса и слои: [`GTM_PRODUCTION_CONTRACT.md`](GTM_PRODUCTION_CONTRACT.md)
- Латентный ABI: [`GTM_VAE_ABI.md`](GTM_VAE_ABI.md)
- Eval / CI gates: [`GTM_DATA_EVAL.md`](GTM_DATA_EVAL.md)
- Дерево репозитория: [`ARACHNE-X_ARCHITECTURE_SPEC_NULLXES.md`](ARACHNE-X_ARCHITECTURE_SPEC_NULLXES.md)

V2 добавляет **семантику продукта** (stateful realtime) и **матрицу готовности RunPod**; не заменяет ABI и веса.

---

## 2. Аудит Python-стека (репозиторий, без установочных команд)

### 2.1 Зафиксированные файлы зависимостей

| Файл | Назначение |
|------|------------|
| [`requirements.txt`](../../requirements.txt) | Ядро: torch 2.6.0, torchvision 0.21.0, transformers 4.41.0, diffusers 0.35.1, flash-attn (Linux), **без** закреплённого `xformers` |
| [`requirements_avatar.txt`](../../requirements_avatar.txt) | Надстройка: `-r requirements.txt` + onnx/onnxruntime, `librosa==0.11.0`, `av==13.1.0`, VAD/WebRTC-стек и др. |
| [`requirements-tts.txt`](../../requirements-tts.txt) | Опциональный TTS для CLI — не ядро DiT |
| [`docker/Dockerfile.gpu`](../../docker/Dockerfile.gpu) | База: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| [`services/longcat-worker/requirements.txt`](../../services/longcat-worker/requirements.txt) | Только FastAPI/uvicorn/pydantic |

### 2.2 Импорты `arachne_x/` (критичные цепочки)

| Зависимость | Где тянется | Заметка V2 |
|-------------|-------------|------------|
| **torch / torchvision** | Везде | Канон: **CUDA-сборка**, совпадение с базовым образом RunPod. |
| **triton** | `block_sparse_attention/*` → импорт из цепочки **Attention** → `flash_attn_bsa_3d` | На текущем графе импортов **загрузка модуля attention тянет triton**; без совместимого triton+cuda **импорт пакета падает**, даже если BSA выключен в конфиге. Для V2: **жёстко учитывать в образе** или планировать lazy-import (отдельный RFC кода). |
| **flash-attn** | `attention.py` (ветки FA2/FA3) | В `requirements.txt` только под Linux; версия должна соответствовать **wheel под CUDA образа**. |
| **xformers** | Опциональные ветки `enable_xformers` в attention | В корневом `requirements.txt` **нет**; включение флага без пакета → **RuntimeError** на ветке. Для V2: либо **пинить** xformers под torch+CUDA, либо **запретить** флаг в проде до явной установки. |
| **transformers** | UMT5, Wav2Vec2*, tokenizer | Пин 4.41.0; обновление только с **регрессионным** прогоном загрузки чекпоинта. |
| **diffusers** | ConfigMixin, ModelMixin, VideoProcessor, scheduler utils | 0.35.1 связан с API классов в репо; апгрейд только с полным интеграционным тестом. |
| **safetensors** | Загрузка весов DiT | Критичный пин; не разъезжать с `torch.load` путями. |
| **einops** | Пайплайны, блоки | Низкий риск; держать пин. |
| **librosa** | `inference_audio`, torch_utils, speech | **Конфликт:** `requirements.txt` — `librosa>=0.10.0`, `requirements_avatar.txt` — `librosa==0.11.0`. Сборка «ядро+аватар» должна разрешаться **одной** версией (рекомендация: **0.11.x** везде для стека аватара). |
| **av** | **Конфликт:** 12.0.0 (base) vs 13.1.0 (avatar) | Унифицировать мажор при объединённом образе. |
| **numpy** | 1.26.4 (base) | Согласовать с `scipy`, `opencv`, onnxruntime при обновлениях. |

### 2.3 Устаревшие или рискованные позиции

| Компонент | Риск |
|-----------|------|
| **onnxruntime==1.16.3** (avatar) | Старый релиз; возможен разрыв с **CUDA 12.4** образа и с runtime PyTorch. Для V2: выровнять **ORT GPU** под драйвер Pod или перейти на CPU-ORT только для разделения вокала, явно задокументировав устройство. |
| **transformers 4.41.0** | Отстаёт от линейки torch 2.6 по времени; риск только при смене API HF в будущем — пока **не трогать без нужды**. |
| **streamlit, edge-tts, faster-whisper** в `requirements.txt` | Раздувают **inference-only** образ; для чистого GPU-воркера V2 — вынос в **optional** слой образа (архитектурное разделение Dockerfile build-args уже частично есть). |

### 2.4 Потенциальные конфликты (merge зависимостей)

1. **librosa** / **soundfile** / **soxr** — разные пины между base и avatar.  
2. **av** — два мажора.  
3. **aiohttp** — совпадает по пину в avatar с base — ок.  
4. **Дубли faster-whisper** — `>=` в base и `==` в avatar: унифицировать одной строкой при сборке prod-образа.  

### 2.5 Рекомендуемая политика версий (GPU-first, без команд)

- **PyTorch + CUDA**: одна строка с **базовым образом RunPod** и `Dockerfile.gpu` (сейчас **CUDA 12.4** + **torch 2.6.0**). Любое смещение — только через таблицу совместимости **torch ↔ flash-attn ↔ (опц.) xformers ↔ triton**.  
- **B200 (Blackwell)**: образ **12.4** может быть **ниже** требований драйвера для полного набора фич нового поколения; для B200 заложить **отдельную** строку матрицы: **CUDA minor / PyTorch build** под RunPod template, **до** выката V2.  
- **Triton**: считать **обязательным артефактом окружения** при текущем импорт-графе BSA-модуля.  
- **xformers**: либо полный отказ в прод-конфиге (только FA2/SDPA), либо пин под ту же связку, что torch.  
- **librosa / av**: выровнять к **avatar-стеку** как источнику истины для realtime.

### 2.6 Критично обновить / зафиксировать до V2

1. **Единый lock** зависимостей для образа **Inference Worker** (без дрейфа librosa/av).  
2. **Матрица CUDA** для H100 / H200 / B200 — таблица в **разделе 6** этого документа; заполняется после валидации на RunPod.  
3. **Triton + triton import graph** — зафиксировать как **блокер старта** или RFC на lazy-import.  
4. **xformers**: явная политика prod (вкл/выкл).  
5. **onnxruntime** для vocal separator — совместимость с образом.

---

## 3. GPU-совместимость и RunPod

### 3.1 Типы данных

- Загрузчик пайплайнов по умолчанию: **`torch.bfloat16`** на CUDA — согласовано с целевым H100/H200.  
- **Риск:** смешение `fp16` autocast в отдельных ветках (например streaming) с **bf16** весами — дать **единый dtype policy** документом конфигурации сервиса (без смены архитектуры).  

### 3.2 Память

- Декларируемый целевой VRAM под полный инференс **высокий** (см. README-цели); **H100 80 GB** — узкое место без **context parallel**, уменьшения разрешения/кадров или шардирования модели. **H200 141 GB** / **B200** — ближе к заявленным целям.  
- **VAE encode/decode** пики — отдельные всплески рядом с DiT; для realtime — буферизация и очередь чанков аудио.

### 3.3 Точки падения на H100/H200

| Точка | Причина |
|-------|---------|
| Импорт без **triton** | Ошибка при загрузке `attention` → BSA interface. |
| **flash-attn** wheel не под minor CUDA образа | Сборка образа падает или рантайм SIGILL/undefined symbol. |
| **OOM** | Разрешение 720p + длинное окно кадров на 80 GB без CP. |
| **bf16 на старых GPU** | Не применимо к H100+; на смешанных кластерах — явная политика. |

### 3.4 B200

- Риск: **базовый образ 12.4** vs требования **нового драйвера** для B200 в выбранном RunPod template — валидация **до** прод-лейбла V2.  
- Риск: **flash-attn / triton** ночные сборки — только проверенная тройка версий под шаблон Pod.

---

## 4. Аудио-пайплайн (подтверждение)

| Утверждение | Статус |
|--------------|--------|
| Не фейк | **Да:** реальный сигнал → **Wav2Vec2** (веса в чекпоинте) → **embedding → DiT**. |
| `speak_text` | **TTS → WAV → та же цепочка** — не заглушка. |
| Упрощения | Fallback **single-stream** wav2vec; optional ветки (fusion / phoneme) могут **отключаться** с логом — остаётся валидный wav2vec-поток. |

### 4.1 Слабые места качества (без изменения архитектуры)

- **Границы чанков** в streaming: фазовый сдвиг эмбеддинга между окнами.  
- **Ресэмплинг 16 kHz** — потери ВЧ при плохом исходном материале.  
- **Отключение multi-stream** — потеря разделения артикуляция / просодия / голова (если optional модули не активны).  
- **TTS артефакты** — входят в эмбеддинг как есть.  

### 4.2 Цель V2 (hint, без реализации)

- Явный **session state**: окно wav2vec, carry-over буфера, политика сброса.  
- **Стабилизация multi-stream** как prefer-path при доступности весов.  
- **Единый контракт** частоты/глубины PCM между клиентом и Inference Worker.

---

## 5. Дублирование логики (архитектурная карта)

| Зона | Дублирование | Решение V2 |
|------|----------------|------------|
| **`arachne_x.runtime`** vs **`gpu_avatar_runtime`** | Два входа в **одни и те же** методы пайплайна (разный glue-код). | **Оставить** оба слоя; пометить **конвергенцию** на общий фасад (`InferenceJob` / shared helper) как **обязательный** рефакторинг жизненного цикла V2 (код — отдельный тикет). |
| **`src/server/`** vs **Inference Worker** | Сервер **не** считает DiT локально при HTTP к воркеру — **не дубль GPU**. | **Оставить:** сервер = **internal orchestration**; воркер = **единственный** GPU-исполнитель. |
| **Пайплайн** vs **runtime** | Runtime вызывает пайплайн; дубля нет на уровне весов. | **Core не трогать.** |

**Legacy (git path / backward compatibility):** каталог воркера `services/longcat-worker/`; канонический HTTP MP4 — **`POST /v1/arachne/generate`**, legacy alias — `POST /v1/longcat/generate` (скрыт из OpenAPI schema). Внешнее имя: **NULLXES Inference Worker** (NIGHT FURY).

---

## 6. Матрица CUDA / образ (заполнение командой инфра)

Файл-заготовка: фиксировать после первого успешного прогона на каждом шаблоне RunPod.

| GPU | RunPod template (id) | CUDA (driver) | PyTorch | flash-attn | triton | Статус V2 |
|-----|----------------------|-----------------|---------|------------|--------|-----------|
| H100 | | 12.4 / … | 2.6.0 | | | |
| H200 | | | | | | |
| B200 | | | | | | |

---

## 7. Чек-лист V2 READY («Ночная Фурия»)

### Runtime

- [ ] `arachne_x.runtime` — единственный программный контракт для offline/batch сценариев.  
- [ ] `InferenceJob`-спека из `GTM_PRODUCTION_CONTRACT.md` согласована с полями HTTP воркера (таблица соответствия).  
- [ ] Политика **stateful session** (что хранится между чанками аудио) задокументирована продуктом и согласована с `streaming_inference` / worker.  

### Weights

- [ ] VIDEO и AVATAR снимки с Hub — полный layout, проверка `loader` на обоих корнях.  
- [ ] Версия чекпоинта записана в **eval baseline** (`GTM_DATA_EVAL.md`).  

### Audio

- [ ] Контракт **16 kHz mono** end-to-end (клиент → воркер → librosa/wav2vec).  
- [ ] Логирование отключения optional веток включено на **warning** в проде V2 (политика команды).  

### GPU

- [ ] Образ: **CUDA + torch + flash-attn + triton** согласованы.  
- [ ] Единый **dtype policy** (bf16) на пути инференса.  
- [ ] Матрица **H100 / H200 / B200** заполнена и подписана.  
- [ ] Нагрузочный тест OOM-границы для целевого `resolution` × `num_frames`.  

### API

- [ ] Только **Inference Worker** на GPU в целевой топологии; оркестратор без локального DiT.  
- [ ] Аутентификация ключом воркера зафиксирована в RunPod secrets.  

### Eval

- [ ] Gates **E-*** из `GTM_DATA_EVAL.md` подключены к CI на GPU runner (или ночной job на Pod).  
- [ ] `eval_manifest.json` покрывает realtime-режим (короткий стресс-стрим).  

### Infra

- [ ] Dockerfile / compose: **один** lock-файл или единый слой pip для worker+avatar.  
- [ ] Разделение образов: **train / inference / realtime** по `GTM_PRODUCTION_CONTRACT.md` §7.  
- [ ] Документированный **rollback** чекпоинта при fail gate.  

---

**NULLXES · ARACHNE-X-ULTRA V2 «Ночная Фурия» · stateful realtime system**
