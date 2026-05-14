# NULLXES Inference Worker (NIGHT FURY)

Production **GPU avatar** HTTP service for **RunPod Linux** (H200 primary, H100 secondary, B200 experimental). Реализация: **in-process** CUDA — FastAPI вызывает общий runtime [`arachne_x/runtime/avatar_serving.py`](../../arachne_x/runtime/avatar_serving.py) через тонкий lazy-прокси [`gpu_avatar_runtime.py`](gpu_avatar_runtime.py) (uvicorn не импортирует тяжёлый стек до первого inference).

## Endpoints

| Method | Path | Описание |
|--------|------|----------|
| GET | `/health` | Liveness; **не** грузит веса на GPU. |
| POST | `/v1/realtime/avatar_frames` | NDJSON stream (`application/x-ndjson`), RGB frames. JSON body: `engine` — `arachne` (core), `arachne_ultra_avatar` / `arachne_ultra_video` (aliases to core, NULLXES HR AI), `nullxes` / `longcat` / `core` / `""`. |
| POST | `/v1/arachne/generate` | Синхронный MP4 (`video/mp4`) для поддерживаемых audio-* задач. |
| POST | `/v1/longcat/generate` | **Legacy alias** того же handler (не в OpenAPI schema). |
| POST | `/v1/infer/jobs` | Async очередь MP4 → poll status → one-shot result. |

## Канонические модули DiT (библиотека `arachne_x`)

Инференс грузит веса через `arachne_x.loader` из **`arachne_x/modules/arachne_video_dit.py`** (базовое видео) и **`arachne_x/modules/avatar/arachne_avatar_dit.py`** (аватар). Файлы `longcat_video_dit*.py` — **thin shim** для обратной совместимости импортов; публичные имена классов (`LongCatVideoTransformer3DModel`, …) не менялись (ABI чекпоинтов).

Скрипт **`longcat_generate_once.py`** в этом каталоге — **deprecated**: ориентирован на внешний пакет `longcat_video.*`, не на `arachne_x`; **не** использовать в RunPod / `GTM_ONE_SHOT_DEPLOY` (см. docstring в файле).

## Обязательные переменные (prod)

| Переменная | Назначение |
|------------|------------|
| `NULLXES_CHECKPOINT_DIR` или `ARACHNE_CHECKPOINT_DIR` | Каталог весов avatar (tokenizer, vae, dit, avatar_single, audio и т.д.). |

## Аутентификация (опционально)

Если задан один из ключей, клиент обязан передать заголовок **`X-NULLXES-Avatar-Inference-Key`**:

1. `NULLXES_INFERENCE_SERVICE_KEY` (канон)
2. `NULLXES_AVATAR_INFERENCE_SERVICE_KEY` (совместимость с aiohttp-клиентом в `src/server`)
3. `LONGCAT_INFERENCE_SERVICE_KEY` (legacy имя env)

## Зависимости HTTP-слоя

```bash
cd services/longcat-worker
pip install -r requirements.txt
```

Полный torch/flash-attn/ARACHNE-X stack — из корня репозитория: `requirements.txt`, `requirements_avatar.txt` (см. [`docker/Dockerfile.gpu`](../../docker/Dockerfile.gpu)).

## Запуск

Из корня репозитория (чтобы резолвились `arachne_x` и воркер):

```bash
export NULLXES_CHECKPOINT_DIR=/path/to/avatar-weights
export PYTHONPATH=/path/to/ARACHNE-X:/path/to/ARACHNE-X/services/longcat-worker
cd services/longcat-worker
uvicorn main:app --host 0.0.0.0 --port 9090
```

Оркестратор: `NULLXES_AVATAR_INFERENCE_URL=http://<host>:9090`, опционально `NULLXES_AVATAR_INFERENCE_PATH=/v1/arachne/generate` (по умолчанию в клиенте уже `/v1/arachne/generate`).

## Поле `engine` (NDJSON)

Канонические core-значения: `arachne` (default в теле запроса), `nullxes`, пустая строка, legacy `longcat`, `core`. Значение `wan_s2v` отклоняется с понятной ошибкой, если не развёрнут отдельный сервис.

## Очередь jobs

`POST /v1/infer/jobs` → `GET /v1/infer/jobs/{jobId}` → один раз `GET /v1/infer/jobs/{jobId}/result` (MP4). Глубина: `INFERENCE_MAX_QUEUE` (по умолчанию 32).

## Dev / mock (только с явным флагом)

Исторические переменные вида `LONGCAT_MOCK_MP4_PATH` не должны использоваться в prod без **`ALLOW_INFERENCE_DEV_MOCK=1`** (см. код воркера / политику в `GTM_PRE_RELEASE_AUDIT.md`).

## Дополнительно

- One-shot RunPod: [`Documentation/DOC_CHECK/GTM_ONE_SHOT_DEPLOY.md`](../../Documentation/DOC_CHECK/GTM_ONE_SHOT_DEPLOY.md)
- Контракт HTTP ↔ job: [`Documentation/DOC_CHECK/GTM_SCHEMA_TRUTH_INFERENCE_HTTP.md`](../../Documentation/DOC_CHECK/GTM_SCHEMA_TRUTH_INFERENCE_HTTP.md)
