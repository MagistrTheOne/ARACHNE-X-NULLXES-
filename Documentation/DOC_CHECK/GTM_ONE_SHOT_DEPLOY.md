# GTM One-Shot Deploy — NIGHT FURY (RunPod)

Цель: поднять **NULLXES Inference Worker** на **RunPod Linux GPU** (H200 primary, H100 secondary, B200 experimental) одной согласованной последовательностью команд.

Артефакт образа: `docker/Dockerfile.gpu` (база `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`). Сборка из корня репозитория:

```bash
docker build -f docker/Dockerfile.gpu -t nullxes/arachne-x-nightfury:2.0 .
```

Для полного avatar+TTS в образе (тяжелее):

```bash
docker build -f docker/Dockerfile.gpu \
  --build-arg INSTALL_AVATAR_DEPS=1 \
  --build-arg INSTALL_TTS_DEPS=1 \
  -t nullxes/arachne-x-nightfury:2.0-full .
```

---

## Required env vars (worker / GPU pod)

| Переменная | Обяз. | Описание |
|------------|-------|----------|
| `NULLXES_CHECKPOINT_DIR` | да* | Каталог весов avatar (tokenizer, vae, dit, avatar_single, audio). *Алиас: `ARACHNE_CHECKPOINT_DIR`. |
| `NULLXES_INFERENCE_SERVICE_KEY` | нет | Общий секрет; заголовок `X-NULLXES-Avatar-Inference-Key`. Алиасы: `NULLXES_AVATAR_INFERENCE_SERVICE_KEY`, `LONGCAT_INFERENCE_SERVICE_KEY` (legacy). |

Оркестратор (aiohttp `src/server`), если вызывает воркер:

| Переменная | Описание |
|------------|----------|
| `NULLXES_AVATAR_INFERENCE_URL` | Base URL воркера, без завершающего `/`. |
| `NULLXES_AVATAR_INFERENCE_PATH` | По умолчанию `/v1/arachne/generate` (legacy путь `/v1/longcat/generate` поддерживается воркером). |
| `NULLXES_AVATAR_INFERENCE_FRAMES_PATH` | По умолчанию `/v1/realtime/avatar_frames`. |
| `NULLXES_AVATAR_INFERENCE_ASYNC` | `1` — очередь jobs MP4. |
| `NULLXES_AVATAR_INFERENCE_JOBS_PATH` | По умолчанию `/v1/infer/jobs`. |

---

## Mounted weights layout

Рекомендуемый mount (пример):

```
/runpod-volume/weights/arachne-avatar/
  tokenizer/
  vae/
  dit/
  avatar_single/
  audio/
  ... (см. структуру bundle в README репозитория / training export)
```

```bash
export NULLXES_CHECKPOINT_DIR=/runpod-volume/weights/arachne-avatar
```

---

## Startup order

1. Поднять Pod с GPU (H200 preferred), диск с весами смонтирован.
2. Установить `NULLXES_CHECKPOINT_DIR` и при необходимости ключ.
3. Запустить uvicorn из каталога воркера (или через `PYTHONPATH` к репо):

```bash
cd /workspace/ARACHNE-X/services/arachnex-worker
export PYTHONPATH=/workspace/ARACHNE-X:/workspace/ARACHNE-X/services/arachnex-worker
uvicorn main:app --host 0.0.0.0 --port 9090
```

4. Дождаться `/health` → `{"status":"ok"}` (пайплайн **не** грузится на health).

---

## Healthcheck

```bash
curl -fsS http://127.0.0.1:9090/health
```

Ожидание: HTTP 200, JSON `status: ok`.

---

## Warmup flow

Первый реальный запрос к `POST /v1/realtime/avatar_frames` или MP4/job инициирует **lazy** загрузку пайплайна на GPU (десятки секунд — несколько минут в зависимости от диска и модели). Для стабильного SLO после деплоя выполнить один контрольный NDJSON-запрос (см. smoke).

---

## Inference smoke test (NDJSON)

Используйте `scripts/gpu/smoke_avatar_frames.sh` или минимальный `curl` с валидным `imageBase64` и аудио PCM16. Ожидание: `200`, `Content-Type: application/x-ndjson`, строки JSON с полями `seq`, `frameBase64` / rgb payload.

Переменная `NULLXES_SMOKE_ENGINE` по умолчанию `arachne`. Допустимы те же значения, что и поле `engine` у `POST /v1/realtime/avatar_frames` (включая `arachne_ultra_video` / `arachne_ultra_avatar` как алиасы к core — см. ниже). Алиасы `longcat` / пустое — core engine.

### Поле `engine` в `POST /v1/realtime/avatar_frames`

Тело запроса соответствует `StreamFramesBody` в [`services/arachnex-worker/main.py`](../../services/arachnex-worker/main.py) (camelCase в JSON). Реализация DiT: [`arachne_x/modules/arachne_video_dit.py`](../../arachne_x/modules/arachne_video_dit.py) и [`arachne_x/modules/avatar/arachne_avatar_dit.py`](../../arachne_x/modules/avatar/arachne_avatar_dit.py). Веса prod: [ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR).

| Значение `engine` | Поведение |
|-------------------|-----------|
| `arachne`, `""`, `core`, `nullxes`, `longcat` | Аудио-условленный avatar, NDJSON `rgb24_base64`. |
| `arachne_ultra_avatar`, `arachne_ultra_video` | **Алиасы к тому же пайплайну, что и `arachne`** — нужны для NULLXES HR AI: `VIDEO_ENGINE` / `resolveArachnePodEngine()` отдаёт эти строки в теле к воркеру. |
| `wan_s2v` | Ответ с ошибкой (на этом воркере не развёрнут). |
| иное | Первая строка NDJSON: `{"error": "..."}`. |

Имена полей: `sessionId`, `imageBase64`, `audioPcm16Base64` (предпочтительно, PCM16 mono 16 kHz) или `audioFloat32Base64`, опционально `prompt`, `negativePrompt`, `numInferenceSteps`, `textGuidanceScale`, `audioGuidanceScale`, `resolution`, `numFrames`.

---

## NULLXES HR AI — плашка аватара и RunPod (чеклист)

**Landing + KAIRA (25.05.2026):** полный deployment-контур (под → bridge → LiveKit → NULLXESLanding) — [`NULLXES_DEPLOYMENT_2026-05-25.md`](../NULLXES_DEPLOYMENT_2026-05-25.md).

Realtime «плашка» в HR идёт через **realtime-gateway**: TTS PCM → HTTP NDJSON к Inference Worker → кадры → публикация в SFU (GetStream или LiveKit — см. конфиг gateway). Полный smoke **на GPU-поде** (локальная машина без CUDA bundle обычно не подходит).

**На RunPod (Inference Worker):**

1. Задать `NULLXES_CHECKPOINT_DIR` (layout ULTRA-AVATAR).
2. Запустить uvicorn (см. [Startup order](#startup-order)); публичный URL прокси RunPod — base для HR.
3. Если включён секрет: один и тот же ключ в `NULLXES_INFERENCE_SERVICE_KEY` (или алиас) на поде и в HR.
4. `curl -fsS …/health` → `{"status":"ok"}`.
5. Опционально: `NULLXES_URL=https://… NULLXES_SMOKE_ENGINE=arachne_ultra_video bash scripts/gpu/smoke_avatar_frames.sh` — проверка HR-алиаса `engine`.

**В HR realtime-gateway (`.env`, минимум):**

| Переменная | Назначение |
|------------|------------|
| `AVATAR_VIDEO_ENABLED` | `true` |
| `VIDEO_ENGINE` или `VIDEO_MODEL` | `arachne`, `arachne_ultra_avatar` или `arachne_ultra_video` (на воркере все три сходятся в core avatar NDJSON) |
| `AVATAR_POD_URL` | Base URL воркера без завершающего `/` |
| `AVATAR_FRAMES_PATH` | Обычно `/v1/realtime/avatar_frames` (дефолт в gateway) |
| `NULLXES_AVATAR_INFERENCE_SERVICE_KEY` (или `NULLXES_INFERENCE_SERVICE_KEY`) | Совпадает с ключом на поде, если ключ задан |

---

## HR плашка: ULTRA-VIDEO и FURIA-EIDOLON (вне realtime NDJSON)

Текущая плашка заточена под **низколатентный** поток `avatar_frames`. **ARACHNE-X-ULTRA-VIDEO** и полуавтомат **FURIA-EIDOLON** ([`scripts/run_semiauto_turn.py`](../../scripts/run_semiauto_turn.py), job_runner) — **отдельные** продуктовые линии (другая латентность, MP4, HITL).

- **Преролл / фон (MP4):** короткий ролик через `POST /v1/arachne/generate` или async `/v1/infer/jobs` — отдельный UI-слой на HR, без замены NDJSON-потока.
- **Мост EIDOLON → плашка:** потребовал бы нового HTTP-адаптера в репозитории и смены источника видео на gateway (файл/HLS вместо NDJSON) — отдельный эпик по латентности и буферизации.

---

## MP4 generation validation

- Синхронно: `POST /v1/arachne/generate` с `task: audio-text-to-video` или `audio-image-to-video`, корректные `audioBase64` / `imageBase64`.
- Async: `POST /v1/infer/jobs` → poll `GET /v1/infer/jobs/{id}` → один раз `GET /v1/infer/jobs/{id}/result` → тело `video/mp4`.

Клиент оркестратора: при `NULLXES_AVATAR_INFERENCE_ASYNC=1` используется job API.

---

## VRAM expectations

Зависят от разрешения, `num_frames`, checkpoint и torch compile. Ориентиры фиксируются в `GTM_V2_RUNPOD_DEP_MATRIX.md` после прогона на H200/B200; закрыть чекбокс в `GTM_PRE_RELEASE_AUDIT.md` измеренными цифрами.

---

## Expected startup time

- Процесс uvicorn: секунды.
- Первая загрузка весов на GPU: **не** входит в cold start health; планируйте отдельный warmup (см. выше). Типично 30–120+ с для больших bundle с сетевого диска.

---

## Recovery procedure

1. Проверить `/health` и логи uvicorn.
2. Проверить `NULLXES_CHECKPOINT_DIR` и права чтения.
3. OOM / CUDA: уменьшить `num_frames` / разрешение в клиенте или перейти на больший GPU класс.
4. Очередь full (`503 inference queue full`): увеличить `INFERENCE_MAX_QUEUE` или горизонтально масштабировать воркеры (out of scope single-pod MVP).
5. После сбоя диска результата job: клиент должен повторить job (результат одноразовый).

---

## NDJSON validation (оркестратор)

Убедиться, что `NULLXES_WS_AVATAR_STREAM_STUB=0` и `NULLXES_AVATAR_INFERENCE_URL` указывает на живой воркер; WebSocket цепочка описана в `src/server/openapi_spec.py` и `WIRE_EXAMPLES` при наличии.
