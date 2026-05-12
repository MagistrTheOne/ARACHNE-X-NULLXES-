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
cd /workspace/ARACHNE-X/services/longcat-worker
export PYTHONPATH=/workspace/ARACHNE-X:/workspace/ARACHNE-X/services/longcat-worker
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

Переменная `NULLXES_SMOKE_ENGINE` по умолчанию `arachne` (алиасы `longcat` / пустое — core engine).

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
