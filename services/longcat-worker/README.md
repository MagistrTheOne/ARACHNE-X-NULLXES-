# LongCat-Video HTTP worker (GPU sidecar)

ARACHNE-X вызывает **`POST /v1/longcat/generate`** и ожидает **тело ответа = MP4** (`video/mp4`) или JSON с **`videoBase64`**.

Реализовано:

1. **Полный inference** — `longcat_generate_once.py` повторяет трёхстадийный пайплайн из upstream ([`run_demo_text_to_video.py`](https://github.com/meituan-longcat/LongCat-Video/blob/main/run_demo_text_to_video.py), `run_demo_image_to_video.py`, `run_demo_video_continuation.py`): базовая генерация → distill (LoRA) → refinement (720p / 30 fps для refine).
2. **Оркестрация** — `main.py` (FastAPI) пишет входы во временный каталог, вызывает **`torchrun`** через `longcat_subprocess.py`, отдаёт байты MP4.

## Зависимости (только для HTTP-слоя)

```bash
cd services/longcat-worker
pip install -r requirements.txt
```

Сам **torch / flash-attn / longcat_video** ставятся в окружение [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) по их README и [модели](https://huggingface.co/meituan-longcat/LongCat-Video).

## Обязательные переменные для прода

| Переменная | Назначение |
|------------|------------|
| `LONGCAT_VIDEO_REPO` | Корень клонированного репозитория LongCat-Video (каталог, где лежит пакет `longcat_video`). |
| `LONGCAT_CHECKPOINT_DIR` | Путь к весам, например `./weights/LongCat-Video` после `huggingface-cli download`. |

Запуск API (лучше тот же Python/conda, где установлен LongCat и CUDA):

```bash
export LONGCAT_VIDEO_REPO=/workspace/LongCat-Video
export LONGCAT_CHECKPOINT_DIR=/workspace/LongCat-Video/weights/LongCat-Video
uvicorn main:app --host 0.0.0.0 --port 9090
```

В ARACHNE-X: `NULLXES_AVATAR_INFERENCE_URL=http://<gpu-host>:9090`.

## Опционально

| Переменная | По умолчанию | Смысл |
|------------|--------------|--------|
| `LONGCAT_NPROC` | `1` | `torchrun --nproc_per_node` |
| `LONGCAT_CONTEXT_PARALLEL_SIZE` | = `LONGCAT_NPROC` | context parallel (как в upstream multi-GPU) |
| `LONGCAT_ENABLE_COMPILE` | выкл. | `1` / `true` — `--enable_compile` (первый прогон дольше) |
| `LONGCAT_TORCHRUN` | `torchrun` | Путь к `torchrun`, если не в `PATH` |
| `LONGCAT_SUBPROCESS_TIMEOUT_SEC` | `7200` | Таймаут одного запроса |
| `LONGCAT_INFERENCE_SERVICE_KEY` | пусто | Если задан — проверка заголовка `X-NULLXES-Avatar-Inference-Key` |
| `LONGCAT_MOCK_MP4_PATH` | — | Dev без GPU: отдать готовый mp4 без torch |
| `LONGCAT_MOCK_VIDEO_BASE64` | — | Ответ JSON `videoBase64` (редкие тесты) |

## Dev без GPU

```bash
export LONGCAT_MOCK_MP4_PATH=/path/to/sample.mp4
uvicorn main:app --host 127.0.0.1 --port 9090
```

## Тело запроса `POST /v1/longcat/generate`

```json
{
  "task": "text-to-video",
  "prompt": "…",
  "sessionId": "ui_sess_…",
  "negative_prompt": "опционально",
  "imageBase64": "<только для image-to-video>",
  "continuationState": "<base64 mp4 для video-continuation>"
}
```

`task`: `text-to-video` | `image-to-video` | `video-continuation` — соответствует режимам из [карточки модели](https://huggingface.co/meituan-longcat/LongCat-Video).

## Команда torchrun (эквивалент ручного запуска)

Один GPU:

```bash
cd "$LONGCAT_VIDEO_REPO"
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 \
  /path/to/ARACHNE-X/services/longcat-worker/longcat_generate_once.py \
  --checkpoint_dir "$LONGCAT_CHECKPOINT_DIR" \
  --job_json /tmp/job.json \
  --context_parallel_size 1
```

Два GPU (как в upstream):

```bash
export LONGCAT_NPROC=2
export LONGCAT_CONTEXT_PARALLEL_SIZE=2
```

## Примечания

- Один запрос = один полный прогон пайплайна; время — минуты, не миллисекунды.
- Для продакшена обычно ставят **очередь** (Redis/Celery) и отдельный пул GPU-воркеров; этот сервис — минимальный синхронный HTTP фасад.
- При ошибках CUDA / OOM клиент ARACHNE-X получит **502** с фрагментом stderr из subprocess.

## ARACHNE-X ULTRA (MagistrTheOne / HF)

Если заданы **`ARACHNE_VIDEO_REPO`** и **`ARACHNE_CHECKPOINT_DIR`**, воркер использует **`arachne_subprocess.py`**: `torchrun ... run_t2v.py` / `run_at2v.py` и `--input_json` (см. [ARACHNE-X-ULTRA](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA)).

| Переменная | Смысл |
|------------|--------|
| `ARACHNE_VIDEO_REPO` | Корень репозитория с `run_t2v.py`, `run_at2v.py` |
| `ARACHNE_CHECKPOINT_DIR` | Веса (например после `huggingface-cli download MagistrTheOne/ARACHNE-X-ULTRA`) |
| `ARACHNE_SCRIPT_T2V` / `ARACHNE_SCRIPT_AT2V` | Имена скриптов (по умолчанию `run_t2v.py`, `run_at2v.py`) |
| `ARACHNE_NPROC` | `torchrun --nproc_per_node` (по умолчанию 1) |
| `ARACHNE_INPUT_JSON_OUTPUT_KEY` | Ключ в JSON для абсолютного пути выходного mp4 (по умолчанию `output_video`) |
| `ARACHNE_IMAGE_TO_VIDEO_SCRIPT` | `t2v` или `at2v` — какой скрипт для `image-to-video` |

**Задачи HTTP:** `text-to-video`, `image-to-video`, `video-continuation`, `audio-text-to-video`, `audio-image-to-video` (последние три только при пути ARACHNE).

Поля: `audioBase64`, `numSegments`, `refImgIndex`, `inputJson` (доп. поля в `input_json`).

**Моки:** `LONGCAT_MOCK_MP4_PATH` / `LONGCAT_MOCK_VIDEO_BASE64` работают **только** при **`ALLOW_INFERENCE_DEV_MOCK=1`**. Без этого и без репо+весов — **503**.

## Очередь jobs (один GPU / один под H200)

Сериальная очередь в памяти процесса: **`POST /v1/infer/jobs`** → `{"jobId","status":"queued"}` → **`GET /v1/infer/jobs/{jobId}`** до `status: done` → **`GET /v1/infer/jobs/{jobId}/result`** (один раз, `video/mp4`).

| Env | По умолчанию | Смысл |
|-----|--------------|--------|
| `INFERENCE_MAX_QUEUE` | `32` | Макс. глубина очереди (ожидающих jobs) |

Синхронный **`POST /v1/longcat/generate`** без изменений.

На стороне ARACHNE-X aiohttp: **`NULLXES_AVATAR_INFERENCE_ASYNC=1`** — клиент использует job API; **`NULLXES_AVATAR_INFERENCE_JOBS_PATH`** (по умолчанию `/v1/infer/jobs`), **`NULLXES_AVATAR_INFERENCE_POLL_MS`** (по умолчанию `500`).
