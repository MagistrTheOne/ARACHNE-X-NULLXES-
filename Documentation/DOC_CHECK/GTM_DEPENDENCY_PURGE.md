# GTM Dependency Purge — единый production stack

Цель: один согласованный набор пинов для **RunPod Linux GPU** inference; без дублирующих деревьев и скрытых конфликтов.

## Канонические файлы

| Файл | Назначение |
|------|------------|
| `requirements.txt` | Core: torch, torchvision, flash-attn (Linux), transformers, diffusers, av, librosa, faster-whisper, … |
| `requirements_avatar.txt` | Доп. зависимости avatar-пайплайна (xformers и пр.). |
| `requirements-tts.txt` | Qwen3-TTS и связанное. |
| `requirements-audiodit.txt` | **Изолированно**: AudioDiT / upstream vendored tree — предпочтительно отдельный venv. |
| `requirements-datasets.txt` | Обучение / датасеты. |
| `services/longcat-worker/requirements.txt` | Только HTTP слой воркера (FastAPI, uvicorn, pydantic, …). |
| `docker/Dockerfile.gpu` | Слои установки из файлов выше. |

## Матрица ключевых пакетов (проверка на merge)

- [ ] **torch / torchvision** — одна пара версий в `requirements.txt`, согласована с базовым образом Dockerfile при необходимости.
- [ ] **xformers** — политика в `GTM_V2_RUNPOD_DEP_MATRIX.md`; пин только в `requirements_avatar.txt` если включён в образ.
- [ ] **flash-attn** — Linux-only marker в `requirements.txt`; не ставить на Windows prod path.
- [ ] **triton** — транзитивно от torch; не пинить отдельно без причины.
- [ ] **onnxruntime** — если используется, не дублировать `onnxruntime-gpu` + `onnxruntime` в одном env без документированного разделения.
- [ ] **transformers** — одна версия на runtime env.
- [ ] **diffusers** — одна версия, согласованная с кодом пайплайна.
- [ ] **librosa** — одна версия в core; audiodit venv может иметь свой pin (изолировать).
- [ ] **av** — согласован с torch/torchvision для декодирования видео.
- [ ] **faster-whisper** — в core `requirements.txt` для ASR path; не тянуть второй ctranslate2 stack в audiodit venv без нужды.

## Политика audiodit

Провайдер CLI `--tts_provider longcat_audiodit` (legacy имя провайдера) / `audiodit` — опциональный путь. Зависимости из `requirements-audiodit.txt` **не** смешивать с prod avatar worker venv, если это приводит к конфликту версий librosa/onnx.

## Удалить / не допускать

- [ ] Дублирующие противоречивые пины одного пакета в одном файле.
- [ ] Abandoned compatibility hacks без ссылки на активный код.
- [ ] Неиспользуемые optional пакеты в `requirements.txt` без обоснования (перенос в optional файл).

## Следующий шаг после аудита

Зафиксировать результат прогона `pip check` (или эквивалент) в CI и закрыть соответствующие чекбоксы в `GTM_PRE_RELEASE_AUDIT.md`.
