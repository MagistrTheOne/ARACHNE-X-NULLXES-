# NULLXES DEPLOY — ARACHNE-X ULTRA (13.6B)

**Document ID:** `NULLXES_DEPLOY_29-05-2026`  
**Date:** 2026-05-29  
**Branch:** `arachne-last-patch`  
**Audience:** enterprise customers, ML ops, backend engineers (self-hosted / RunPod)


| Channel    | Contact                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------ |
| Email      | [ceo@nullxes.com](mailto:ceo@nullxes.com)                                                              |
| Telegram   | [@MagistrTheOne](https://t.me/MagistrTheOne)                                                           |
| Repository | [github.com/MagistrTheOne/ARACHNE-X-NULLXES-](https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git) |
| License    | [NULLXES Proprietary License 2.0](../LICENSE)                                                          |


---

## Область документа

### Входит в развёртку


| Компонент                  | Описание                                                                    |
| -------------------------- | --------------------------------------------------------------------------- |
| **ARACHNE-X-ULTRA-VIDEO**  | Базовый DiT + Wan VAE + UMT5 + scheduler (`t2v`, `i2v`, `vc`)               |
| **ARACHNE-X-ULTRA-AVATAR** | Avatar DiT 13.6B + wav2vec + режимы `ai2v`, `streaming_ai2v`, `at2v`, `avc` |
| **CLI inference**          | `scripts/infer.py` — оцифровка, QA, smoke                                   |
| **GPU worker**             | `services/arachnex-worker` — NDJSON realtime + MP4 jobs                     |
| **Orchestrator**           | `src/server/`* — WebSocket, STT/LLM/TTS → worker (отдельный хост)           |


### Не входит (не разворачивать по этому документу)


| Исключено                                                       | Причина                                         |
| --------------------------------------------------------------- | ----------------------------------------------- |
| **ARACHNE-FOUNDATION-50B** / depth surgery                      | Отдельный research/pretrain track               |
| **ARACHNE-SPECTRUM-100B** / MoE                                 | Roadmap, не prod                                |
| `requirements-training.txt`, `requirements-datasets.txt`        | Latent export / corpus prep                     |
| `requirements-audiodit.txt`                                     | Lab AudioDiT — **конфликт transformers** с core |
|                                                                 |                                                 |
| `pipeline_audio_i2v.py`, `streaming_inference.py` как prod path | Lab / prototype only (`Claude_senior.md`)       |


См. также: `[ARCHITECTURE.md](../ARCHITECTURE.md)` · `[REQUIREMENTS.md](REQUIREMENTS.md)` · `[NULLXES_ARACHNE_RUNPOD_27-05-2026.md](NULLXES_ARACHNE_RUNPOD_27-05-2026.md)` (детальный RunPod playbook).

---

## Что такое ARACHNE-X ULTRA V2

**ARACHNE-X ULTRA V2** — инфраструктура realtime digital-human.


| Параметр           | Значение                                                                                                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Avatar DiT         | **~13.6B** ACV-DiT (48 blocks, dim 4096)                                                                                                                                          |
| VAE                | Wan (`AutoencoderKLWan`, z=16)                                                                                                                                                    |
| Text               | UMT5-XXL                                                                                                                                                                          |
| Audio conditioning | Wav2Vec2 + audio cross-attention                                                                                                                                                  |
| Scheduler          | Flow-match Euler (`operational` ≈ 12 distill steps)                                                                                                                               |
| Веса               | **NULLXES proprietary** — [ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) · [ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) |


### Что нового в релизе 2026-05-27 (Wave 1 hardening)

По сравнению с «одним скриптом infer»:


| Возможность               | Эффект для prod                                              |
| ------------------------- | ------------------------------------------------------------ |
| **Единый WS-путь**        | Один orchestrator; нет shadow playback в `realtime_api.py`   |
| **Explicit GPU queue**    | `503 worker_busy` + `retryAfterMs` вместо тихового зависания |
| **Incremental wav2vec**   | TTFF не ждёт конец длинной реплики                           |
| **operational profile** | Chunked denoise + distill — realtime по умолчанию            |
| **Stability OS**          | Cross-chunk KV, silence gate, identity drift metrics         |
| **Worker lifecycle**      | `/health`, drain/activate, `/v1/runtime/metrics`             |
| **WS schema v1**          | `protocolVersion: "v1"`, event `avatar.stream.chunk`         |
| **Process isolation**     | TTS/LLM **только** в orchestrator; GPU worker — DiT only     |


---

## Архитектура развёртки (три слоя)

```text
┌─────────────────────────────────────────────────────────────┐
│  Layer A — Клиент / Frontend (ваш UI, jobaidemo, HR app)   │
│  WebSocket + REST · без секретов worker/GPU                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Layer B — Orchestrator (CPU, отдельный хост)                 │
│  src/server/webrtc_server.py · SessionWorker · TTS · STT/LLM  │
│  pip: requirements_orchestrator.txt                           │
└───────────────────────────┬─────────────────────────────────┘
                            │ POST /v1/realtime/avatar_frames
┌───────────────────────────▼─────────────────────────────────┐
│  Layer C — GPU Worker (RunPod H200/H100)                      │
│  services/arachnex-worker :9090                               │
│  pip: requirements_avatar.txt + arachnex-worker/requirements.txt │
└─────────────────────────────────────────────────────────────┘

Параллельно на GPU pod (оператор / QA):
  scripts/infer.py --mode ai2v | streaming_ai2v → MP4 + .run.json
```

**Жёсткое правило:** не загружать TTS и 13.6B DiT в **один** процесс. VRAM contention убивает realtime.

---

## Системные требования


| Ресурс                | Минимум (prod)    | Рекомендация                            |
| --------------------- | ----------------- | --------------------------------------- |
| GPU                   | NVIDIA H100 80GB  | **H200** 141GB                          |
| VRAM (avatar runtime) | ~110 GB loaded    | H200 headroom для queue + KV            |
| Disk                  | 250 GB свободно на **volume** `/workspace` | 1200 GB volume (RunPod UI)              |
| RAM                   | 64 GB             | 128 GB+ при compile flash-attn          |
| OS (infer)            | **Linux**         | RunPod Ubuntu / PyTorch CUDA 12.4 image |
| Python                | **3.10** или 3.11 | 3.10 на RunPod                          |
| CUDA (wheel)          | **12.4**          | `torch==2.6.0+cu124`                    |
| Windows               | Только SSH-клиент | Inference на pod, не на Windows         |


---

## Зависимости: установка по шагам (важно)

**Короткий ответ: да — зависимости ставятся отдельно, не одной командой «на всё».**

В репозитории **несколько файлов** `requirements*.txt`. Каждый файл — для **своей части системы**. Если поставить «всё подряд» или перепутать порядок — сломается CUDA, flash-attn не соберётся, или prod упадёт на конфликте версий.

### Три «коробки» софта (что куда ставить)

```text
┌─────────────────────────────────────────────────────────────────┐
│  КОРОБКА 1 — GPU pod (RunPod H200/H100)                         │
│  Нужна для: scripts/infer.py + arachnex-worker :9090            │
│                                                                 │
│  Шаг A → PyTorch (отдельно, с CUDA 12.4)                        │
│  Шаг B → FlashAttention (отдельно, после PyTorch)               │
│  Шаг C → requirements_avatar.txt                                │
│  Шаг D → services/arachnex-worker/requirements.txt              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  КОРОБКА 2 — Orchestrator (CPU, другой сервер / другой pod)     │
│  Нужна для: WebSocket gateway, STT, TTS, маршрутизация на GPU  │
│                                                                 │
│  Шаг E → requirements_orchestrator.txt                          │
│  (опционально) requirements-tts.txt — если свой TTS на CPU      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  НЕ СТАВИТЬ в prod GPU / orchestrator venv                       │
│  requirements-training.txt   — подготовка датасетов             │
│  requirements-datasets.txt   — то же + pandas                   │
│  requirements-audiodit.txt     — lab, ломает transformers       │
└─────────────────────────────────────────────────────────────────┘
```

| Файл | Куда ставить | Простыми словами |
|------|--------------|------------------|
| **PyTorch + torchvision** | GPU pod, **шаг 1** | «Движок» нейросети на видеокарте. Без него ничего не работает. |
| **flash-attn** | GPU pod, **шаг 2** | Ускоритель внимания. Ставится **только после** PyTorch, **только Linux**. |
| **`requirements_avatar.txt`** | GPU pod, **шаг 3** | Библиотеки для avatar DiT: diffusers, transformers, librosa, … |
| **`services/arachnex-worker/requirements.txt`** | GPU pod, **шаг 4** | HTTP-сервер worker: FastAPI, uvicorn (лёгкий слой поверх GPU). |
| **`requirements_orchestrator.txt`** | CPU orchestrator | WebSocket + whisper + edge-tts. **Не нужен** на «глупом» GPU-only pod. |
| **`requirements-tts.txt`** | Отдельно / CPU | Qwen TTS — **не** в процесс с DiT (иначе OOM). |
| **`requirements-training.txt`** | Lab venv | Export latents, training — **не prod infer**. |
| **`requirements-audiodit.txt`** | Отдельный venv | Lab AudioDiT — **конфликт** с avatar core. |

> **`requirements.txt`** — это «ядро» внутри `requirements_avatar.txt`. Клиент **не вызывает** его отдельно в prod: достаточно `pip install -r requirements_avatar.txt` **после** torch и flash-attn.

### Два сценария развёртки

| Сценарий | Сколько venv | Что ставить |
|----------|--------------|-------------|
| **A. Только GPU + smoke** (Фазы 0–7) | **1 venv** на RunPod | Шаги A → B → C → D |
| **B. Полный realtime** (GPU + orchestrator) | **2 venv** (рекомендуется) | GPU: A→D · CPU: E (+ опционально TTS) |

**Сценарий A** — минимум для проверки «модель рисует видео» (`infer.py`, worker).  
**Сценарий B** — prod HR: клиент по WebSocket → orchestrator → GPU worker.

Orchestrator **можно** поставить в тот же venv на GPU pod (для теста), но в prod **лучше отдельный CPU-хост**: TTS и STT не должны делить VRAM с 13.6B DiT.

### После reconnect (GPU pod) — не переустанавливать venv

Если deps (шаги A→D) уже стояли и вы **просто переподключились по SSH**:

```bash
export ARACHNE_ROOT=/workspace/ARACHNE-X
source "$ARACHNE_ROOT/.venv/bin/activate"
export PYTHONPATH="$ARACHNE_ROOT"
```

**Не запускайте** `python3 -m venv` и **не** `pip install -r requirements_*` повторно, пока не сломали venv намеренно.

| Ситуация | Действие |
|----------|----------|
| Новый SSH-сессия | `source .../activate` |
| Stop/Start pod, volume на месте | `source .../activate` |
| Edit Pod, `/workspace` цел | `source .../activate` |
| Первый деплoy, нет `.venv` | `python3 -m venv` + шаги A→D |
| `which python` не из `.venv` | `source` или пересоздать venv |

### Порядок установки на GPU pod (копируйте по очереди)

Каждый блок — **отдельная команда**. Дождитесь `OK` / успеха перед следующим.

```bash
source "$ARACHNE_ROOT/.venv/bin/activate"
cd "$ARACHNE_ROOT"
```

**Шаг A — PyTorch (CUDA 12.4)**

```bash
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print('TORCH OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ожидаете: `TORCH OK 2.6.0+cu124 True NVIDIA H200 ...` (или H100).  
Если `False` — **стоп**, не идите дальше (см. §C).

**Шаг B — FlashAttention**

```bash
pip install ninja packaging psutil wheel
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation

python -c "import flash_attn; print('FLASH OK', flash_attn.__version__)"
```

Сборка 10–30 мин. На Windows **не ставить** — prod только Linux GPU.

**Шаг C — Avatar / DiT stack**

```bash
pip install numpy==1.26.4
pip install -r requirements_avatar.txt
```

**Шаг D — Worker HTTP**

```bash
pip install -r services/arachnex-worker/requirements.txt
```

**Проверка после A–D**

```bash
export PYTHONPATH="$ARACHNE_ROOT"
python -c "from arachne_x.loader import load_avatar_pipeline; print('IMPORT OK')"
python -c "import fastapi, uvicorn; print('WORKER HTTP OK')"
```

### Установка на orchestrator (CPU, сценарий B)

На **другом** сервере (или отдельном venv):

```bash
cd /path/to/ARACHNE-X
python3 -m venv .venv-orch && source .venv-orch/bin/activate
pip install -U pip setuptools wheel

# Тянет requirements_avatar.txt внутри — нужен для импортов src/server,
# но DiT на CPU здесь НЕ загружается, только HTTP к GPU worker.
pip install -r requirements_orchestrator.txt
```

Проверка:

```bash
export PYTHONPATH="$PWD"
python -c "import aiohttp; from src.server.webrtc_server import create_app; print('ORCH OK')"
```

Опционально свой TTS (Qwen) — **отдельный процесс**, не в worker:

```bash
pip install -r requirements-tts.txt
```

### Чего НЕ делать (типичные ошибки)

| Ошибка | Почему плохо |
|--------|--------------|
| `pip install -r requirements.txt` **до** torch | Неправильная версия CUDA / torch |
| `pip install -r requirements_avatar.txt` **до** flash-attn | Может подтянуть несовместимый stack |
| Одна команда `pip install -r requirements-training.txt` на prod GPU | Лишние пакеты, не нужны для infer |
| `pip install -r requirements-audiodit.txt` в тот же venv | **Ломает** `transformers` (4.41 vs ≥5.3) |
| TTS + DiT в одном процессе на GPU | OOM, realtime падает |
| Ставить flash-attn на Windows «для теста» | Не поддерживается; infer только на Linux pod |
| `pip install -r requirements_orchestrator.txt` **вместо** avatar на GPU-only pod | Лишний whisper/tts на GPU; не критично, но не канон |

### Чеклист зависимостей (отметьте в ops log)

| # | Проверка | Команда | Pass |
|---|----------|---------|------|
| D1 | PyTorch + CUDA | шаг A | `cuda.is_available() == True` |
| D2 | FlashAttention | шаг B | `FLASH OK` |
| D3 | Avatar stack | шаг C | `pip show transformers` → **4.41.0** |
| D4 | Worker HTTP | шаг D | `import fastapi` OK |
| D5 | Loader import | после D4 | `load_avatar_pipeline` import OK |
| D6 | Orchestrator (если B) | шаг E | `create_app` import OK |

Подробный audit версий: [`REQUIREMENTS.md`](REQUIREMENTS.md).

---

## Поэтапная развёртка

### RunPod UI — Pod → Edit → Save (сделать до SSH)

> **Edit Pod сбрасывает контейнер.** После Save pod перезапустится.  
> **Все данные вне `/workspace` пропадут.** Код, venv, веса, output — только на volume mount.

Путь в UI: **Pods → ваш pod → Edit → заполнить поля → Save**.

#### Container image

```text
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

| Поле | Значение | Зачем |
|------|----------|-------|
| **Container image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | CUDA 12.4 + devel (нужен для сборки flash-attn) |
| **Container disk** | **750** GB | Временный диск контейнера (сбрасывается при Edit/Recreate) |
| **Volume disk** | **1200** GB | Постоянный диск (веса ~200 GB + venv + output) |
| **Volume mount path** | **`/workspace`** | Единственное место, где данные переживают restart |

После первого входа в pod всё кладите под `/workspace` (репо, `weights/`, `.venv`, `output/`).

#### Expose HTTP ports (max 10)

В поле **Expose HTTP ports** вставьте **ровно**:

```text
8888,8000,8010,8080,9090
```

RunPod даст публичные URL вида `https://<pod-id>-<PORT>.proxy.runpod.net`.

| Port | Сервис ARACHNE | Обязательно? |
|------|----------------|--------------|
| **9090** | `arachnex-worker` — NDJSON realtime, `/health`, MP4 jobs | **Да** (prod GPU worker) |
| **8080** | Orchestrator / WebSocket gateway (`NULLXES_HTTP_PORT`) | Если gateway на **этом же** pod (обычно отдельный CPU) |
| **8888** | Jupyter / notebooks | Нет (dev, опционально) |
| **8000** | Dev / alt HTTP | Нет (резерв) |
| **8010** | Dev / alt HTTP | Нет (резерв) |

Минимум для prod infer + worker: **`9090`**. Остальные — dev/резерв; рекомендуемый набор для NULLXES — строка выше.

Проверка после Save и старта worker:

```bash
curl -fsS http://127.0.0.1:9090/health | jq .
# снаружи:
curl -fsS "https://<pod-id>-9090.proxy.runpod.net/health" | jq .
```

#### Expose TCP ports

В поле **Expose TCP ports** укажите:

```text
22
```

| Port | Назначение |
|------|------------|
| **22** | SSH (Connect → **SSH over exposed TCP** в RunPod UI) |

TCP **9090** для worker **не нужен**, если worker уже в **HTTP ports** — orchestrator ходит на `https://...-9090.proxy.runpod.net`.

#### Чеклист RunPod UI (до Фазы 0)

| # | Поле UI | Значение |
|---|---------|----------|
| R1 | Container image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| R2 | Container disk | 750 GB |
| R3 | Volume disk | 1200 GB |
| R4 | Volume mount path | `/workspace` |
| R5 | Expose HTTP ports | `8888,8000,8010,8080,9090` |
| R6 | Expose TCP ports | `22` |
| R7 | GPU | H200 или H100 |
| R8 | Save | Pod перезапустится — данные только в `/workspace` |

> **После Edit Pod / reconnect SSH:** venv на `/workspace/ARACHNE-X/.venv` **обычно уже есть** — только `source`, **не** `python3 -m venv` снова. См. [Фаза 1 → venv](#venv-кому-где-когда-создавать).

---

Все команды ниже — **на Linux GPU pod** после SSH, если не указано иное.

### Фаза 0 — Проверка железа (~2 мин)

```bash
nvidia-smi
python3 --version
df -h /workspace
```


| Проверка  | Pass          |
| --------- | ------------- |
| GPU виден | H200 / H100   |
| Python    | ≥ 3.10        |
| Диск      | ≥ 250 GB free на `/workspace` (volume 1200 GB в UI) |


**Если упало:** см. [§A. Инфраструктура](#a-инфраструктура-gpu-disk-ssh).

---

### Фаза 1 — Репозиторий и venv (~10 мин)

#### venv: кому, где, когда создавать

| Кто / где | Путь venv | Когда **создавать** (`python3 -m venv`) | Когда **только activate** (`source .../activate`) |
|-----------|-----------|----------------------------------------|--------------------------------------------------|
| **GPU pod (RunPod)** — infer + worker | `/workspace/ARACHNE-X/.venv` | **Один раз**, первый деплой на pod | Каждый новый SSH, reconnect, после Stop/Start pod, после Edit Pod (если volume `/workspace` на месте) |
| **Orchestrator (CPU)** — WebSocket gateway | `/path/to/ARACHNE-X/.venv-orch` | **Один раз** на CPU-сервере (Фаза 8) | Каждый новый SSH на orchestrator |
| **Lab AudioDiT** | отдельный `.venv-audiodit` | Только lab, **не prod** | — |

**Правило для GPU pod:** venv лежит **внутри** `/workspace/ARACHNE-X/.venv` → переживает reconnect и Edit Pod.  
**Не создавайте venv заново** после каждого входа — только активируйте.

**После каждого SSH / reconnect (GPU pod)** — копируйте это:

```bash
export ARACHNE_ROOT=/workspace/ARACHNE-X
source "$ARACHNE_ROOT/.venv/bin/activate"
cd "$ARACHNE_ROOT"
export PYTHONPATH="$ARACHNE_ROOT"
```

Проверка, что venv живой:

```bash
which python
# ожидаете: /workspace/ARACHNE-X/.venv/bin/python

python -c "import torch; print(torch.__version__)"
# если ModuleNotFoundError — venv пустой или не тот; см. ниже
```

**Первый раз на pod** (или venv удалён / не на volume):

```bash
cd /workspace
git clone https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git ARACHNE-X
cd /workspace/ARACHNE-X
git fetch origin && git checkout arachne-last-patch
git log -1 --oneline

export ARACHNE_ROOT=/workspace/ARACHNE-X

# ТОЛЬКО если .venv ещё нет:
test -d "$ARACHNE_ROOT/.venv" || python3 -m venv "$ARACHNE_ROOT/.venv"

source "$ARACHNE_ROOT/.venv/bin/activate"
pip install -U pip setuptools wheel
mkdir -p "$ARACHNE_ROOT/output"

apt-get update && apt-get install -y git ffmpeg jq tmux \
  libsndfile1 libgl1 build-essential ninja-build
```

> **Уже ставили deps (A→D) и просто переподключились?**  
> Достаточно блока «После каждого SSH» выше. **Не** запускайте `pip install` и **не** `python3 -m venv` повторно.

> **Orchestrator (отдельный CPU)** — свой venv, не путать с GPU:  
> `python3 -m venv .venv-orch && source .venv-orch/bin/activate` → Фаза 8.

#### Папка `assets/` (лица и демо-аудио)

Референсы **уже в репозитории** — отдельный каталог `input/` не используется.

```text
assets/avatar/single/<persona>/
  image.jpg | *.png | face.jpg   # портрет для ai2v / enroll_identity
  audio.wav | *.mp3              # речь для lipsync smoke
  <persona>.json                 # prompt + пути (опционально)
```

Примеры в репо: `elena`, `anna`, `kaira`, `Jena`, `MaximOnyushko` — см. `assets/avatar/single/`.

Канонический smoke-пресет (Elena):

```bash
export ARACHNE_FACE="$ARACHNE_ROOT/assets/avatar/single/elena/image.jpg"
export ARACHNE_AUDIO="$ARACHNE_ROOT/assets/avatar/single/elena/audio.wav"
ls -la "$ARACHNE_FACE" "$ARACHNE_AUDIO"
```

Своя персона: положите файлы в `assets/avatar/single/<имя>/` и укажите те же переменные в командах ниже.

---

### Фаза 2 — Веса Hugging Face (~2–5 мин)

Токен Hugging Face с доступом к **ULTRA-AVATAR** и **ULTRA-VIDEO**. **Не коммитить в git, не вставлять в тикеты.**

> На RunPod с `HF_HUB_ENABLE_HF_TRANSFER=1` оба репо (~200 GB) обычно качаются **2–5 минут**.  
> После reconnect: если `$ARACHNE_ROOT/weights/` уже полный — **скачивать снова не нужно**, переходите к Фазе 3 verify.

```bash
export HF_TOKEN=""
# подставьте свой токен, например:
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"

pip install -U "huggingface_hub[cli]>=0.34,<1.0"
huggingface-cli login --token "$HF_TOKEN"

pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$ARACHNE_ROOT/weights"

hf download MagistrTheOne/ARACHNE-X-ULTRA-AVATAR \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"

hf download MagistrTheOne/ARACHNE-X-ULTRA-VIDEO \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"
```

**Проверка AVATAR:**

```bash
CKPT="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"
find "$CKPT" -name '*.incomplete' | wc -l   # → 0
ls "$CKPT/avatar_single"/diffusion_pytorch_model-*.safetensors | wc -l   # → 6
mkdir -p "$CKPT/audio"
ln -sfn "$CKPT/chinese-wav2vec2-base" "$CKPT/audio/wav2vec2"
```

**Если упало:** см. [§B. Веса](#b-веса-hugging-face).

---

### Фаза 3 — Merged runtime layout (~2 мин)

Avatar-режимы требуют **symlink bundle** VIDEO + AVATAR:

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
rm -rf "$NULLXES_CHECKPOINT_DIR" && mkdir -p "$NULLXES_CHECKPOINT_DIR/audio"

for d in tokenizer text_encoder vae scheduler; do
  ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO/$d" "$NULLXES_CHECKPOINT_DIR/$d"
done
for d in avatar_single avatar_multi vocal_separator; do
  ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/$d" "$NULLXES_CHECKPOINT_DIR/$d"
done
ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/chinese-wav2vec2-base" \
  "$NULLXES_CHECKPOINT_DIR/audio/wav2vec2"

export PYTHONPATH="$ARACHNE_ROOT"
python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["NULLXES_CHECKPOINT_DIR"])
need = ["tokenizer", "vae", "text_encoder", "scheduler", "avatar_single", "audio/wav2vec2"]
print("missing:", [p for p in need if not (root / p).exists()] or "none")
PY
```

---

### Фаза 4 — Зависимости GPU pod

> Полная расшифровка: раздел **[Зависимости: установка по шагам](#зависимости-установка-по-шагам-важно)** выше.  
> Ниже — те же шаги A→D для Фазы 4 (не пропускайте и не меняйте порядок).

```bash
source "$ARACHNE_ROOT/.venv/bin/activate"
cd "$ARACHNE_ROOT"

# A — PyTorch
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('TORCH OK', torch.__version__, torch.cuda.is_available())"

# B — FlashAttention
pip install ninja packaging psutil wheel
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation
python -c "import flash_attn; print('FLASH OK')"

# C — Avatar stack
pip install numpy==1.26.4
pip install -r requirements_avatar.txt

# D — Worker HTTP
pip install -r services/arachnex-worker/requirements.txt

# Verify
export PYTHONPATH="$ARACHNE_ROOT"
python -c "from arachne_x.loader import load_avatar_pipeline; print('import OK')"
```

**Не устанавливать в этот venv (prod GPU):**

```bash
# pip install -r requirements-audiodit.txt     # КОНФЛИКТ transformers
# pip install -r requirements-training.txt    # не prod infer
# pip install -r requirements_orchestrator.txt  # только CPU orchestrator (сценарий B)
```

**Если упало:** см. [§C. Python / CUDA / FlashAttention](#c-python--cuda--flashattention).

---

### Фаза 5 — Identity bank (один раз на персону, ~2 мин)

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
export PYTHONPATH="$ARACHNE_ROOT"
export ARACHNE_FACE="$ARACHNE_ROOT/assets/avatar/single/elena/image.jpg"

python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode enroll_identity \
  --image "$ARACHNE_FACE" \
  --identity_id 1 \
  --identity_bank_save_path "$ARACHNE_ROOT/output/elena_identity_bank.pt"
```

Для prod worker:

```bash
export NULLXES_IDENTITY_BANK_PATH="$ARACHNE_ROOT/output/elena_identity_bank.pt"
```

---

### Фаза 6 — CLI smoke (инференс, offline QA)

Подготовка аудио **16 kHz mono** (из `assets/`):

```bash
export ARACHNE_AUDIO="$ARACHNE_ROOT/assets/avatar/single/elena/audio.wav"
export ARACHNE_FACE="$ARACHNE_ROOT/assets/avatar/single/elena/image.jpg"

ffmpeg -y -i "$ARACHNE_AUDIO" -ar 16000 -ac 1 "$ARACHNE_ROOT/output/elena_16k.wav"
```

#### 6.1 Primary: `ai2v` (image + audio → MP4)

```bash
export ARACHNE_RUNTIME_PROFILE=operational
export ARACHNE_CHUNK_KV=1
export NULLXES_IDENTITY_BANK_PATH="$ARACHNE_ROOT/output/elena_identity_bank.pt"

python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --runtime_profile operational \
  --image "$ARACHNE_FACE" \
  --audio "$ARACHNE_ROOT/output/elena_16k.wav" \
  --prompt "Person speaking naturally to camera, stable identity, precise lipsync." \
  --negative_prompt "blurry, distorted face, bad anatomy, watermark" \
  --identity_bank_path "$ARACHNE_ROOT/output/elena_identity_bank.pt" \
  --identity_id 1 \
  --resolution 480p \
  --output "$ARACHNE_ROOT/output/avatar_ai2v.mp4"

jq '.sampling_metrics' "$ARACHNE_ROOT/output/avatar_ai2v.run.json"
```

#### 6.2 Realtime path smoke: `streaming_ai2v`

Тот же pipeline, что вызывает worker (`generate_streaming_ai2v`):

```bash
export ARACHNE_INCREMENTAL_WAV2VEC=1

python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode streaming_ai2v \
  --runtime_profile operational \
  --image "$ARACHNE_FACE" \
  --audio "$ARACHNE_ROOT/output/elena_16k.wav" \
  --prompt "Speaking naturally to camera, stable identity." \
  --num_frames 17 \
  --output "$ARACHNE_ROOT/output/avatar_streaming_smoke.mp4"
```

В `.run.json` ожидайте поля `ttff_sec`, `wav2vec_partial_sec`, `wav2vec_full_sec`.

#### 6.3 Таблица режимов `scripts/infer.py`


| `--mode`                   | Веса             | Входы                   | Назначение                       |
| -------------------------- | ---------------- | ----------------------- | -------------------------------- |
| `**ai2v**`                 | merged avatar    | image + audio + prompt  | Оцифровка, QA                    |
| `**streaming_ai2v**`       | merged avatar    | image + audio + prompt  | Realtime micro-turn (как worker) |
| `**at2v**`                 | merged avatar    | audio + prompt          | Без референс-фото                |
| `**avc**`                  | merged avatar    | video + audio + prompt  | Продолжение / смена речи         |
| `**enroll_identity**`      | merged avatar    | image + `--identity_id` | Identity bank `.pt`              |
| `**t2v**`                  | ULTRA-VIDEO only | prompt                  | Базовое text→video               |
| `**i2v**`                  | ULTRA-VIDEO only | image + prompt          | Базовое image→video              |
| `**vc**`                   | ULTRA-VIDEO only | video + prompt          | Базовое continuation             |
| `audio_i2v`, `imagine_i2v` | lab              | —                       | **Не prod**                      |


#### Runtime profiles


| Profile           | Когда                         | Поведение                                      |
| ----------------- | ----------------------------- | ---------------------------------------------- |
| **operational** | Prod realtime, worker default | Chunked denoise, ~12 distill steps, TTFF-first |
| `**cinematic`**   | QA / quality baseline         | Медленнее, выше качество                       |


Rollback monolithic streaming:

```bash
export ARACHNE_LEGACY_STREAMING=1
```

**Если упало:** см. [§D. CLI inference](#d-cli-inference-scriptsinferpy).

---

### Фаза 7 — GPU worker :9090 (prod realtime)

```bash
source "$ARACHNE_ROOT/.venv/bin/activate"
export PYTHONPATH="$ARACHNE_ROOT:$ARACHNE_ROOT/services/arachnex-worker"
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
export ARACHNE_RUNTIME_PROFILE=operational
export ARACHNE_CHUNK_KV=1
export ARACHNE_INCREMENTAL_WAV2VEC=1
export NULLXES_IDENTITY_BANK_PATH="$ARACHNE_ROOT/output/elena_identity_bank.pt"

export ARACHNE_STREAM_MAX_ACTIVE_JOBS=1
export ARACHNE_STREAM_MAX_QUEUE=3
export ARACHNE_STREAM_QUEUE_TIMEOUT_SEC=15

# Рекомендуется prod:
# export NULLXES_INFERENCE_SERVICE_KEY="<secret>"

tmux new -s arachne-worker
cd "$ARACHNE_ROOT/services/arachnex-worker"
uvicorn main:app --host 0.0.0.0 --port 9090
```

RunPod: порт **9090** должен быть в **Expose HTTP ports** (см. [RunPod UI](#runpod-ui--pod--edit--save-сделать-до-ssh)). Публичный URL: `https://<pod-id>-9090.proxy.runpod.net`.


| Method | Path                         | Назначение                                 |
| ------ | ---------------------------- | ------------------------------------------ |
| GET    | `/health`                    | Liveness, queue, VRAM (**не грузит** веса) |
| GET    | `/v1/runtime/metrics`        | Rejects, wait times (ключ если задан)      |
| POST   | `/v1/realtime/avatar_frames` | **NDJSON RGB** (PCM16 in)                  |
| POST   | `/v1/arachne/generate`       | Sync MP4                                   |
| POST   | `/v1/infer/jobs`             | Async MP4 queue                            |
| POST   | `/v1/admin/drain`            | Stop admit (ключ)                          |
| POST   | `/v1/admin/activate`         | Resume admit (ключ)                        |


Smoke:

```bash
curl -fsS http://127.0.0.1:9090/health | jq .
```

**Если упало:** см. [§E. GPU worker](#e-gpu-worker-arachnex-worker).

---

### Фаза 8 — Orchestrator (отдельный CPU-хост)

> Зависимости orchestrator — **отдельная установка** (шаг E). Не путать с GPU pod (шаги A–D).  
> См. [Зависимости: установка по шагам](#зависимости-установка-по-шагам-важно) → «Установка на orchestrator».

**Рекомендуется:** свой venv на CPU-сервере, не тот же процесс что DiT на GPU.

```bash
cd /path/to/ARACHNE-X
python3 -m venv .venv-orch
source .venv-orch/bin/activate
pip install -U pip setuptools wheel

# Шаг E — orchestrator (внутри уже тянет requirements_avatar.txt для импортов)
pip install -r requirements_orchestrator.txt

export PYTHONPATH="$PWD"
python -c "import aiohttp; from src.server.webrtc_server import create_app; print('ORCH OK')"
```

Опционально Qwen TTS (отдельный процесс, не на GPU worker):

```bash
pip install -r requirements-tts.txt
```

Переменные:

```bash
export PYTHONPATH="$ARACHNE_ROOT"
export NULLXES_AVATAR_INFERENCE_URL=https://<pod-id>-9090.proxy.runpod.net
export NULLXES_AVATAR_INFERENCE_SERVICE_KEY=<same-as-worker-secret>
# Пул воркеров (опционально):
# export NULLXES_AVATAR_WORKER_URLS=https://pod-a:9090,https://pod-b:9090

export NULLXES_PUBLIC_HTTP_BASE=https://your-gateway.example.com
export NULLXES_PUBLIC_WS_BASE=wss://your-gateway.example.com
# export NULLXES_REALTIME_SERVICE_KEY=<gateway-auth>
```

Запуск aiohttp gateway:

```bash
cd "$ARACHNE_ROOT"
python - <<'PY'
import os
from aiohttp import web
from src.server.webrtc_server import create_app
port = int(os.environ.get("NULLXES_HTTP_PORT", "8080"))
web.run_app(create_app(), host="0.0.0.0", port=port)
PY
```

Канонический realtime flow:

```text
WebSocket → realtime_api.py → session_worker.py → realtime_avatar_loop.py
  → avatar_stream_client.py → POST /v1/realtime/avatar_frames
  → avatar.stream.chunk (protocolVersion: v1)
```

TTS: `src/server/tts_runner.py` — **только здесь**, не в worker.

**Если упало:** см. [§F. Orchestrator](#f-orchestrator-srcserver).

---

### Фаза 9 — Приёмочный smoke (записать в ops log)


| #   | Check               | Command                              | Pass                         |
| --- | ------------------- | ------------------------------------ | ---------------------------- |
| 1   | GPU                 | `nvidia-smi`                         | OK                           |
| 2   | Flash               | `import flash_attn`                  | FLASH OK                     |
| 3   | Weights             | Phase 3 script                       | `missing: none`              |
| 4   | CLI ai2v            | Phase 6.1                            | MP4 + `.run.json`            |
| 5   | TTFF                | `jq .sampling_metrics.ttff_sec`      | ≤ ~4s operational            |
| 6   | Incremental wav2vec | `wav2vec_partial_sec` in `.run.json` | partial < full               |
| 7   | Worker health       | `curl /health`                       | `gpuVisible: true`           |
| 8   | Metrics             | `curl /v1/runtime/metrics`           | queue fields                 |
| 9   | Overload            | 4+ parallel NDJSON POST              | `503 worker_busy`            |
| 10  | E2E WS              | orchestrator → worker                | `avatar.stream.chunk` frames |


---

## Справочник переменных окружения

### Веса и пути


| Variable                     | Required                | Description                                |
| ---------------------------- | ----------------------- | ------------------------------------------ |
| `NULLXES_CHECKPOINT_DIR`     | **Yes** (worker/avatar) | Merged runtime (`arachne-avatar-runtime`)  |
| `ARACHNE_CHECKPOINT_DIR`     | Alt                     | То же                                      |
| `NULLXES_IDENTITY_BANK_PATH` | Recommended             | `.pt` identity bank                        |
| `PYTHONPATH`                 | **Yes**                 | `$ARACHNE_ROOT` (+ worker dir для uvicorn) |
| `HF_TOKEN`                   | Download only           | `export HF_TOKEN="hf_..."` — never commit  |


### Sampling / Stability


| Variable                      | Default       | Description                    |
| ----------------------------- | ------------- | ------------------------------ |
| `ARACHNE_RUNTIME_PROFILE`     | `operational` | `operational`                  |
| `ARACHNE_LEGACY_STREAMING`    | off           | `1` = monolithic rollback      |
| `ARACHNE_CHUNK_KV`            | off           | `1` = cross-chunk KV           |
| `ARACHNE_INCREMENTAL_WAV2VEC` | `1`           | Partial wav2vec before chunk-0 |
| `ARACHNE_INFER_ENABLE_BSA`    | off           | Sparse attn infer-only         |


### Worker queue


| Variable                           | Default | Description              |
| ---------------------------------- | ------- | ------------------------ |
| `ARACHNE_STREAM_MAX_ACTIVE_JOBS`   | `1`     | Concurrent GPU streams   |
| `ARACHNE_STREAM_MAX_QUEUE`         | `3`     | Wait slots before reject |
| `ARACHNE_STREAM_QUEUE_TIMEOUT_SEC` | `15`    | Max queue wait           |
| `ARACHNE_STREAM_ESTIMATED_JOB_MS`  | `8000`  | `retryAfterMs` hint      |
| `INFERENCE_MAX_QUEUE`              | `32`    | Async MP4 jobs           |


### Auth


| Variable                               | Header                                    |
| -------------------------------------- | ----------------------------------------- |
| `NULLXES_INFERENCE_SERVICE_KEY`        | `X-NULLXES-Avatar-Inference-Key` (worker) |
| `NULLXES_AVATAR_INFERENCE_SERVICE_KEY` | Alias (orchestrator client)               |
| `NULLXES_REALTIME_SERVICE_KEY`         | Gateway WS/REST                           |


### Orchestrator → worker


| Variable                               | Description                               |
| -------------------------------------- | ----------------------------------------- |
| `NULLXES_AVATAR_INFERENCE_URL`         | Single worker base URL                    |
| `NULLXES_AVATAR_WORKER_URLS`           | Comma-separated pool (`hash(session_id)`) |
| `NULLXES_AVATAR_INFERENCE_FRAMES_PATH` | Default `/v1/realtime/avatar_frames`      |
| `NULLXES_AVATAR_INFERENCE_RETRY_MAX`   | Retries on `503` (default 3)              |


---

## Runbook: что делать, если компонент упал

Используйте порядок: **симптом → диагностика → действие → эскалация**.

---

### A. Инфраструктура (GPU, disk, SSH)


| Симптом                     | Диагностика          | Действие                                                  |
| --------------------------- | -------------------- | --------------------------------------------------------- |
| `nvidia-smi` failed         | Pod без GPU / driver | RunPod UI: GPU H200/H100, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Данные пропали после Edit Pod | данные не на volume | Хранить всё в `/workspace`; Edit Pod = reset контейнера |
| `CUDA not available`        | CPU-only image       | Reinstall torch cu124 wheel (Фаза 4)                      |
| `No space left on device`   | `df -h`              | Удалить `.incomplete`, старые `output/`, расширить volume |
| SSH обрывается mid-download | Нет tmux             | `tmux new -s dl` + `hf download`                          |
| OOM при compile flash-attn  | RAM                  | `MAX_JOBS=4` или `MAX_JOBS=2`                             |


**Эскалация:** snapshot диска, лог `nvidia-smi`, `df -h` → [ceo@nullxes.com](mailto:ceo@nullxes.com)

---

### B. Веса (Hugging Face)


| Симптом                              | Диагностика               | Действие                                                  |
| ------------------------------------ | ------------------------- | --------------------------------------------------------- |
| `401` / `403` на download            | `huggingface-cli whoami`  | `export HF_TOKEN="hf_..."`; access к ULTRA-AVATAR + ULTRA-VIDEO |
| `missing: audio/wav2vec2`            | `ls $CKPT/audio/wav2vec2` | Symlink Phase 2 / 3                                       |
| `missing: avatar_single`             | incomplete shards         | `find *.incomplete`; докачать `hf download`               |
| `KeyError` / shape mismatch при load | чужой checkpoint dir      | Только merged `arachne-avatar-runtime` из NULLXES ULTRA repos |
| 6 shards expected, меньше файлов     | `wc -l` safetensors       | Докачать AVATAR repo                                      |


---

### C. Python / CUDA / FlashAttention


| Симптом                                   | Диагностика             | Действие                                                |
| ----------------------------------------- | ----------------------- | ------------------------------------------------------- |
| `ModuleNotFoundError: flash_attn`         | Linux pod?              | Установить flash-attn **после** torch cu124             |
| flash-attn compile OOM                    | build log               | `MAX_JOBS=4 pip install ...`                            |
| `transformers` version conflict           | `pip show transformers` | Core **4.41.0**; не ставить `requirements-audiodit.txt` |
| Import OK fails on `load_avatar_pipeline` | traceback               | Проверить `NULLXES_CHECKPOINT_DIR`, `PYTHONPATH`        |
| Works on Windows, fails prod              | —                       | **Prod только Linux GPU**                               |


---

### D. CLI inference (`scripts/infer.py`)


| Симптом                    | Диагностика             | Действие                                                                 |
| -------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| CUDA OOM на ai2v           | `nvidia-smi` during run | Закрыть worker; один процесс на GPU; H200                                |
| TTFF очень высокий         | `.run.json`             | `ARACHNE_INCREMENTAL_WAV2VEC=1`, profile `operational`                   |
| Lip sync слабый            | metrics                 | `--audio_guidance_scale 5.0` (prod often 5.0–5.5)                        |
| `num_frames` error         | frame rule              | Frames = **4n+1** (1, 5, 9, 17, …)                                       |
| Black / corrupt MP4        | ffmpeg                  | Проверить исходное аудио из `assets/` → 16 kHz mono (`elena_16k.wav`)   |
| cinematic слишком медленно | profile                 | `--runtime_profile operational`                                          |
| Flicker между chunks       | Stability               | `ARACHNE_CHUNK_KV=1`; смотреть `identity_cosine_per_chunk` в `.run.json` |


Rollback quality path:

```bash
export ARACHNE_RUNTIME_PROFILE=cinematic
export ARACHNE_LEGACY_STREAMING=1
```

---

### E. GPU worker (`arachnex-worker`)


| Симптом                              | HTTP / log            | Действие                                                               |
| ------------------------------------ | --------------------- | ---------------------------------------------------------------------- |
| Worker не стартует                   | uvicorn traceback     | `PYTHONPATH`, `NULLXES_CHECKPOINT_DIR`, Фаза 4                         |
| `/health` 200 но `gpuVisible: false` | health JSON           | CUDA/torch; перезапуск pod                                             |
| `401 Unauthorized`                   | missing header        | `X-NULLXES-Avatar-Inference-Key` = `NULLXES_INFERENCE_SERVICE_KEY`     |
| `**503 worker_busy**`                | queue full            | **Ожидаемо** под нагрузкой; client retry `retryAfterMs`; scale workers |
| `**503 queue_timeout`**              | wait > 15s            | Увеличить `ARACHNE_STREAM_MAX_QUEUE` или добавить GPU pod              |
| `**worker_draining`**                | lifecycle             | `POST /v1/admin/activate` или restart worker                           |
| Первый запрос 60–120s                | cold load             | Warmup: один `ai2v` CLI до prod traffic                                |
| TTS OOM **в worker**                 | process map           | **Убрать TTS из worker** — только orchestrator                         |
| NDJSON обрывается mid-stream         | worker logs           | Check GPU OOM; reduce concurrent jobs to 1                             |
| MP4 jobs stuck                       | `/v1/infer/jobs/{id}` | Queue depth `INFERENCE_MAX_QUEUE`; restart after drain                 |


**Graceful maintenance:**

```bash
curl -X POST http://127.0.0.1:9090/v1/admin/drain \
  -H "X-NULLXES-Avatar-Inference-Key: $NULLXES_INFERENCE_SERVICE_KEY"
# дождаться active=0
# deploy / restart
curl -X POST http://127.0.0.1:9090/v1/admin/activate \
  -H "X-NULLXES-Avatar-Inference-Key: $NULLXES_INFERENCE_SERVICE_KEY"
```

**Масштабирование:** несколько worker URL → `NULLXES_AVATAR_WORKER_URLS` (hash по `session_id`). Redis affinity (A6) — roadmap, пока sticky только через hash.

---

### F. Orchestrator (`src/server`)


| Симптом                        | Диагностика         | Действие                                                     |
| ------------------------------ | ------------------- | ------------------------------------------------------------ |
| WS connect OK, нет видео       | worker URL          | `NULLXES_AVATAR_INFERENCE_URL` reachable с orchestrator host |
| `Connection refused` to worker | curl from orch host | Firewall, RunPod proxy, port 9090 exposed                    |
| Avatar 401                     | key mismatch        | Same key on worker + `NULLXES_AVATAR_INFERENCE_SERVICE_KEY`  |
| Retry storm                    | logs `worker_busy`  | Normal; tune `NULLXES_AVATAR_INFERENCE_RETRY_MAX`; add GPU   |
| TTS fails, avatar OK           | `tts_runner`        | edge-tts network; отдельный TTS provider config              |
| STT fails                      | whisper             | `faster-whisper` model download; CPU RAM                     |
| No `avatar.stream.chunk`       | event name          | Client must handle v1 schema, not legacy `avatar.chunk`      |
| Preview only, no generation    | env                 | `NULLXES_REALTIME_ROUTE_VIA_SESSION_WORKER=1` (default)      |


**Не чинить в prod без review:** shadow playback, legacy HTTP generate aliases, второй orchestrator в `realtime_api.py`.

---

### G. Frontend / gateway (ваш контур)


| Симптом            | Действие                                                             |
| ------------------ | -------------------------------------------------------------------- |
| CORS errors        | `NULLXES_CORS_ORIGIN` на gateway                                     |
| WS URL wrong       | `NULLXES_PUBLIC_WS_BASE`                                             |
| Secrets in browser | **Никогда** не отдавать `NULLXES_INFERENCE_SERVICE_KEY`, RunPod keys |
| HR RTMP audio-only | Отдельный path без DiT — by design                                   |


---

### H. Матрица «кто владеет чем»


| Процесс          | Владеет                             | Не должен владеть |
| ---------------- | ----------------------------------- | ----------------- |
| **GPU worker**   | DiT, wav2vec, VAE decode, NDJSON    | TTS, LLM, Qwen    |
| **Orchestrator** | WS, VAD, STT, LLM, **TTS**, session | GPU weights       |
| **CLI infer**    | Offline MP4, bench                  | —                 |


---

## Диаграмма отказов (упрощённо)

```text
Client WS error
  ├─ 401 gateway → NULLXES_REALTIME_SERVICE_KEY
  ├─ connect fail → PUBLIC_WS_BASE / TLS
  └─ connected, no frames
        ├─ orchestrator log → worker URL / key
        └─ worker 503 → queue / scale GPU
              ├─ worker_busy → retry (normal)
              ├─ queue_timeout → increase queue or GPUs
              └─ 500 / OOM → §E worker, §D infer profile

CLI infer fail (no worker)
  ├─ missing weights → §B
  ├─ flash_attn → §C
  └─ OOM → single process, H200
```

---

## Связанные документы


| Document                                                                       | Purpose                                           |
| ------------------------------------------------------------------------------ | ------------------------------------------------- |
| `[ARCHITECTURE.md](../ARCHITECTURE.md)`                                        | Policy, endpoints, Wave 1 status                  |
| `[NULLXES_ARACHNE_RUNPOD_27-05-2026.md](NULLXES_ARACHNE_RUNPOD_27-05-2026.md)` | Extended RunPod commands                          |
| `[REQUIREMENTS.md](REQUIREMENTS.md)`                                           | Dependency matrix                                 |
| `[services/arachnex-worker/README.md](../services/arachnex-worker/README.md)`  | Worker HTTP contract                              |
| `[Claude_senior.md](../Claude_senior.md)`                                      | Engineering guardrails (prod file classification) |
| `[README.md](../README.md)`                                                    | Project overview                                  |


---

## Handoff checklist (клиент)

1. **RunPod UI:** Edit Pod — image, disks, HTTP/TCP ports (чеклист R1–R8).
2. SSH на GPU pod (Linux), Фаза 0.
3. Git + venv + `assets/` (Фаза 1).
4. **Зависимости GPU pod по шагам A→D** — чеклист D1–D5 (раздел [Зависимости](#зависимости-установка-по-шагам-важно), Фаза 4).
5. Скачать **ULTRA-AVATAR + ULTRA-VIDEO**, merged layout (Фазы 2–3).
6. Smoke **ai2v** + **streaming_ai2v**, сохранить `.run.json` (Фаза 6).
7. Запустить **worker :9090**, `/health` OK (Фаза 7).
8. **Orchestrator на CPU:** отдельный venv, шаг E (Фаза 8).
9. Пройти **Фазу 9** acceptance, записать метрики TTFF.
10. **Не** ставить в prod: `requirements-training.txt`, `requirements-audiodit.txt`, Foundation weights.

Support: **[ceo@nullxes.com](mailto:ceo@nullxes.com)** | Telegram **@MagistrTheOne**

**Unauthorized modification of code or weights is prohibited** — see `[LICENSE](../LICENSE)`.