# RunPod H200 — ARACHNE-X AVATAR (оцифровка: картинка + аудио + промпт)

Пошаговый playbook для поднятия **NULLXES Inference Worker** ([`arachnex-worker`](./)) на **RunPod GPU H200** (H100 — те же шаги, медленнее).

| Артефакт | URL |
|----------|-----|
| Исходники | [github.com/MagistrTheOne/ARACHNE-X-NULLXES-](https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git) |
| Веса AVATAR | [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) |
| Веса VIDEO (база tokenizer/vae/dit + T2V) | [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) |

Режим оцифровки: **`ai2v`** (audio + image + prompt) → MP4 или NDJSON realtime через HTTP.

Связанные документы: [`GTM_ONE_SHOT_DEPLOY.md`](../../Documentation/DOC_CHECK/GTM_ONE_SHOT_DEPLOY.md), [`RUNPOD_SEMIAUTO_PIPELINE.md`](../../Documentation/DOC_CHECK/RUNPOD_SEMIAUTO_PIPELINE.md).

---

## 0. RunPod pod (H200)

Рекомендации:

| Параметр | Значение |
|----------|----------|
| GPU | **H200** (80GB+ VRAM) |
| Disk | **≥ 200 GB** (оба HF snapshot + venv + cache) |
| Image | PyTorch 2.6 + CUDA 12.4 (например `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`) |
| Workspace | `/workspace` |

После старта pod:

```bash
nvidia-smi
python3 --version   # 3.10.x
```

---

## 1. Клон репозитория

```bash
cd /workspace
git clone https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git ARACHNE-X
cd /workspace/ARACHNE-X
git fetch origin
# для патча с arachnex-worker:
# git checkout arachne-last-patch
```

Корень далее: `ARACHNE_ROOT=/workspace/ARACHNE-X`.

### 1.1 Виртуальное окружение + Hugging Face (только в `.venv`)

**Сразу после клона** создаём один venv в корне репозитория. Токен HF и CLI ставим **только туда**, не в system Python pod'а — так не смешиваются версии `pip`/`huggingface_hub` с образом RunPod.

```bash
export ARACHNE_ROOT=/workspace/ARACHNE-X
cd "$ARACHNE_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

Дальше **все** команды документа (`pip`, `huggingface-cli`, `python scripts/infer.py`, `uvicorn`) — только с `source .venv/bin/activate`.

Токен и CLI (при `401/403` или rate limit на HF):

```bash
# всё ещё внутри активированного .venv
export HF_TOKEN=hf_xxxxxxxx
pip install -U "huggingface_hub[cli]>=0.34,<1.0"
huggingface-cli login --token "$HF_TOKEN"
```

Ускорение скачивания (опционально, тоже в `.venv`):

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Проверка:

```bash
which python    # должен указывать на .../ARACHNE-X/.venv/bin/python
which huggingface-cli
huggingface-cli whoami
```

---

## 2. Скачивание весов (HF Hub API / CLI)

**Перед командами:** `cd "$ARACHNE_ROOT" && source .venv/bin/activate` (§1.1).

### 2.1 Каталоги

```bash
export ARACHNE_ROOT=/workspace/ARACHNE-X
mkdir -p "$ARACHNE_ROOT/weights"
```

### 2.2 VIDEO (обязателен для полного layout `tokenizer/`, `vae/`, `dit/`)

```bash
huggingface-cli download MagistrTheOne/ARACHNE-X-ULTRA-VIDEO \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO" \
  --local-dir-use-symlinks False
```

Ожидаемая структура:

```text
weights/ARACHNE-X-ULTRA-VIDEO/
  tokenizer/
  text_encoder/
  vae/
  scheduler/
  dit/
  lora/          # опционально для T2V distill
```

### 2.3 AVATAR (avatar DiT + audio conditioning)

```bash
huggingface-cli download MagistrTheOne/ARACHNE-X-ULTRA-AVATAR \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR" \
  --local-dir-use-symlinks False
```

Ожидаемая структура (карточка HF):

```text
weights/ARACHNE-X-ULTRA-AVATAR/
  avatar_single/
  avatar_multi/
  audio/                 # wav2vec2, vocal_separator (см. loader WeightsLayout)
  chinese-wav2vec2-base/  # legacy path на части snapshot
```

### 2.4 Единый runtime bundle для воркера (`NULLXES_CHECKPOINT_DIR`)

`arachne_x.loader` требует **один** каталог с `tokenizer/`, `vae/`, `scheduler/`, `text_encoder/`, `avatar_single/`, `audio/wav2vec2/`.

Если **AVATAR** snapshot уже содержит `tokenizer/` + `vae/` — используйте его напрямую:

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"
```

Иначе соберите **merged runtime** (VIDEO base + AVATAR avatar/audio) — §2.4:

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
mkdir -p "$NULLXES_CHECKPOINT_DIR"

# базовые компоненты из VIDEO
for d in tokenizer text_encoder vae scheduler; do
  ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO/$d" "$NULLXES_CHECKPOINT_DIR/$d"
done

# avatar DiT + audio из AVATAR
for d in avatar_single avatar_multi audio; do
  if [ -d "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/$d" ]; then
    ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/$d" "$NULLXES_CHECKPOINT_DIR/$d"
  fi
done

# если wav2vec лежит в chinese-wav2vec2-base:
if [ -d "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/chinese-wav2vec2-base" ] \
   && [ ! -d "$NULLXES_CHECKPOINT_DIR/audio/wav2vec2" ]; then
  mkdir -p "$NULLXES_CHECKPOINT_DIR/audio"
  ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/chinese-wav2vec2-base" \
    "$NULLXES_CHECKPOINT_DIR/audio/wav2vec2"
fi
```

### 2.5 Проверка layout (обязательно)

```bash
cd "$ARACHNE_ROOT"
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$ARACHNE_ROOT"

python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["NULLXES_CHECKPOINT_DIR"])
need = ["tokenizer", "vae", "text_encoder", "scheduler", "avatar_single"]
missing = [p for p in need if not (root / p).is_dir()]
audio_w2v = root / "audio" / "wav2vec2"
if not audio_w2v.is_dir():
    missing.append("audio/wav2vec2")
print("checkpoint:", root)
print("missing:", missing or "none — OK")
PY
```

Альтернатива через Hub (без предварительного `local-dir`):

```bash
python - <<'PY'
from arachne_x.weights_resolve import resolve_weights_root
p = resolve_weights_root("MagistrTheOne/ARACHNE-X-ULTRA-AVATAR", allow_hub=True)
print("resolved:", p)
PY
```

(`allow_hub=True` только если локального bundle нет.)

---

## 3. Зависимости inference (тот же `.venv`)

Доставляем torch/flash-attn и стек ARACHNE-X в **уже созданный** `.venv` из §1.1. Новый venv не создаём. Не смешивать с `.venv_stage3` Qwen — см. semiauto doc.

```bash
cd "$ARACHNE_ROOT"
source .venv/bin/activate
```

### 3.1 PyTorch (CUDA 12.4)

```bash
pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

### 3.2 Core + avatar extras

```bash
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r requirements_avatar.txt
```

### 3.3 FlashAttention (Linux H200)

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### 3.4 HTTP worker (тонкий слой)

```bash
pip install -r services/arachnex-worker/requirements.txt
```

### 4.5 Sanity CUDA

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn:", e)
PY
```

---

## 4. Оцифровка аватара (CLI) — режим `ai2v`

**Вход:** reference image + WAV/PCM audio + text prompt.  
**Выход:** MP4.

### 4.1 Пресет KAIRA (в репо)

Файл: [`assets/avatar/single/kaira/kaira.json`](../../assets/avatar/single/kaira/kaira.json)  
Изображение: `assets/avatar/single/kaira/kaira.png`

### 4.2 Пример: свой image + audio + prompt

```bash
cd "$ARACHNE_ROOT"
source .venv/bin/activate
export PYTHONPATH="$ARACHNE_ROOT"
export NULLXES_CHECKPOINT_DIR="${NULLXES_CHECKPOINT_DIR:-$ARACHNE_ROOT/weights/arachne-avatar-runtime}"

# Подготовьте:
#   /workspace/input/face.png
#   /workspace/input/speech.wav   # mono, 16 kHz или 24 kHz — runtime ресемплит при необходимости

python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --prompt "Executive digital human speaking naturally to camera, stable identity, cinematic office lighting, photorealistic." \
  --negative_prompt "anime, cartoon, blurry, distorted face, extra fingers, open mouth teeth artifact" \
  --image /workspace/input/face.png \
  --audio /workspace/input/speech.wav \
  --height 480 --width 480 \
  --num_frames 17 \
  --num_inference_steps 4 \
  --text_guidance_scale 3.0 \
  --audio_guidance_scale 3.0 \
  --output /workspace/ARACHNE-X/output/avatar_ai2v_smoke.mp4
```

Режимы avatar (полный список CLI): `ai2v`, `at2v`, `avc`, `streaming_ai2v` — для **картинка+аудио+промпт** используйте **`ai2v`**.

Лёгкий smoke (мало шагов, быстрее на H200):

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --prompt "$(jq -r .prompt assets/avatar/single/kaira/kaira.json)" \
  --image assets/avatar/single/kaira/kaira.png \
  --audio /workspace/input/speech.wav \
  --num_frames 17 \
  --num_inference_steps 2 \
  --output /workspace/ARACHNE-X/output/kaira_ai2v_smoke.mp4
```

---

## 5. Запуск `arachnex-worker` (HTTP, realtime NDJSON)

```bash
cd "$ARACHNE_ROOT"
source .venv/bin/activate
export ARACHNE_ROOT
export NULLXES_CHECKPOINT_DIR
export PYTHONPATH="$ARACHNE_ROOT:$ARACHNE_ROOT/services/arachnex-worker"

# опционально:
# export NULLXES_INFERENCE_SERVICE_KEY=your-secret

cd services/arachnex-worker
uvicorn main:app --host 0.0.0.0 --port 9090
```

### 5.1 Health

```bash
curl -fsS http://127.0.0.1:9090/health
```

### 5.2 Warmup + NDJSON smoke

```bash
export NULLXES_URL=http://127.0.0.1:9090
export NULLXES_SMOKE_ENGINE=arachne_ultra_avatar
# export X_NULLXES_KEY=...   # если задан NULLXES_INFERENCE_SERVICE_KEY

bash "$ARACHNE_ROOT/scripts/gpu/smoke_avatar_frames.sh"
```

Тело запроса worker: `imageBase64` + `audioFloat32Base64` + `prompt` — тот же контур **image+audio+prompt**, что и CLI `ai2v`.

---

## 6. Чеклист «готово»

| # | Проверка | OK |
|---|----------|-----|
| 1 | `nvidia-smi` → H200 | ☐ |
| 2 | Оба HF snapshot на диске | ☐ |
| 3 | `NULLXES_CHECKPOINT_DIR` — layout без `missing` | ☐ |
| 4 | `.venv` активен, `huggingface-cli whoami` | ☐ |
| 5 | `torch` + `flash_attn` import | ☐ |
| 6 | `scripts/infer.py --mode ai2v` → MP4 | ☐ |
| 7 | `/health` + `smoke_avatar_frames.sh` → NDJSON | ☐ |

---

## 7. Типовые ошибки

| Симптом | Действие |
|---------|----------|
| `missing tokenizer/ or vae/` | Соберите merged runtime (§2.4) или скачайте полный AVATAR bundle |
| `huggingface-cli: command not found` | `source .venv/bin/activate` (§1.1) |
| `flash_attn` build fail | Используйте prebuilt wheel под CUDA 12.4 / образ RunPod с готовым flash-attn |
| OOM на H200 | Уменьшите `num_frames`, resolution `480p`, `num_inference_steps` |
| Worker 401 | Заголовок `X-NULLXES-Avatar-Inference-Key` = `NULLXES_INFERENCE_SERVICE_KEY` |
| Первый запрос 60–180 s | Норма: lazy load pipeline на GPU |

---

## 8. Переменные окружения (кратко)

| Переменная | Назначение |
|------------|------------|
| `HF_TOKEN` | Доступ к private/gated HF repos |
| `NULLXES_CHECKPOINT_DIR` | Корень весов для avatar pipeline |
| `ARACHNE_CHECKPOINT_DIR` | Алиас |
| `PYTHONPATH` | `$ARACHNE_ROOT` + `services/arachnex-worker` |
| `NULLXES_INFERENCE_SERVICE_KEY` | Опциональный секрет HTTP |
| `NULLXES_URL` | Base URL для smoke script |

---

*Документ: `services/arachnex-worker` — RunPod H200 avatar digitization playbook. Worker renamed from `longcat-worker` → `arachnex-worker`.*
