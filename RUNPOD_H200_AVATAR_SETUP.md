# RunPod H200 — ARACHNE-X AVATAR (оцифровка + тесты режимов CLI)

Playbook для **RunPod GPU H200** (H100 — те же шаги, медленнее): веса → merged runtime → `scripts/infer.py` по режимам.

**Текущий этап:** оцифровка и прогон режимов **только через CLI**. HTTP-воркер (`services/arachnex-worker`) — **не обязателен**; см. §5, когда понадобится realtime NDJSON для HR.


| Артефакт                                  | URL                                                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Исходники                                 | [github.com/MagistrTheOne/ARACHNE-X-NULLXES-](https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git) |
| Веса AVATAR                               | [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR)    |
| Веса VIDEO (база tokenizer/vae/dit + T2V) | [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO)      |


Основной режим оцифровки: **`ai2v`** (image + audio + prompt) → MP4. Остальные режимы — §4.

Связанные документы: [`GTM_ONE_SHOT_DEPLOY.md`](Documentation/DOC_CHECK/GTM_ONE_SHOT_DEPLOY.md), [`RUNPOD_SEMIAUTO_PIPELINE.md`](Documentation/DOC_CHECK/RUNPOD_SEMIAUTO_PIPELINE.md), воркер: [`services/arachnex-worker/README.md`](services/arachnex-worker/README.md).

---

## Сейчас: что делать на pod (по шагам)

Если **AVATAR уже скачан** (~120G, §2.3.1 OK) — не повторяйте §2.3, идите по списку:

| Шаг | Действие | Готово когда |
| --- | -------- | ------------ |
| 1 | §2.2 — скачать **VIDEO** | `tokenizer/`, `vae/` на диске |
| 2 | §2.3.2 — symlink `audio/wav2vec2` в AVATAR | `ls` symlink OK |
| 3 | §2.4 — собрать `weights/arachne-avatar-runtime` | symlinks созданы |
| 4 | §2.5 — layout merged | `missing: none` |
| 5 | §3.1–3.3 (+ 3.5 sanity) — torch, deps, flash-attn | import CUDA OK |
| 6 | §4 — тесты режимов (сначала **4.2 `ai2v` smoke**, потом остальные по желанию) | MP4 на диске |
| 7 | §5 — **пропустить**, пока не нужен HTTP/realtime для NULLXES HR | — |

Подготовьте на pod:

```bash
mkdir -p /workspace/input /workspace/ARACHNE-X/output
# face.png — reference лицо
# speech.wav — mono 16 kHz (или 24 kHz, runtime ресемплит)
```

---

## 0. RunPod pod (H200)

Рекомендации:


| Параметр  | Значение                                                                           |
| --------- | ---------------------------------------------------------------------------------- |
| GPU       | **H200** (80GB+ VRAM)                                                              |
| Disk      | **≥ 200 GB** (оба HF snapshot + venv + cache)                                      |
| Image     | PyTorch 2.6 + CUDA 12.4 (например `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`) |
| Workspace | `/workspace`                                                                       |


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

Дальше **все** команды документа (`pip`, `huggingface-cli`, `python scripts/infer.py`) — только с `source .venv/bin/activate`. `uvicorn` — только в §5 (опционально).

Токен и CLI (при `401/403` или rate limit на HF):

```bash
# всё ещё внутри активированного .venv
# Токен только из секретов RunPod / env — не коммитьте в git
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in pod env or: export HF_TOKEN=hf_...}"
pip install -U "huggingface_hub[cli]>=0.34,<1.0"
huggingface-cli login --token "$HF_TOKEN"
# или: hf auth login --token "$HF_TOKEN"
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

**Рекомендуемый порядок:** §2.3 AVATAR → §2.3.1 проверка → §2.2 VIDEO → §2.3.2 symlink wav2vec → §2.4 merged runtime → §2.5 layout. Только после `missing: none` в §2.5 переходите к §3.

**CLI:** `huggingface-cli download` помечен deprecated — эквивалент:

```bash
hf download <repo_id> --local-dir <path>
```

Предупреждения при скачивании (норма): `local-dir-use-symlinks` ignored; `Still waiting to acquire lock` на `.gitignore.lock`; наложение progress-bar’ов.

### 2.1 Каталоги

```bash
export ARACHNE_ROOT=/workspace/ARACHNE-X
mkdir -p "$ARACHNE_ROOT/weights"
```

### 2.2 VIDEO (обязателен для `tokenizer/`, `vae/`, `text_encoder/`, `scheduler/`)

Скачивать **после** AVATAR (§2.3) и его проверки (§2.3.1), **до** merged runtime (§2.4).

```bash
hf download MagistrTheOne/ARACHNE-X-ULTRA-VIDEO \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"
```

Проверка после VIDEO:

```bash
du -sh "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"
find "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO" -name '*.incomplete' 2>/dev/null | wc -l   # 0
ls "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"
# ожидаете: tokenizer  text_encoder  vae  scheduler  dit  [lora]
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
hf download MagistrTheOne/ARACHNE-X-ULTRA-AVATAR \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"
```

Ожидаемая структура (карточка HF):

```text
weights/ARACHNE-X-ULTRA-AVATAR/
  avatar_single/
  avatar_multi/
  chinese-wav2vec2-base/   # wav2vec (в loader — audio/wav2vec2, см. §2.3.2)
  vocal_separator/
  assets/
```

В логе успешного завершения: `Fetching 34 files: 100%` и `34/34`.

#### 2.3.1 Проверка, что AVATAR докачался

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"

# нет зависших incomplete
find "$NULLXES_CHECKPOINT_DIR" -name '*.incomplete' 2>/dev/null | wc -l
# ожидание: 0

# размер и верхний уровень
du -sh "$NULLXES_CHECKPOINT_DIR"
ls -la "$NULLXES_CHECKPOINT_DIR"

# 34 файла в репо (без .cache)
find "$NULLXES_CHECKPOINT_DIR" -type f ! -path '*/.cache/*' | wc -l
# ожидание: 34

for d in avatar_single avatar_multi chinese-wav2vec2-base vocal_separator assets; do
  [ -d "$NULLXES_CHECKPOINT_DIR/$d" ] && echo "OK  $d" || echo "MISSING $d"
done

# 6 шардов single DiT
ls "$NULLXES_CHECKPOINT_DIR/avatar_single"/diffusion_pytorch_model-*.safetensors 2>/dev/null | wc -l
# ожидание: 6
ls -lh "$NULLXES_CHECKPOINT_DIR/avatar_single/diffusion_pytorch_model.safetensors.index.json"
```

Ориентиры при успехе:

| Метрика | Ожидание |
|---------|----------|
| Размер `ARACHNE-X-ULTRA-AVATAR` | ~**120G** |
| `.incomplete` | **0** |
| Файлов без `.cache` | **34** |
| Шардов `avatar_single` | **6** |

Проверка layout **только на AVATAR** (до VIDEO) — ожидаемо покажет отсутствие базы:

```bash
export PYTHONPATH="$ARACHNE_ROOT"
python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["NULLXES_CHECKPOINT_DIR"])
need = ["tokenizer", "vae", "text_encoder", "scheduler", "avatar_single"]
missing = [p for p in need if not (root / p).is_dir()]
wav2v_alt = root / "chinese-wav2vec2-base"
if not (root / "audio" / "wav2vec2").is_dir() and not wav2v_alt.is_dir():
    missing.append("audio/wav2vec2 or chinese-wav2vec2-base")
print("checkpoint:", root)
print("missing:", missing or "none")
if missing == ["tokenizer", "vae", "text_encoder", "scheduler"]:
    print("OK — AVATAR полный; дальше §2.2 VIDEO + §2.4 merged runtime")
if wav2v_alt.is_dir() and not (root / "audio" / "wav2vec2").is_dir():
    print("note: сделайте §2.3.2 symlink перед inference")
PY
```
##2.3.1.1
1. СТАВИМ BUILD TOOLS
apt update && apt install -y ninja-build build-essential cmake gcc g++ git
2. АКТИВИРУЕМ VENV
cd /workspace/ARACHNE-X && source .venv/bin/activate
3. ОБНОВЛЯЕМ BUILD STACK
pip install -U pip setuptools wheel packaging ninja
4. СТАВИМ BUILD DEPS РУКАМИ 
pip install psutil numpy einops packaging
5. ПРОВЕРЯЕМ CUDA 
nvcc --version
6. ПРОВЕРЯЕМ TORCH CUDA 
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

Должно быть:

2.6.0+cu124
True
7. ТЕПЕРЬ FLASH-ATTN 😈
ВАЖНО:

без build isolation.

pip install flash-attn==2.7.4.post1 --no-build-isolation
#### 2.3.2 Symlink `audio/wav2vec2` (после AVATAR)

`loader` ищет `audio/wav2vec2`; в snapshot wav2vec лежит в `chinese-wav2vec2-base/`:

```bash
CKPT="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"
mkdir -p "$CKPT/audio"
ln -sfn "$CKPT/chinese-wav2vec2-base" "$CKPT/audio/wav2vec2"
ls -la "$CKPT/audio/wav2vec2"
```

Тот же symlink повторится в merged runtime (§2.4).

### 2.4 Единый runtime bundle для воркера (`NULLXES_CHECKPOINT_DIR`)

`arachne_x.loader` требует **один** каталог с `tokenizer/`, `vae/`, `scheduler/`, `text_encoder/`, `avatar_single/`, `audio/wav2vec2/`.

Если **AVATAR** snapshot уже содержит `tokenizer/` + `vae/` — используйте его напрямую:

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"
```

Иначе (типичный случай после §2.2 + §2.3) соберите **merged runtime**:

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
mkdir -p "$NULLXES_CHECKPOINT_DIR"

# базовые компоненты из VIDEO
for d in tokenizer text_encoder vae scheduler; do
  ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO/$d" "$NULLXES_CHECKPOINT_DIR/$d"
done

# avatar DiT + audio из AVATAR
for d in avatar_single avatar_multi vocal_separator; do
  if [ -d "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/$d" ]; then
    ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/$d" "$NULLXES_CHECKPOINT_DIR/$d"
  fi
done

# wav2vec из chinese-wav2vec2-base
mkdir -p "$NULLXES_CHECKPOINT_DIR/audio"
ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/chinese-wav2vec2-base" \
  "$NULLXES_CHECKPOINT_DIR/audio/wav2vec2"
```

Для всех следующих шагов (§3–§5) экспортируйте merged path:

```bash
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
```

### 2.5 Проверка layout merged bundle (обязательно перед §3)

```bash
cd "$ARACHNE_ROOT"
source .venv/bin/activate
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
export PYTHONPATH="$ARACHNE_ROOT"

python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["NULLXES_CHECKPOINT_DIR"])
need = ["tokenizer", "vae", "text_encoder", "scheduler", "avatar_single", "audio/wav2vec2"]
missing = [p for p in need if not (root / p).exists()]
print("checkpoint:", root)
print("missing:", missing or "none — OK, можно §3 torch + infer")
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

### 3.4 HTTP worker — **не на текущем этапе** (опционально)

Ставить только перед §5:

```bash
pip install -r services/arachnex-worker/requirements.txt
```

### 3.5 Sanity CUDA

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

## 4. Оцифровка и тесты режимов (CLI)

Общие env перед любым прогоном:

```bash
cd "$ARACHNE_ROOT"
source .venv/bin/activate
export PYTHONPATH="$ARACHNE_ROOT"
export NULLXES_CHECKPOINT_DIR="${NULLXES_CHECKPOINT_DIR:-$ARACHNE_ROOT/weights/arachne-avatar-runtime}"
mkdir -p output
```

Аудио: **`--audio`** (WAV на диске) **или** **`--speak_text`** (TTS внутри пайплайна; нужен `requirements-tts.txt`). Явный `--audio` имеет приоритет.

### 4.1 Таблица режимов (`scripts/infer.py --mode`)

| Режим | Checkpoint | Обязательные входы | Выход | Когда тестировать |
| ----- | ------------ | ------------------ | ----- | ----------------- |
| **`ai2v`** | merged avatar (§2.4) | `--image`, `--audio` или `--speak_text`, `--prompt` | MP4 | **Первым** — основная оцифровка |
| `streaming_ai2v` | merged avatar | то же + длинный WAV (чанки) | MP4 (покадровая сборка) | После `ai2v`; проверка micro-turn |
| `at2v` | merged avatar | `--audio` или `--speak_text`, `--prompt` (без лица) | MP4 | Генерация «с нуля» под аудио |
| `avc` | merged avatar | `--video`, `--audio` или `--speak_text`, `--prompt` | MP4 | Редакт/континуэйшн по видео |
| `enroll_identity` | merged avatar | `--image`, `--identity_id`, bank path | bank file | Перед identity-guided `ai2v` |
| `t2v` | **VIDEO** dir (§2.2) | `--prompt` | MP4 | Опционально; базовое видео |
| `i2v` | VIDEO | `--image`, `--prompt` | MP4 | Опционально |
| `vc` | VIDEO | `--video`, `--prompt` | MP4 | Опционально |

Для §4.6–4.8: `NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"`.  
Для §4.9: `--checkpoint_dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"`.

### 4.1.1 Промпты (рекомендации)

| Поле | Содержание |
| ---- | ---------- |
| `--prompt` | Кто в кадре, освещение, «speaking to camera», **stable identity**, синхрон губ/челюсти, минимальные движения головы |
| `--negative_prompt` | `anime, cartoon, blurry, distorted face, extra fingers, teeth artifact, watermark` |

Пресет в репо: [`assets/avatar/single/kaira/kaira.json`](assets/avatar/single/kaira/kaira.json) — поле `prompt` + `_arachne_x_infer` (steps/frames для smoke).

Smoke-параметры (быстро на H200): `--num_frames 17`, `--num_inference_steps 2`, `--text_guidance_scale 3`, `--audio_guidance_scale 3`. Качество: steps 25–50, frames 93, resolution `480p`/`720p`.

### 4.2 `ai2v` — image + audio + prompt (основной)

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --prompt "Executive digital human speaking naturally to camera, stable identity, cinematic office lighting, photorealistic." \
  --negative_prompt "anime, cartoon, blurry, distorted face, extra fingers, open mouth teeth artifact" \
  --image /workspace/input/face.png \
  --audio /workspace/input/speech.wav \
  --height 480 --width 480 \
  --num_frames 17 \
  --num_inference_steps 2 \
  --text_guidance_scale 3.0 \
  --audio_guidance_scale 3.0 \
  --output output/avatar_ai2v_smoke.mp4
```

KAIRA smoke (промпт из JSON):

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --prompt "$(jq -r .prompt assets/avatar/single/kaira/kaira.json)" \
  --negative_prompt "anime, cartoon, blurry, distorted face" \
  --image assets/avatar/single/kaira/kaira.png \
  --audio /workspace/input/speech.wav \
  --num_frames 17 \
  --num_inference_steps 2 \
  --output output/kaira_ai2v_smoke.mp4
```

ELENA — production (из `elena.json` → `_arachne_x_infer`: 720p, 181f, 30 steps):

```bash
PRESET=assets/avatar/single/elena/elena.json
ffmpeg -y -i "$(jq -r .cond_audio "$PRESET")" -ar 16000 -ac 1 /workspace/input/elena_16k.wav

python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --prompt "$(jq -r .prompt "$PRESET")" \
  --negative_prompt "$(jq -r .negative_prompt "$PRESET")" \
  --image "$(jq -r .cond_image "$PRESET")" \
  --audio /workspace/input/elena_16k.wav \
  --resolution "$(jq -r '._arachne_x_infer.resolution' "$PRESET")" \
  --num_frames "$(jq -r '._arachne_x_infer.num_frames' "$PRESET")" \
  --num_inference_steps "$(jq -r '._arachne_x_infer.num_inference_steps' "$PRESET")" \
  --text_guidance_scale "$(jq -r '._arachne_x_infer.text_guidance_scale' "$PRESET")" \
  --audio_guidance_scale "$(jq -r '._arachne_x_infer.audio_guidance_scale' "$PRESET")" \
  --output output/elena_ai2v_production.mp4
```

ELENA smoke (`_arachne_x_infer_smoke` в том же JSON):

```bash
PRESET=assets/avatar/single/elena/elena.json
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --prompt "$(jq -r .prompt "$PRESET")" \
  --negative_prompt "$(jq -r .negative_prompt "$PRESET")" \
  --image "$(jq -r .cond_image "$PRESET")" \
  --audio /workspace/input/speech.wav \
  --resolution "$(jq -r '._arachne_x_infer_smoke.resolution' "$PRESET")" \
  --num_frames "$(jq -r '._arachne_x_infer_smoke.num_frames' "$PRESET")" \
  --num_inference_steps "$(jq -r '._arachne_x_infer_smoke.num_inference_steps' "$PRESET")" \
  --text_guidance_scale "$(jq -r '._arachne_x_infer_smoke.text_guidance_scale' "$PRESET")" \
  --audio_guidance_scale "$(jq -r '._arachne_x_infer_smoke.audio_guidance_scale' "$PRESET")" \
  --output output/elena_ai2v_smoke.mp4
```

`ai2v` + TTS вместо WAV (нужен Qwen TTS в venv, см. `requirements-tts.txt`):

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --image /workspace/input/face.png \
  --speak_text "Здравствуйте, это тестовая фраза для оцифровки аватара." \
  --tts_provider qwen \
  --tts_language Russian \
  --tts_speaker Ryan \
  --prompt "..." \
  --num_frames 17 --num_inference_steps 2 \
  --output output/avatar_ai2v_tts.mp4
```

### 4.3 `streaming_ai2v` — тот же контур, чанки аудио

Проверяет micro-turn по `--audio_chunk_sec` (default см. CLI). Полезно перед realtime HTTP, но **без воркера**:

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode streaming_ai2v \
  --image /workspace/input/face.png \
  --audio /workspace/input/speech_long.wav \
  --prompt "Speaking naturally to camera, stable identity." \
  --audio_chunk_sec 2.0 \
  --num_frames 17 \
  --num_inference_steps 2 \
  --output output/avatar_streaming_smoke.mp4
```

### 4.4 `at2v` — audio + prompt (без reference image)

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode at2v \
  --audio /workspace/input/speech.wav \
  --prompt "Professional presenter, neutral background, speaking to camera." \
  --height 480 --width 832 \
  --num_frames 17 --num_inference_steps 2 \
  --output output/avatar_at2v_smoke.mp4
```

### 4.5 `avc` — video + audio + prompt (variant `multi`)

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode avc \
  --video /workspace/input/reference_clip.mp4 \
  --audio /workspace/input/speech.wav \
  --prompt "Same person, lip sync to new audio, stable identity." \
  --num_cond_frames 13 \
  --num_frames 17 --num_inference_steps 2 \
  --output output/avatar_avc_smoke.mp4
```

### 4.6 `enroll_identity` — банк лица (опционально)

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode enroll_identity \
  --image /workspace/input/face.png \
  --identity_id 1 \
  --identity_bank_save_path output/kaira_identity_bank.pt
```

Дальше в `ai2v` / `streaming_ai2v`:

```bash
  --identity_bank_path output/kaira_identity_bank.pt \
  --identity_id 1 --identity_strength 1.0
```

### 4.7 Identity / emotion / LoRA (доп. флаги к `ai2v`)

```bash
# emotion (если поддерживается чекпоинтом)
--emotion_id neutral --emotion_intensity 0.3 --emotion_guidance_scale 1.0

# LoRA после train_lora_avatar.py
--lora_path /path/to/lora.safetensors --lora_key train
```

### 4.8 Чеклист прогона режимов (текущий этап)

| # | Режим | Артефакт | ☐ |
| - | ----- | -------- | - |
| 1 | `ai2v` + свой face + speech.wav | `output/avatar_ai2v_smoke.mp4` | ☐ |
| 2 | `ai2v` + KAIRA preset | `output/kaira_ai2v_smoke.mp4` | ☐ |
| 3 | `streaming_ai2v` (длинный WAV) | `output/avatar_streaming_smoke.mp4` | ☐ |
| 4 | `at2v` | `output/avatar_at2v_smoke.mp4` | ☐ |
| 5 | `avc` (если есть reference video) | `output/avatar_avc_smoke.mp4` | ☐ |
| 6 | `enroll_identity` + `ai2v` с bank | bank + MP4 | ☐ |

### 4.9 Базовое VIDEO (`t2v` / `i2v` / `vc`) — опционально

Отдельный checkpoint, **не** merged avatar:

```bash
export VIDEO_CKPT="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"

python scripts/infer.py --checkpoint_dir "$VIDEO_CKPT" --mode t2v \
  --prompt "Cinematic drone shot over city at dusk." \
  --num_frames 49 --num_inference_steps 25 \
  --output output/video_t2v_smoke.mp4
```

---

## 5. `arachnex-worker` (HTTP) — **позже, не текущий этап**

Когда CLI-оцифровка стабильна и нужен NDJSON/realtime для NULLXES HR — §3.4 + запуск:

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


| #   | Проверка                                                    | OK  |
| --- | ----------------------------------------------------------- | --- |
| 1   | `nvidia-smi` → H200                                         | ☐   |
| 2   | AVATAR: ~120G, 34 files, 0 incomplete (§2.3.1)              | ☐   |
| 3   | VIDEO скачан, `tokenizer/` + `vae/` (§2.2)                  | ☐   |
| 4   | `arachne-avatar-runtime` merged, §2.5 `missing: none`       | ☐   |
| 5   | `.venv` активен, `hf auth whoami` / `huggingface-cli whoami` | ☐   |
| 6   | `torch` + `flash_attn` import (§3)                          | ☐   |
| 7   | `ai2v` smoke → MP4 (§4.2)                                   | ☐   |
| 8   | Прогон §4.8 (хотя бы 1–3)                                   | ☐   |
| 9   | Worker `/health` + NDJSON (§5) — **только когда нужен HTTP** | ☐   |


---

## 7. Типовые ошибки


| Симптом                              | Действие                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `missing tokenizer,vae,...` на **только AVATAR** | Норма — §2.2 VIDEO + §2.4 merged runtime                         |
| `missing ...` на **merged** после §2.4 | Проверьте symlink’и VIDEO/AVATAR, §2.3.2 wav2vec                  |
| `huggingface-cli: command not found` | `source .venv/bin/activate` (§1.1); используйте `hf download`                |
| Lock / `.incomplete` при download    | Дождаться; не запускать два `hf download` в один `--local-dir`               |
| `flash_attn` build fail              | Используйте prebuilt wheel под CUDA 12.4 / образ RunPod с готовым flash-attn |
| OOM на H200                          | Уменьшите `num_frames`, resolution `480p`, `num_inference_steps`             |
| Worker 401                           | Заголовок `X-NULLXES-Avatar-Inference-Key` = `NULLXES_INFERENCE_SERVICE_KEY` |
| Первый запрос 60–180 s               | Норма: lazy load pipeline на GPU                                             |


---

## 8. Переменные окружения (кратко)


| Переменная                      | Назначение                                   |
| ------------------------------- | -------------------------------------------- |
| `HF_TOKEN`                      | Доступ к private/gated HF repos              |
| `NULLXES_CHECKPOINT_DIR`        | Корень весов для avatar pipeline             |
| `ARACHNE_CHECKPOINT_DIR`        | Алиас                                        |
| `PYTHONPATH`                    | `$ARACHNE_ROOT` + `services/arachnex-worker` |
| `NULLXES_INFERENCE_SERVICE_KEY` | Опциональный секрет HTTP                     |
| `NULLXES_URL`                   | Base URL для smoke script                    |


---

*Документ: RunPod H200 — CLI оцифровка и тесты режимов; HTTP worker — §5 (опционально).*