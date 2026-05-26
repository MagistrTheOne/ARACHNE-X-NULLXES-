# NULLXES × ARACHNE-X-ULTRA — Runbook развёртывания аватаров для Job.ai

| Поле | Значение |
|------|----------|
| **Дата** | 21.05.2026 |
| **Версия репо** | ветка `arachne-last-patch` |
| **Репозиторий** | https://github.com/MagistrTheOne/ARACHNE-X-NULLXES- |
| **Контакт** | ceo@nullxes.com |
| **Целевая платформа** | RunPod GPU (рекомендуется **NVIDIA H200**, ≥320 GB disk на `/workspace`) |

---

## 0. Для кого этот документ

Это **пошаговый runbook** для команды, которая поднимает **цифровых HR-аватаров** (Elena, Svetlana) на GPU-сервере и получает готовые MP4 + identity bank (`.pt`) для передачи в Job.ai.

**Не нужно знать PyTorch.** Достаточно уметь:

- подключиться по SSH к RunPod;
- копировать команды в терминал;
- дождаться завершения (иногда 30–90 минут на скачивание весов);
- проверить, что файлы появились в `output/`.

**PyTorch** — это библиотека для нейросетей на GPU. **Hugging Face (HF)** — каталог моделей. **Checkpoint** — папка с уже обученными весами модели. **Infer** — один прогон генерации (видео из картинки + аудио).

---

## 1. Что мы строим (простыми словами)

```
Фото лица (JPG)  +  WAV речи  +  текстовый prompt
        ↓
   ARACHNE-X avatar runtime (ULTRA-AVATAR + ULTRA-VIDEO)
        ↓
   MP4 с lipsync + файл identity bank (.pt) для стабильного лица
```

| Компонент | Зачем |
|-----------|--------|
| **ULTRA-VIDEO** | Базовое видео (t2v, i2v без губ) |
| **ULTRA-AVATAR** | Говорящая голова, lipsync, identity |
| **Merged runtime** | Одна папка-«сборка» для всех avatar-режимов |
| **Identity bank (.pt)** | «Память лица» — секунды GPU, не обучение 40 часов |
| **Job.ai** | Продукт; этот runbook — GPU bring-up и оцифровка аватаров |

**Production-путь для HR-аватаров:** режим **`ai2v`** + **`enroll_identity`**.

**Не production для Elena/Sveta на H200:** экспериментальный `audio_i2v` / `imagine_i2v` на VIDEO checkpoint (не хватает VRAM ~140 GB).

---

## 2. Два семейства checkpoint (запомнить)

| Режимы | Переменная | Папка |
|--------|------------|--------|
| `t2v`, `i2v`, `vc` | `$VIDEO_CKPT` | `weights/ARACHNE-X-ULTRA-VIDEO` |
| `ai2v`, `at2v`, `avc`, `streaming_ai2v`, `enroll_identity` | `$NULLXES_CHECKPOINT_DIR` | `weights/arachne-avatar-runtime` |

Avatar-режимы **всегда** используют **merged runtime**, не голый AVATAR snapshot.

---

## 3. Карта этапов (audit pipeline)

| # | Этап | Время | Критерий успеха |
|---|------|-------|-----------------|
| 0 | Диск / pod | 1 мин | `df -h` ≥ 280 GB free |
| 1 | Клон репо + venv | 5 мин | `git log -1` на `arachne-last-patch` |
| 2 | HF auth | 2 мин | `huggingface-cli whoami` OK |
| 3 | Скачать VIDEO (+ AVATAR если ещё нет) | 1–3 ч | `dit/`, `avatar_single/` на месте |
| 4 | **Merged runtime** (только после скачки!) | 2 мин | `test -d .../avatar_single` |
| 5 | PyTorch cu124 → flash-attn → deps | 30–60 мин | `FLASH OK`, версии ниже |
| 6 | Sanity GPU | 1 мин | CUDA True, H200 |
| 7 | Smoke VIDEO (t2v / i2v) | 10–40 мин | MP4 в `output/` |
| 8 | Оцифровка Elena / Svetlana | 15–60 мин | `.pt` + `.mp4` |
| 9 | Упаковка для Denis / Job.ai | 5 мин | zip архив |

**Порядок важен:** merged runtime **после** скачки; flash-attn **после** torch cu124; **не** ставить `torch` из `requirements.txt` поверх cu124.

---

## 4. Этап 0 — Pod и диск

```bash
df -h /workspace
nvidia-smi
```

Нужно **~280–320 GB свободно**:

- AVATAR ≈ 120 GB  
- VIDEO ≈ 80 GB+  
- HF cache, venv, output  

---

## 5. Этап 1 — Репозиторий и venv

```bash
cd /workspace
rm -rf ARACHNE-X   # только если нужна чистая установка

git clone https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git ARACHNE-X
cd /workspace/ARACHNE-X
git fetch origin
git checkout arachne-last-patch
git pull origin arachne-last-patch
git log -1 --oneline

export ARACHNE_ROOT=/workspace/ARACHNE-X
cd "$ARACHNE_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel packaging ninja psutil
```

**После каждого reconnect SSH** — снова:

```bash
cd /workspace/ARACHNE-X
source .venv/bin/activate
export ARACHNE_ROOT=/workspace/ARACHNE-X
export PYTHONPATH="$ARACHNE_ROOT"
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
export VIDEO_CKPT="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"
mkdir -p output
pwd   # ДОЛЖНО быть /workspace/ARACHNE-X
```

Ошибка `python: can't open file '//scripts/infer.py'` = вы в `/`, а не в репо.

---

## 6. Этап 2 — Hugging Face auth

**Токен только через env.** Не коммитить, не писать в чат.

```bash
cd /workspace/ARACHNE-X
source .venv/bin/activate

export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN=hf_... first}"

pip install -U "huggingface_hub[cli]>=0.34,<1.0"
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli whoami
```

Ускорение скачивания:

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Модели на HF (принять license где требуется):

| Модель | Назначение |
|--------|------------|
| [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) | Avatar + wav2vec |
| [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) | Video DiT |
| `google/gemma-2-2b-it` | Опционально: prompt compiler |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Опционально: TTS |

Скачивание (если ещё не скачано):

```bash
mkdir -p "$ARACHNE_ROOT/weights"

huggingface-cli download MagistrTheOne/ARACHNE-X-ULTRA-AVATAR \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR"

huggingface-cli download MagistrTheOne/ARACHNE-X-ULTRA-VIDEO \
  --local-dir "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"
```

---

## 7. Этап 4 — Merged avatar runtime (после скачки!)

```bash
export ARACHNE_ROOT=/workspace/ARACHNE-X
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"

mkdir -p "$NULLXES_CHECKPOINT_DIR/audio"

for d in tokenizer vae text_encoder scheduler dit; do
  ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO/$d" "$NULLXES_CHECKPOINT_DIR/$d"
done

for d in avatar_single avatar_multi chinese-wav2vec2-base vocal_separator; do
  ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/$d" "$NULLXES_CHECKPOINT_DIR/$d"
done

ln -sfn "$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-AVATAR/chinese-wav2vec2-base" \
  "$NULLXES_CHECKPOINT_DIR/audio/wav2vec2"

test -d "$NULLXES_CHECKPOINT_DIR/avatar_single" && echo "AVATAR RUNTIME OK"
test -d "$VIDEO_CKPT/dit" && echo "VIDEO OK"
```

---

## 8. Этап 5 — Зависимости (строгий порядок)

### 8.1 PyTorch CUDA 12.4 (первым)

```bash
source .venv/bin/activate
pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Ожидаем: 2.6.0+cu124 True
```

### 8.2 FlashAttention (только после torch)

```bash
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
python -c "import flash_attn; print('FLASH OK', flash_attn.__version__)"
```

**Не делать:** `pip install -U torch`, `pip install xformers`, `pip install torchaudio` с PyPI без cu124 index.

### 8.3 torchaudio cu124 (если слетел)

```bash
pip uninstall -y torchaudio
pip install torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 8.4 Остальные пакеты (torch уже стоит — не переустанавливать)

```bash
pip install diffusers==0.35.1 transformers==4.41.0 huggingface-hub==0.36.0 \
  accelerate==1.12.0 safetensors==0.7.0 loguru==0.7.2 einops==0.8.0 \
  ftfy==6.2.0 regex tqdm Pillow psutil av opencv-python \
  streamlit pyarrow imageio imageio-ffmpeg webdataset aiohttp

pip install librosa==0.11.0 soundfile==0.13.1 scikit-learn==1.6.1 \
  scikit-image==0.25.2 soxr pyloudnorm audio-separator \
  nvidia-ml-py tzdata onnx onnxruntime openai cffi chardet \
  aiortc silero-vad faster-whisper edge-tts

pip install qwen-tts   # опционально TTS
```

**AudioDiT** (`requirements-audiodit.txt`, transformers≥5.3) — **отдельный venv**, не смешивать с core.

### 8.5 ffmpeg

```bash
apt-get update && apt-get install -y ffmpeg jq tmux
ffmpeg -version | head -1
```

### 8.6 Runtime env (рекомендуется)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="$ARACHNE_ROOT"
```

### 8.7 Финальная проверка стека

```bash
python - <<'PY'
import torch, torchaudio, flash_attn, diffusers, transformers, librosa
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("flash:", flash_attn.__version__)
print("transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Ожидаемо на H200:

```
torch: 2.6.0+cu124
flash: 2.7.4.post1
transformers: 4.41.0
CUDA: True NVIDIA H200
```

---

## 9. Этап 6–7 — Smoke VIDEO (проверка весов)

Долгие прогоны — в **tmux**:

```bash
tmux new -s arachne
# Ctrl+B, D — отсоединиться; tmux attach -t arachne
```

### 9.1 t2v (текст → видео, без лица)

```bash
cd /workspace/ARACHNE-X && source .venv/bin/activate
export VIDEO_CKPT="$ARACHNE_ROOT/weights/ARACHNE-X-ULTRA-VIDEO"
export PYTHONPATH="$ARACHNE_ROOT"

python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode t2v \
  --prompt "Cinematic drone shot over a modern city at dusk, smooth camera motion, photorealistic." \
  --negative_prompt "blurry, low quality, watermark, text, distorted" \
  --height 480 --width 832 \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --output output/video_t2v_smoke.mp4
```

Если SSH оборвался — **перезапустить команду** (файл не будет, пока не дошло до конца).

### 9.2 i2v (картинка → видео, без аудио)

```bash
python scripts/infer.py \
  --checkpoint_dir "$VIDEO_CKPT" \
  --mode i2v \
  --image assets/avatar/single/elena/image.jpg \
  --prompt "Professional woman portrait, subtle natural motion, stable identity, clean background." \
  --negative_prompt "anime, cartoon, blurry, low quality, watermark, flicker" \
  --resolution 480p \
  --num_frames 49 \
  --num_inference_steps 25 \
  --text_guidance_scale 4.0 \
  --output output/video_i2v_smoke.mp4
```

Проверка:

```bash
ls -lh output/video_*.mp4
ffprobe -hide_banner output/video_t2v_smoke.mp4 2>&1 | grep Duration
```

---

## 10. Этап 8 — Оцифровка аватаров (Job.ai)

### 10.1 Подготовка аудио

WAV: **mono, 16 kHz**.

```bash
ffmpeg -y -i input.wav -ar 16000 -ac 1 output/elena_16k.wav
```

### 10.2 Identity bank — что это

- Файл `.pt` (~несколько MB), не 87 MB.  
- Создаётся за **секунды** командой `enroll_identity`.  
- **Не LoRA**, не «обучение модели с нуля».  
- Один и тот же **JPG** для enroll и для ai2v.

Проверка валидности:

```bash
python - <<'PY'
import torch
p = torch.load("output/elenaV2_identity_bank.pt", map_location="cpu")
print("OK keys:", "identity_embedding" in p, p["identity_embedding"].shape)
PY
```

Ошибка `failed finding central directory` = файл **битый** (переименовали чужой файл, обрезали при копировании).

---

### 10.3 Elena — enroll + ai2v

```bash
cd /workspace/ARACHNE-X
source .venv/bin/activate
export NULLXES_CHECKPOINT_DIR="$ARACHNE_ROOT/weights/arachne-avatar-runtime"
export PYTHONPATH="$ARACHNE_ROOT"
pkill -f "python scripts/infer.py" 2>/dev/null || true; sleep 2

# 1) Enroll
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode enroll_identity \
  --image assets/avatar/single/elena/image.jpg \
  --identity_id 1 \
  --identity_bank_save_path output/elenaV2_identity_bank.pt \
  --resolution 720p

# 2) ai2v — sync (лучший lipsync, ~3.2 s на 6 s аудио)
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --image assets/avatar/single/elena/image.jpg \
  --audio assets/avatar/single/elena/audio.wav \
  --prompt "ELENA, ultra realistic executive woman, speaking naturally straight to camera, talking, precise lipsync, stable identity, photorealistic skin, minimal head movement, high temporal consistency" \
  --negative_prompt "anime, cartoon, blurry, low quality, distorted face, duplicated mouth, frozen lips, bad anatomy, watermark, text, jitter" \
  --identity_bank_path output/elenaV2_identity_bank.pt \
  --identity_id 1 \
  --identity_strength 1.0 \
  --resolution 720p \
  --num_frames_mode sync \
  --num_inference_steps 35 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.5 \
  --output output/elena_ai2v_sync_v2.mp4
```

### 10.4 Elena — полные ~6 секунд (cinematic)

`sync` режет окно (~97 frames). Для **всего аудио**:

```bash
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --image assets/avatar/single/elena/image.jpg \
  --audio assets/avatar/single/elena/audio.wav \
  --prompt "ELENA, speaking naturally to camera, talking, precise lipsync, stable identity." \
  --negative_prompt "anime, cartoon, blurry, distorted face, duplicated mouth, frozen lips" \
  --identity_bank_path output/elenaV2_identity_bank.pt \
  --identity_id 1 \
  --identity_strength 1.0 \
  --resolution 720p \
  --num_frames_mode duration \
  --embedding_fps_auto \
  --num_inference_steps 35 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.5 \
  --output output/elena_ai2v_duration_v2.mp4
```

В логе смотреть строку `[frame-budget]` — там `chosen`, `sync_max`, `duration_sec`.

---

### 10.5 Svetlana — enroll + ai2v

```bash
# Enroll (не использовать битый assets/.../Svetlana.pt)
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode enroll_identity \
  --image assets/avatar/single/svetlana/sveta.jpg \
  --identity_id 1 \
  --identity_bank_save_path output/svetlanaV2_identity_bank.pt \
  --resolution 720p

# ai2v
python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --image assets/avatar/single/svetlana/sveta.jpg \
  --audio assets/avatar/single/svetlana/audio.wav \
  --prompt "SVETLANA, ultra realistic professional woman, speaking naturally straight to camera, talking, precise lipsync, stable identity, photorealistic skin, minimal head movement" \
  --negative_prompt "anime, cartoon, blurry, low quality, distorted face, duplicated mouth, frozen lips, bad anatomy, watermark, text, jitter" \
  --identity_bank_path output/svetlanaV2_identity_bank.pt \
  --identity_id 1 \
  --identity_strength 1.0 \
  --resolution 720p \
  --num_frames_mode sync \
  --num_inference_steps 35 \
  --text_guidance_scale 4.0 \
  --audio_guidance_scale 5.5 \
  --output output/svetlana_ai2v_v2.mp4
```

Preset: `assets/avatar/single/svetlana/svetlana.json`

---

## 11. Справочник режимов infer (кратко)

| Режим | Checkpoint | Вход | Выход | Job.ai |
|-------|------------|------|-------|--------|
| **ai2v** | avatar runtime | JPG + WAV + prompt | MP4 + mux audio | **Основной** |
| **enroll_identity** | avatar runtime | JPG | `.pt` bank | **Обязательно** |
| t2v | VIDEO | prompt | MP4 без звука | Smoke |
| i2v | VIDEO | JPG + prompt | MP4 без звука | Smoke |
| at2v | avatar | WAV + prompt (без фото) | MP4 | Редко |
| avc | avatar | video + WAV | MP4 | Re-dub |
| streaming_ai2v | avatar | JPG + WAV chunks | короткий MP4 | Realtime R&D |
| audio_i2v / imagine_i2v | VIDEO | lab | OOM на H200 | **Не для HR demo** |

### CFG (качество губ)

| Параметр | Smoke | Production |
|----------|-------|------------|
| `--text_guidance_scale` | 3–4 | **4.0** |
| `--audio_guidance_scale` | 3 | **5.0–5.5** |
| `--num_inference_steps` | 2–25 | **35** |

Prompt: слова **speaking, talking, lipsync, stable identity**.

### Длина кадров

- `num_frames` = **4n+1** (17, 49, 97, 165, 185…).  
- `--num_frames_mode sync` — лучший lipsync, короче.  
- `--num_frames_mode duration` — на всё аудио.

---

## 12. Упаковка для передачи (Denis / Job.ai)

```bash
cd /workspace/ARACHNE-X/output

zip -r ../jobai_avatar_pack_2026-05-21.zip \
  elenaV2_identity_bank.pt \
  svetlanaV2_identity_bank.pt \
  elena_ai2v_sync_v2.mp4 \
  elena_ai2v_duration_v2.mp4 \
  svetlana_ai2v_v2.mp4 \
  *.run.json 2>/dev/null || true

ls -lh ../jobai_avatar_pack_2026-05-21.zip
```

Скачать с pod: RunPod File Browser или `scp`.

---

## 13. Troubleshooting

| Симптом | Причина | Действие |
|---------|---------|----------|
| `can't open file //scripts/infer.py` | Не в `/workspace/ARACHNE-X` | `cd /workspace/ARACHNE-X` |
| `git pull` conflict | Лок правки на pod | `git checkout -- <file>` или `git stash` |
| `FLASH` import error | torch не cu124 или flash до torch | §8.1 → §8.2 |
| `torchaudio 2.11` | PyPI перетёр cu124 | §8.3 |
| t2v оборвался, MP4 нет | SSH disconnect | tmux + перезапуск |
| CUDA OOM на `audio_i2v` | VIDEO DiT ~139 GB | Использовать **ai2v** |
| `identity bank zip error` | Битый `.pt` | **enroll_identity** заново |
| Видео 3 s вместо 6 s | `--num_frames_mode sync` | `--num_frames_mode duration --embedding_fps_auto` |
| `[frame-budget] chosen=97` | sync max window | duration mode для полной длины |
| GPU память не освобождается | Старый процесс | `pkill -f "python scripts/infer.py"` + `nvidia-smi` |

Очистка GPU:

```bash
pkill -f "python scripts/infer.py" 2>/dev/null || true
sleep 3
python - <<'PY'
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    f,t = torch.cuda.mem_get_info()
    print(f"free {f/2**30:.1f} GB / {t/2**30:.1f} GB")
PY
nvidia-smi
```

---

## 14. Рекомендуемый порядок прогона (чеклист Job.ai)

- [ ] Pod H200, disk OK  
- [ ] Repo `arachne-last-patch`, venv  
- [ ] HF auth + weights AVATAR + VIDEO  
- [ ] Merged runtime symlinks  
- [ ] torch cu124 + flash-attn + deps  
- [ ] t2v smoke MP4  
- [ ] i2v smoke MP4  
- [ ] Elena enroll → `elenaV2_identity_bank.pt`  
- [ ] Elena ai2v sync + duration MP4  
- [ ] Svetlana enroll → `svetlanaV2_identity_bank.pt`  
- [ ] Svetlana ai2v MP4  
- [ ] zip pack → передача  

---

## 15. Экспериментальное (не блокирует Job.ai)

| Фича | Статус на H200 |
|------|----------------|
| `imagine_i2v` (Qwen + VIDEO) | Lab; VRAM tight |
| `audio_i2v` + adapter | Нужен trained adapter; без него ≈ base i2v |
| Qwen voice clone (Base model) | Отдельный скрипт TTS → потом ai2v |
| LoRA (`train_lora_avatar.py`) | Опционально после 20–50 пар |

Подробнее: `RUNPOD_H200_AVATAR_SETUP.md`, `Documentation/AUDIO_CONDITIONED_I2V.md`.

---

## 16. Контакты

**NULLXES / Job.ai avatar bring-up:** ceo@nullxes.com  

**Репозиторий:** https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-  
**Ветка:** `arachne-last-patch`  

---

*Документ подготовлен для операционного развёртывания цифровых HR-аватаров NULLXES на базе ARACHNE-X-ULTRA. Обновляйте дату и версию commit при изменении pipeline.*
