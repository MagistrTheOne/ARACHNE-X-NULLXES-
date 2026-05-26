# ARACHNE-X — зависимости и порядок установки (NULLXES)

**Цель:** не сломать prod RunPod / Docker при обновлении пинов.  
**Платформа prod:** Linux + NVIDIA CUDA (H200 primary). Windows — только unit-тесты без полного DiT-импорта.

---

## 1. Файлы requirements (роли)

| Файл | Когда ставить | Prod avatar infer |
|------|----------------|-------------------|
| [`requirements.txt`](../requirements.txt) | Всегда первым (после torch cu124 на pod) | **Да** |
| [`requirements_avatar.txt`](../requirements_avatar.txt) | Avatar / HR (`ai2v`, worker, train LoRA) | **Да** |
| [`requirements-tts.txt`](../requirements-tts.txt) | CLI `--speak_text`, semiauto TTS | Опционально |
| [`requirements-audiodit.txt`](../requirements-audiodit.txt) | `--tts_provider audiodit` | **Нет** — отдельный venv |
| [`requirements-datasets.txt`](../requirements-datasets.txt) | `scripts/fetch_hf_datasets.py` | **Нет** |
| [`services/arachnex-worker/requirements.txt`](../services/arachnex-worker/requirements.txt) | HTTP worker (FastAPI) | Перед uvicorn |

**Не смешивать** `requirements-audiodit.txt` с core: там `transformers>=5.3`, в core закреплено `transformers==4.41.0`.

---

## 2. Prod-установка (RunPod / Docker)

Порядок совпадает с [`RUNPOD_H200_AVATAR_SETUP.md`](../RUNPOD_H200_AVATAR_SETUP.md) §3.

```bash
# 1) Torch CUDA 2.6 + cu124 (образ RunPod или pytorch:2.6.0-cuda12.4)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2) flash-attn — только Linux, с --no-build-isolation (см. runbook §3.3)
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 3) Core
pip install -r requirements.txt

# 4) Avatar stack (не переустанавливать torch / triton вручную)
pip install -r requirements_avatar.txt

# 5) Опционально TTS для --speak_text
pip install -r requirements-tts.txt

# 6) Worker HTTP
pip install -r services/arachnex-worker/requirements.txt
```

Docker: [`docker/Dockerfile.gpu`](../docker/Dockerfile.gpu) — по умолчанию только `requirements.txt`; avatar/tts — build-args `INSTALL_AVATAR_DEPS=1`, `INSTALL_TTS_DEPS=1`.

---

## 3. Что реально импортирует prod-path

| Пакет | Где нужен |
|-------|-----------|
| `torch`, `diffusers`, `transformers`, `flash-attn` (Linux) | DiT, VAE, UMT5, attention |
| `librosa`, `soundfile`, `soxr`, `pyloudnorm` | `pipeline_arachne_x_video_avatar`, `inference_audio` |
| `opencv-python`, `av`, `Pillow` | кадры / mux |
| `openai` | prompt compiler backend `openai` (опционально на infer) |
| `onnx` / `onnxruntime` / `audio-separator` | **Demo** vocal separation, не `arachne_x.runtime` |
| `aiortc`, `silero-vad` | **Demo** / semiauto, не core worker |
| `qwen-tts` | только `--speak_text` / orchestration TTS |
| `streamlit`, `edge-tts`, `faster-whisper` | демо / вспомогательные CLI, не минимальный worker |

Удаление пакетов из `requirements_avatar.txt` без проверки ломает только Demo/semiauto, **не** NDJSON worker с готовым WAV.

---

## 4. Критические пины (не менять без матрицы)

| Пакет | Пин | Причина |
|-------|-----|---------|
| `torch` | 2.6.0 + cu124 | Согласован с flash-attn 2.7.4.post1 |
| `transformers` | 4.41.0 | UMT5 / wav2vec loaders |
| `diffusers` | 0.35.1 | VAE / scheduler API |
| `numpy` | 1.26.4 | бинарная совместимость стека |
| `flash-attn` | 2.7.4.post1 | только `platform_system == "Linux"` в requirements.txt |

**Triton** не пинится отдельно: приходит с CUDA torch; импорт `arachne_x.modules.attention` тянет BSA/triton на Linux. Не делать `pip install -U triton` поверх pod-стека (см. runbook).

---

## 5. Проверка без GPU (dev / CI)

```bash
export PYTHONPATH=/path/to/ARACHNE-X
python -m pytest tests/test_inference_frames.py tests/test_training_vae_latent.py tests/test_audio_conditioning_adapter.py -q
```

Полный `python scripts/infer.py --help` — только на Linux CUDA env с triton + flash-attn.

---

## 6. Веса (не pip)

Production checkpoints (NULLXES, не LongCat runtime):

- [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR)
- [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO)

```bash
export NULLXES_CHECKPOINT_DIR=/path/to/arachne-avatar-runtime
```

См. [`ARCHITECTURE.md`](../ARCHITECTURE.md) и [`ARACHNE_X_CLASSIFICATION_2026-05-21.md`](ARACHNE_X_CLASSIFICATION_2026-05-21.md).
