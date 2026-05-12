# ARACHNE-X-ULTRA V2 — NIGHT FURY — internal engineering directive (NULLXES)

**Codename:** `ARACHNE-X-ULTRA V2 — NIGHT FURY`  
**Цель:** production-grade **persistent semi-autonomous digital actor** (не offline-only video generator, не demo-only вход).  
**MVP deployment:** один **RunPod Linux GPU Pod** — CoreNULLXES + SpeechAdapters + session layer **colocated**; scale-out — Phase2.

---

## 1. Deployment policy

| Разрешено | Запрещено для production path |
|-----------|--------------------------------|
| **RunPod** Linux GPU (H200 primary, H100 secondary, B200 experimental) | Windows deployment, локальные ad-hoc CUDA/Triton сборки |
| Один Pod MVP (все нейромодули на одной машине до SLA/VRAM лимита) | CPU как основной путь DiT/VAE |

Детали стека и матрица версий: [`GTM_V2_RUNPOD_DEP_MATRIX.md`](GTM_V2_RUNPOD_DEP_MATRIX.md), [`GTM_ULTRA_V2_NIGHT_FURY.md`](GTM_ULTRA_V2_NIGHT_FURY.md).

---

## 2. Model ownership (checkpoints)

**Proprietary, independently trained, production checkpoints** — только NULLXES для VIDEO/AVATAR:

| Линейка | Hugging Face (публикация) |
|---------|---------------------------|
| VIDEO | [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) |
| AVATAR | [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) |

Отдельно: **SpeechAdapters** (ASR / emotion / TTS) — сторонние веса с Hub; апрув и пины — [`GTM_V2_HF_MODEL_APPROVAL.md`](GTM_V2_HF_MODEL_APPROVAL.md). Не смешивать с папкой proprietary checkpoint без явного решения о layout кэша.

---

## 3. Каноническая матрица слоёв (Layer / Model)

| Layer | Model / компонент | Примечание |
|-------|-------------------|------------|
| Text understanding | **UMT5** | Из layout чекпоинта; [`arachne_x/loader.py`](../../arachne_x/loader.py) |
| Video latents | **Proprietary VAE** | NULLXES; ABI — `GTM_VAE_ABI.md` |
| Diffusion runtime | **ARACHNE-X DiT** | NULLXES weights |
| Lip conditioning | **Internal Wav2Vec2** | Веса в AVATAR `audio/wav2vec2`; [`arachne_x/inference_audio.py`](../../arachne_x/inference_audio.py) |
| ASR | **Whisper Large V3 Turbo** | SpeechAdapter; HF id — см. approval table |
| Emotion embeddings | **emotion2vec** | SpeechAdapter → ControlBus |
| TTS | **Qwen3-TTS** | SpeechAdapter; см. `requirements-tts.txt` |
| Session state | **ControlBus + SessionMemory** | [`arachne_x/actor_v2/`](../../arachne_x/actor_v2/) |
| Temporal stabilization | **TemporalGovernor** | [`arachne_x/actor_v2/temporal_governor.py`](../../arachne_x/actor_v2/temporal_governor.py) |

**CoreNULLXES** vs **SpeechAdapters** — разные политики лицензий и обновлений.

---

## 4. MVP single-Pod topology

- **Один Pod:** Inference Worker (FastAPI) + `arachne_x.runtime` + pipeline + Whisper + emotion2vec + Qwen3-TTS + SessionMemory/ControlBus/TemporalGovernor в одном процессе/образе MVP.
- **VRAM:** H200 ~141 GB — целевой комфорт; H100 80 GB — политика **model lifecycle** (не все модели резидентны одновременно); фиксировать замерами — [`GTM_V2_RUNTIME_AUDIT.md`](GTM_V2_RUNTIME_AUDIT.md).
- **Phase2:** вынести speech на отдельные Pod’ы при превышении SLA/VRAM.

Поток данных MVP: PCM/события → SessionMemory → ASR/emotion/TTS → ControlBus → runtime → pipeline → TemporalGovernor → NDJSON/клиент.

---

## 5. Inference contour (canonical)

```text
Inference Worker (FastAPI)
  → arachne_x.runtime (execute_infer / InferenceEngine + avatar_serving)
  → ARACHNE-X pipelines
  → streaming frames / MP4
```

`scripts/infer.py` — **thin CLI wrapper only**.

Соответствие HTTP JSON полей и логического `InferenceJob`: [`GTM_SCHEMA_TRUTH_INFERENCE_HTTP.md`](GTM_SCHEMA_TRUTH_INFERENCE_HTTP.md).

---

## 6. Employee packs (deployable artifact)

Каталог: `employee_packs/<employee_id>/`

| Файл | Назначение |
|------|------------|
| `manifest.json` | Версия схемы, `employee_id`, ссылки на остальные файлы, `checkpoint_tag` (VIDEO/AVATAR), опционально `eval_baseline_id` |
| `identity_bank.pt` или путь | Банк идентичности (совместимо с `load_identity_bank` / CLI) |
| `reference_images/` | Референс лица |
| `prompt_profile.json` | Шаблоны prompt / negative_prompt по сценам |
| `lora.json` | Пути к LoRA и ключи (если используются) |
| `voice_profile.json` | Параметры TTS (Qwen3-TTS и т.д.) |
| `behavior_profile.json` | Стартовые значения для ControlBus / эмоций |

Схема JSON может эволюционировать; версия в `manifest.json` (`schema_version`).

---

## 7. Связанные документы

- [`GTM_PRODUCTION_CONTRACT.md`](GTM_PRODUCTION_CONTRACT.md)
- [`GTM_V2_HF_MODEL_APPROVAL.md`](GTM_V2_HF_MODEL_APPROVAL.md)
- [`GTM_SCHEMA_TRUTH_INFERENCE_HTTP.md`](GTM_SCHEMA_TRUTH_INFERENCE_HTTP.md)
- [`GTM_V2_RUNTIME_AUDIT.md`](GTM_V2_RUNTIME_AUDIT.md)
- [`GTM_V2_RUNPOD_DEP_MATRIX.md`](GTM_V2_RUNPOD_DEP_MATRIX.md)
- [`ADR_V2_SPEECH_STACK.md`](ADR_V2_SPEECH_STACK.md)

---

**NULLXES** · internal directive · не изменяет план-файл Cursor
