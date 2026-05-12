# V2 runtime audit — NIGHT FURY (ARACHNE-X)

Снимок репозитория для подготовки V2. **Не** заменяет прогон на RunPod.

---

## 1. Дублирование логики

| Зона | Факт | Рекомендация V2 |
|------|------|-----------------|
| Worker vs CLI streaming | Реализация в [`arachne_x/runtime/avatar_serving.py`](../../arachne_x/runtime/avatar_serving.py); [`gpu_avatar_runtime.py`](../../services/longcat-worker/gpu_avatar_runtime.py) — **lazy** re-export для uvicorn без PYTHONPATH на старте | Один модуль `avatar_serving`; воркер только HTTP |
| `execute_infer` streaming | [`arachne_x/runtime/inference_engine.py`](../../arachne_x/runtime/inference_engine.py) вызывает `pipe.generate_streaming_ai2v` с другим аудио-источником (файл/chunk iterator) | Тот же pipeline API; параметры negative_prompt для worker — gap (см. schema truth) |

---

## 2. CPU traps (production path)

| Место | Поведение |
|-------|-----------|
| `avatar_serving.get_avatar_pipeline` | `device == "cpu"` → **RuntimeError** (как раньше в воркере) |
| `inference_engine.execute_infer` | `device = cuda if available else cpu` — CLI на CPU возможен; **prod RunPod** — только CUDA |

---

## 3. Triton / BSA import graph

Импорт [`arachne_x/modules/attention.py`](../../arachne_x/modules/attention.py) тянет `flash_attn_bsa_3d` из [`arachne_x/block_sparse_attention/bsa_interface.py`](../../arachne_x/block_sparse_attention/bsa_interface.py), который импортирует **triton**.  
Итог: процесс с полным `import arachne_x...` на Linux CUDA должен иметь **совместимый triton** с колёсами PyTorch, иначе старт падает до выбора BSA.

---

## 4. Dtype split (bf16 vs fp16)

| Компонент | Dtype |
|-----------|--------|
| [`loader.py`](../../arachne_x/loader.py) default | `torch.bfloat16` на CUDA |
| [`streaming_inference.py`](../../arachne_x/streaming_inference.py) `StreamingVAEDecoder` | `autocast(..., dtype=torch.float16)` при `enable_amp` |

**Риск:** смешение fp16 decode с bf16 латентами — зафиксировать единую политику в конфиге сервиса (без смены архитектуры сети).

---

## 5. Legacy HTTP / naming

- Каталог `services/longcat-worker/` — internal path в git; внешнее имя: **NULLXES Inference Worker** (NIGHT FURY).
- Канон MP4: **`POST /v1/arachne/generate`**. **`POST /v1/longcat/generate`** остаётся **legacy alias** (не в OpenAPI schema); клиенты переводятся на `NULLXES_AVATAR_INFERENCE_PATH` по умолчанию `/v1/arachne/generate`.

---

## 6. MVP VRAM lifecycle (single Pod)

Для colocation **DiT + Whisper + Qwen + emotion2vec** на **H100 80 GB**:

- Замерить пик VRAM по фазам (загрузка только core → +ASR → unload ASR → +TTS → …).
- Документировать **sequential residency** или CPU offload для части SpeechAdapters.
- H200 ~141 GB — baseline для «всё резидентно» до профилирования.

---

**NULLXES** · V2 runtime audit
