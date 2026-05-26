# ARACHNE-X Avatar Architecture (NULLXES)

Operational doctrine for the avatar stack: what trains, what infers, what is forbidden, and why.  
This document is policy. Code is enforced from this contract — not the other way around.

**Install / deps:** [`Documentation/REQUIREMENTS.md`](Documentation/REQUIREMENTS.md)  
**Classification:** [`Documentation/ARACHNE_X_CLASSIFICATION_2026-05-21.md`](Documentation/ARACHNE_X_CLASSIFICATION_2026-05-21.md)  
**Engine iteration (GPT handoff):** [`Documentation/ARACHNE_ENGINE_ITERATION_HANDOFF_2026.md`](Documentation/ARACHNE_ENGINE_ITERATION_HANDOFF_2026.md)  
**Iteration roadmap (Sprint 1–4):** [`Documentation/ARACHNE_ITERATION_ROADMAP.md`](Documentation/ARACHNE_ITERATION_ROADMAP.md)

---

## Weights & lineage (production)

| | |
|--|--|
| **Runtime checkpoints** | NULLXES production only — [ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) (HR avatar), [ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) (T2V/I2V/VC) |
| **Not used in prod** | Hugging Face weights from [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video) / LongCat-Video-Avatar as runtime source |
| **Architecture reference** | 3D ACV-DiT family (~13.6B, d4096, L48) — same *class* of model as public LongCat-Video reports ([arXiv:2510.22200](https://arxiv.org/abs/2510.22200)); **tensor ABI** keeps historical class names (`LongCatVideoAvatarTransformer3DModel`) for checkpoint compatibility |
| **Code** | `arachne_x/` — NULLXES implementation; `longcat_video_dit*.py` are import shims only |

Merged avatar runtime layout (`NULLXES_CHECKPOINT_DIR`):

```
checkpoint_dir/
  tokenizer/  text_encoder/  vae/  scheduler/
  avatar_single/   avatar_multi/
  audio/wav2vec2/   audio/vocal_separator/Kim_Vocal_2.onnx
```

Resolve Hub → local: `arachne_x.weights_resolve.resolve_weights_root(..., allow_hub=True)`.

---

## Layer overview

```
   ┌──────────────────────────────────────────────────────────────┐
   │                    Behavior Layer (live)                     │  ← orchestration (HR monorepo)
   │  voice turn-taking · prompt builder · session memory         │
   ├──────────────────────────────────────────────────────────────┤
   │                    Runtime (serving)                         │  ← inference engine
   │  avatar_serving · streaming_inference · ffmpeg / RTMP        │
   ├──────────────────────────────────────────────────────────────┤
   │                    Motion Adapter (future)                   │  ← optional motion LoRA
   │  audio-driven motion residual on top of identity manifold    │
   ├──────────────────────────────────────────────────────────────┤
   │                    Identity LoRA (per-character)             │  ← attention-only LoRA
   │  Q/K/V/Out projections inside DiT blocks                     │
   ├──────────────────────────────────────────────────────────────┤
   │              Foundation DiT (frozen, NULLXES ULTRA weights)    │  ← ACV-DiT 13.6B
   │  3D RoPE · flow-match Euler · Wan VAE · UMT5 · wav2vec2      │
   └──────────────────────────────────────────────────────────────┘
```

---

## Realtime orchestration (production)

Canonical live path (single writer via WebSocket pump):

```
WebSocket chat.send / voice.pcm16
  → src/server/session_worker.py (SessionWorker)
  → src/server/realtime_avatar_loop.py (LLM → TTS → stream)
  → src/server/avatar_stream_client.py (PCM16 NDJSON POST)
  → services/arachnex-worker POST /v1/realtime/avatar_frames
  → arachne_x/runtime/avatar_serving.py → generate_streaming_ai2v()
  → SessionWorker.out_queue → realtime_api pump → WS avatar.stream.chunk
```

| Component | Role |
|-----------|------|
| `realtime_api.py` | WS gateway: auth, token mint, pump egress only — **not** a second orchestrator |
| `session_worker.py` | Per-session pipeline owner (VAD → ASR → LLM → TTS → GPU) |
| `arachnex-worker` | Dumb GPU machine: `imageBase64` + `audioPcm16Base64` → rgb24 NDJSON |

Worker contract: **no TTS inside GPU process**. TTS runs in orchestrator (`src/server/tts_runner.py`).

Streaming overload policy (env on worker): `ARACHNE_STREAM_MAX_ACTIVE_JOBS=1`, `ARACHNE_STREAM_MAX_QUEUE=3`, `ARACHNE_STREAM_QUEUE_TIMEOUT_SEC=15`. Overflow → HTTP 503 `{"error":"worker_busy","retryAfterMs":8000}`. Orchestrator client retries with jitter (`avatar_stream_client.py`).

Multi-worker routing (orchestrator): `NULLXES_AVATAR_WORKER_URLS=url1,url2` → `hash(session_id) % N` via `src/server/avatar_worker_router.py`.

Incremental wav2vec (TTFF): `ARACHNE_INCREMENTAL_WAV2VEC=1` (default) — partial encode (~400ms prefix) before chunk-0 denoise; full encode before chunk 1+. Tune via `ARACHNE_INCREMENTAL_WAV2VEC_MIN_MS`.

---

## 1. Foundation DiT

`arachne_x/modules/avatar/arachne_avatar_dit.py` — `LongCatVideoAvatarTransformer3DModel` (ABI name).

| Field | Value |
|-------|-------|
| Class (checkpoint ABI) | `LongCatVideoAvatarTransformer3DModel` |
| Hidden size | 4096 |
| Depth | 48 |
| Num heads | 32 |
| Channels (in / out) | 16 / 16 |
| Patch size | `(1, 2, 2)` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (shift 12, linear time-shift) |
| VAE | `AutoencoderKLWan` (z_dim 16, temporal stride 4, spatial stride 8) |
| Text encoder | UMT5-XXL (4096 dim) |
| Audio encoder | Wav2Vec2 (`audio/wav2vec2` in NULLXES bundle) |

**Status:** frozen at train time. Base weights = **ARACHNE-X-ULTRA-AVATAR** snapshot. Identity adaptation = LoRA + identity bank above.

Loading: `arachne_x.loader.load_avatar_pipeline()`.

---

## 2. Identity LoRA

`scripts/train_lora_avatar.py` + `arachne_x/modules/lora_utils.py`.

### Scope (locked policy)

LoRA is applied **only** to attention Q/K/V/Out projections inside transformer blocks. The denylist is enforced inside `avatar_attention_only_lora_filter` and cannot be bypassed by `--lora_prefixes` (the deny check runs before the override).

| Module path (real names in ARACHNE-X DiT) | LoRA? |
|--------------------------------------------|-------|
| `blocks.{N}.attn.qkv` | YES |
| `blocks.{N}.attn.proj` | YES |
| `blocks.{N}.cross_attn.q_linear` | YES |
| `blocks.{N}.cross_attn.kv_linear` | YES |
| `blocks.{N}.cross_attn.proj` | YES |
| `blocks.{N}.audio_cross_attn.q_linear` | YES |
| `blocks.{N}.audio_cross_attn.kv_linear` | YES |
| `blocks.{N}.audio_cross_attn.proj` | YES |
| `blocks.{N}.ffn.*` | **NO** |
| `blocks.{N}.adaLN_modulation.*` | **NO** |
| `audio_proj.*` | **NO** |
| `final_layer.*` | **NO** |
| `x_embedder.*` / `y_embedder.*` / `t_embedder.*` | **NO** |

### Anti-snow stack

| Layer | Mechanism | Why |
|-------|-----------|-----|
| Loss | Min-SNR flow-match (`γ=5`) | Down-weight high-σ steps that teach grain |
| Sampling | Timestep bias power 2.0 on export | More clean-σ samples for LoRA |
| Audio cond | Per-token RMS normalize | Stabilize lip / cheek jitter |
| Optimization | EMA decay 0.9995 | Smooth noisy LoRA updates |
| Attention | BSA off, FlashAttn-2 dense | Train ≡ Infer attention distribution |
| Scope | Attention-only | MLP / audio_proj LoRA overfits texture |

### Defaults (`scripts/train_lora_avatar.py`)

| Arg | Default | Notes |
|-----|---------|-------|
| `--lora_rank` | 128 (or `--config` JSON) | small character: 16 |
| `--lora_alpha` | 64 | small character: 8 |
| `--lora_key` | `train` | Elena HR uses `elenahr` |
| `--min_snr_gamma` | 5.0 | 0 = disable |
| `--normalize_audio_embs` | True | `--no-normalize_audio_embs` to A/B |
| `--ema_decay` | 0.9995 | 0 = off |
| `--enable_aux_losses` | False | Phase B only |

---

## 3. Motion Adapter

Reserved layer. Will sit between Foundation DiT and Identity LoRA when we introduce audio-driven motion residuals.

Until then: one identity LoRA per character; motion from base DiT + audio CFG, not from LoRA on FFN/audio_proj.

---

## 4. Behavior Layer

Outside `arachne_x/` — `backend/realtime-gateway/`, LiveKit agents, frontend.

The Behavior Layer **never** writes to avatar runtime; it sends `(audio chunks, prompt, lora_key)` and reads frames.

Схема **STT → LLM → TTS → avatar PP**: [`Documentation/ARACHNE_AVATAR_STT_LLM_TTS_SCHEMA.md`](Documentation/ARACHNE_AVATAR_STT_LLM_TTS_SCHEMA.md).

---

## 4.5 Prompt Intelligence Layer

`arachne_x/prompt_compiler/` — instruction step before frozen UMT5 (not a DiT adapter).

| Backend | Deployment | Role |
|---------|------------|------|
| `off` | Default / shipped | Passthrough + avatar template merge |

Wiring: `inference_engine.apply_prompt_compiler` → `encode_prompt` → identity tokens → DiT.  
See [`Documentation/PROMPT_COMPILER.md`](Documentation/PROMPT_COMPILER.md).

---

## 5. Runtime & startup paths

| Path | Entry | Notes |
|------|-------|-------|
| **CLI** | `scripts/infer.py` | Thin wrapper over `arachne_x.runtime.execute_infer` |
| **Library** | `arachne_x.runtime.InferenceEngine` | Same contract as CLI |
| **GPU worker** | `services/arachnex-worker/main.py` | FastAPI → `gpu_avatar_runtime` → `avatar_serving` (lazy CUDA load) |
| **Internal orchestrator** | `src/server/*` | HTTP client to worker; no second DiT process |

### Sampling OS (operational vs cinematic)

| Profile | Steps | Distill | Chunks | Frame cap | Use case |
|---------|-------|---------|--------|-----------|----------|
| `operational` | 12 | yes | F=33, overlap=8 | 65 (sync) | Worker realtime, short clips, TTFF |
| `cinematic` | 35 | no | monolithic | none | Quality baseline, eval rollback |

Wiring: `arachne_x/runtime/sampling_profiles.py` → `execute_infer` / `avatar_serving`.  
Chunk path: `generate_chunked_ai2v` (pipeline-level windows) + `chunk_stitch.cosine_blend`.  
Streaming: `generate_streaming_ai2v` delegates to chunked yield unless `ARACHNE_LEGACY_STREAMING=1`.  
Metrics: `RuntimeSamplingMetrics` → `.run.json` (`ttff_sec`, `dit_forwards`, `denoise_wall_sec`).  
**Stability OS (Sprint 2, active):** [`Documentation/ARACHNE_STABILITY_OS_SPRINT2.md`](Documentation/ARACHNE_STABILITY_OS_SPRINT2.md) — KV consume, identity refresh, silence gate, drift monitor.  
Cross-chunk KV seed (interim): `ARACHNE_CHUNK_KV=1` — **не prod** until Sprint 2 `S2-1-CHUNK-CONSUME` ships.

```bash
# Operational CLI (after H200 eval gate)
python scripts/infer.py --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v --runtime_profile operational \
  --image ref.jpg --audio speech.wav --prompt "speaking to camera"
```

### Environment (prod)

| Variable | Role |
|----------|------|
| `NULLXES_CHECKPOINT_DIR` / `ARACHNE_CHECKPOINT_DIR` | Merged AVATAR bundle root |
| `PYTHONPATH` | Repo root + `services/arachnex-worker` for uvicorn |
| `NULLXES_INFERENCE_SERVICE_KEY` | Optional; header `X-NULLXES-Avatar-Inference-Key` |
| `ARACHNE_INFER_ENABLE_BSA` | Infer-only block-sparse attention (default ON) |
| `ARACHNE_RUNTIME_PROFILE` | `operational` \| `cinematic` (CLI `--runtime_profile` overrides when set) |
| `ARACHNE_LEGACY_STREAMING` | `1` = monolithic denoise + stream VAE (one release rollback) |
| `ARACHNE_CHUNK_KV` | `1` = seed `kv_cache_dict` between chunks (Sprint 1.5; full consume pending) |

### CLI (HR primary)

```bash
export PYTHONPATH=/workspace/ARACHNE-X
export NULLXES_CHECKPOINT_DIR=/workspace/weights/arachne-avatar-runtime

python scripts/infer.py \
  --checkpoint_dir "$NULLXES_CHECKPOINT_DIR" \
  --mode ai2v \
  --image assets/avatar/single/elena/elena.jpg \
  --audio path/to/speech.wav \
  --prompt "speaking, looking at camera"
```

Modes: `ai2v`, `enroll_identity`, `at2v`, `avc`, `streaming_ai2v` (avatar); `t2v`, `i2v`, `vc` (VIDEO ckpt).

### Worker

```bash
export PYTHONPATH=/workspace/ARACHNE-X:/workspace/ARACHNE-X/services/arachnex-worker
export NULLXES_CHECKPOINT_DIR=/workspace/weights/arachne-avatar-runtime
cd services/arachnex-worker
uvicorn main:app --host 0.0.0.0 --port 9090
```

| Endpoint | Role |
|----------|------|
| `GET /health` | Liveness + lifecycle (`active`/`draining`), queue depth, VRAM |
| `GET /v1/runtime/metrics` | Queue rejects, wait times, active jobs (auth key) |
| `POST /v1/realtime/avatar_frames` | NDJSON RGB stream (PCM16 in); explicit queue + `503 worker_busy` |
| `POST /v1/admin/drain` | Stop admitting new streams (auth key) |
| `POST /v1/arachne/generate` | Sync MP4 |
| `POST /v1/infer/jobs` | Async MP4 job queue |

Worker details: [`services/arachnex-worker/README.md`](services/arachnex-worker/README.md).  
RunPod bring-up: [`RUNPOD_H200_AVATAR_SETUP.md`](RUNPOD_H200_AVATAR_SETUP.md).

### BSA at infer

`arachne_x/infer_attention.py::configure_infer_bsa(dit, enabled=None)` — env `ARACHNE_INFER_ENABLE_BSA`.

**Strict policy:** training calls `dit.disable_bsa()`; never import `configure_infer_bsa` from train code.

### Audio CFG (NULLXES prod)

| Context | `audio_guidance_scale` |
|---------|------------------------|
| Smoke / docs legacy | 3–5 |
| **Production lipsync** | **5.0–5.5** |
| Presets (`elena.json`) | per-block 3–5 |

---

## Policy table (binding)

| Feature | Train | Infer |
|---------|-------|-------|
| BSA | **NO** | Optional (env-gated) |
| EMA | **YES** | N/A |
| Min-SNR | **YES** | N/A |
| FFN LoRA | **NO** | **NO** |
| `audio_proj` LoRA | **NO** | **NO** |
| `final_layer` LoRA | **NO** | **NO** |
| Attention-only LoRA | **YES** | **YES** |
| Audio embs RMS norm | YES | implicit (windowed) |
| Timestep bias power | 2.0 | N/A |
| Gradient checkpointing | YES (DiT) | N/A |

Enforcement: `avatar_attention_only_lora_filter`, `verify_lora_avatar.py`, `train_lora_avatar.py` + `infer_attention.py` import guard.

---

## File map (production)

```
arachne_x/
  loader.py
  weights_resolve.py
  pipeline_arachne_x_video_avatar.py
  inference_audio.py
  inference_frames.py
  infer_attention.py
  streaming_inference.py
  runtime/inference_engine.py
  runtime/avatar_serving.py
  modules/avatar/arachne_avatar_dit.py
  modules/lora_utils.py
scripts/
  infer.py
  train_lora_avatar.py
  verify_lora_avatar.py
services/arachnex-worker/
  main.py
  gpu_avatar_runtime.py
```

Removed (NULLXES hardening): `arachne_x/model_adapter.py`, `arachne_x/config_realtime.py`.
