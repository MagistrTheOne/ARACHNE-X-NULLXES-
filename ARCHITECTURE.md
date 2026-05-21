# ARACHNE-X Avatar Architecture (NULLXES)

Operational doctrine for the avatar stack: what trains, what infers, what is forbidden, and why.
This document is policy. Code is enforced from this contract — not the other way around.

Lineage: derives from upstream [`meituan-longcat/LongCat-Video-Avatar`](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar) (MIT). Components below extend that base for production HR / realtime avatar use.

---

## Layer overview

```
   ┌──────────────────────────────────────────────────────────────┐
   │                    Behavior Layer (live)                     │  ← orchestration
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
   │                    Foundation DiT (frozen)                   │  ← LongCat avatar base
   │  3D RoPE · flow-match Euler · Wan VAE · UMT5 · wav2vec2      │
   └──────────────────────────────────────────────────────────────┘
```

---

## 1. Foundation DiT

`arachne_x/modules/avatar/arachne_avatar_dit.py` — `LongCatVideoAvatarTransformer3DModel`.

| Field | Value |
|-------|-------|
| Class | `LongCatVideoAvatarTransformer3DModel` |
| Hidden size | 4096 |
| Depth | 48 |
| Num heads | 32 |
| Channels (in / out) | 16 / 16 |
| Patch size | `(1, 2, 2)` |
| Scheduler | `FlowMatchEulerDiscreteScheduler` (shift 12, linear time-shift) |
| VAE | `AutoencoderKLWan` (z_dim 16, temporal stride 4, spatial stride 8) |
| Text encoder | UMT5-XXL (4096 dim) |
| Audio encoder | Wav2Vec2 (`chinese-wav2vec2-base` in upstream layout, `audio/wav2vec2` in our layout) |

**Status:** frozen at train time. We never fine-tune base weights. Identity adaptation happens in the LoRA layer above.

Loading goes through `arachne_x.loader.load_avatar_pipeline()` which expects the directory layout in `WeightsLayout`:
```
checkpoint_dir/
  tokenizer/  text_encoder/  vae/  scheduler/
  avatar_single/   avatar_multi/
  audio/wav2vec2/   audio/vocal_separator/Kim_Vocal_2.onnx
```

---

## 2. Identity LoRA

`scripts/train_lora_avatar.py` + `arachne_x/modules/lora_utils.py`.

### Scope (locked policy)

LoRA is applied **only** to attention Q/K/V/Out projections inside transformer blocks. The denylist is enforced inside `avatar_attention_only_lora_filter` and cannot be bypassed by `--lora_prefixes` (the deny check runs before the override).

| Module path (real names in ARACHNE-X DiT) | LoRA? | Upstream-style equivalent |
|--------------------------------------------|-------|---------------------------|
| `blocks.{N}.attn.qkv` | YES | `to_q` + `to_k` + `to_v` (combined) |
| `blocks.{N}.attn.proj` | YES | `to_out` |
| `blocks.{N}.cross_attn.q_linear` | YES | `cross_attn.to_q` |
| `blocks.{N}.cross_attn.kv_linear` | YES | `cross_attn.to_k` + `to_v` (combined) |
| `blocks.{N}.cross_attn.proj` | YES | `cross_attn.to_out` |
| `blocks.{N}.audio_cross_attn.q_linear` | YES | audio Q |
| `blocks.{N}.audio_cross_attn.kv_linear` | YES | audio K/V |
| `blocks.{N}.audio_cross_attn.proj` | YES | audio Out |
| `blocks.{N}.ffn.*` | **NO** | — |
| `blocks.{N}.adaLN_modulation.*` | **NO** | — |
| `audio_proj.*` | **NO** | (head, not block) |
| `final_layer.*` | **NO** | — |
| `x_embedder.*` / `y_embedder.*` / `t_embedder.*` | **NO** | — |

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
| `--lora_scope` | *removed* | scope is constant `attention` |
| `--min_snr_gamma` | 5.0 | 0 = disable |
| `--normalize_audio_embs` | True | `--no-normalize_audio_embs` to A/B |
| `--ema_decay` | 0.9995 | 0 = off |
| `--enable_aux_losses` | False | Phase B only |

---

## 3. Motion Adapter

Reserved layer. Will sit between Foundation DiT and Identity LoRA when we introduce audio-driven motion residuals (e.g. style adapters trained on motion clips, separate from identity LoRA).

Until then:
* train one LoRA per character (identity) with the locked attention scope
* motion characteristics come from base DiT + audio CFG, not from LoRA

Documenting the layer here so identity LoRA training never absorbs motion responsibilities (which is what FFN / audio_proj LoRA implicitly does and why they are banned).

---

## 4. Behavior Layer

Lives outside `arachne_x/` — in `backend/realtime-gateway/` and frontend orchestration.

Not in scope for this document. Listed for completeness so the avatar stack boundaries are clear:

* turn-taking, VAD, interrupt handling — `staged-voice-client.ts`, `useJobaiLiveKitInterviewV2.ts`
* prompt build / session memory — gateway dialogue store
* TTS / STT — OpenAI Realtime or staged STT→LLM→TTS

The Behavior Layer **never** writes to avatar runtime; it sends `(audio chunks, prompt, lora_key)` and reads frames.

---

## 4.5 Prompt Intelligence Layer

`arachne_x/prompt_compiler/` — LTX-style **instruction** step before frozen UMT5 (not a DiT adapter).

| Backend | Deployment | Role |
|---------|------------|------|
| `openai` | Production | Expand short intent via OpenAI (`prompt_enhancer`, `force=True`) |
| `gemma` | RunPod calibration | Local `ARACHNE_GEMMA_MODEL` on CUDA |
| `off` | Default | Passthrough + avatar template merge (lipsync, static camera, negative guard) |

Wiring: `inference_engine.apply_prompt_compiler` → `encode_prompt` → identity tokens → DiT.  
See [`Documentation/PROMPT_COMPILER.md`](Documentation/PROMPT_COMPILER.md).

**Phase B (planning tokens):** `arachne_x/planning/planning_token_head.py` — optional extra cross-attn tokens; disabled by default (`planning_enabled=False`).

**Phase C (audio plate):** `arachne_x/modules/audio/nullxes_audio_encoder.py` — shape-compatible wav2vec replacement; `ARACHNE_AUDIO_ENCODER=nullxes|wav2vec`.

---

## 5. Runtime

`arachne_x/runtime/inference_engine.py`, `arachne_x/runtime/avatar_serving.py`, `arachne_x/streaming_inference.py`, `scripts/infer.py`.

### CLI

`scripts/infer.py --mode {at2v,ai2v,avc,streaming_ai2v,enroll_identity} --checkpoint_dir … [--lora_path … --lora_key …]`

### BSA at infer

`arachne_x/infer_attention.py::configure_infer_bsa(dit, enabled=None)` toggles block-sparse attention. Controlled by `ARACHNE_INFER_ENABLE_BSA` env (default ON for upstream parity).

**Strict policy:** training code does NOT import `configure_infer_bsa` and explicitly calls `dit.disable_bsa()`. Mixing BSA-train with dense-infer (or vice versa) drifts the attention distribution → temporal flicker.

### Audio CFG

| Source | Recommended range |
|--------|-------------------|
| Upstream docs | 3–5 |
| Production lipsync (NULLXES) | 5–6 |
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

Violations are caught:

* `avatar_attention_only_lora_filter` denylist (lora_utils.py)
* `verify_lora_avatar.py` asserts no `audio_proj` / `ffn` / `final_layer` keys in saved LoRA
* `train_lora_avatar.py` calls `dit.disable_bsa()` and logs `BSA disabled for LoRA train`
* `infer_attention.py` header warning; never imported by train

---

## File map (current production)

```
arachne_x/
  loader.py                         # canonical bundle loader
  weights_resolve.py                # HF repo or local dir resolver
  pipeline_arachne_x_video_avatar.py# main avatar pipeline (ai2v/at2v/avc/streaming)
  inference_audio.py                # windowed audio embs (train+infer parity)
  inference_frames.py               # 4n+1 frame budget helpers
  infer_attention.py                # BSA toggle (INFER-ONLY)
  streaming_inference.py            # KV cache, streaming VAE decoder
  training_latent_export.py         # avatar latent .pt export (Min-SNR ts bias)
  training_latent_export_base.py    # base-video latent export (no audio)
  training_latent_common.py         # collate / validate
  training_wds.py                   # WebDataset iterable
  training_lora_loss.py             # Min-SNR flow-match loss + audio stabilize
  training_vae_latent.py            # z0 estimation + VAE denorm (matches pipeline)
  training_avatar_aux.py            # staged aux losses (Phase B)
  modules/
    avatar/arachne_avatar_dit.py    # the DiT
    avatar/attention.py             # avatar attention (qkv + cross_attn + audio_cross_attn)
    avatar_losses.py                # ARACHNEAvatarLossModule (perceptual/identity/lip/temporal)
    autoencoder_kl_wan.py           # Wan VAE
    scheduling_flow_match_euler_discrete.py
    identity_encoder.py             # DINOv2 (aux only)
    lora_utils.py                   # avatar_attention_only_lora_filter (policy-locked)
scripts/
  train_lora_avatar.py              # the trainer
  infer.py                          # the inferer
  verify_lora_avatar.py             # LoRA policy + roundtrip smoke tests
  export_latent_training_sample.py  # one-sample export CLI
  gpu/train_elenahr_lora.sh         # H200 train wrapper (Elena HR)
  gpu/export_elena_lora_smoke.sh    # H200 export wrapper (Elena HR)
```

Removed (NULLXES hardening): `arachne_x/model_adapter.py`, `arachne_x/config_realtime.py`.
