# NULLXES Classification Card — ARACHNE-X

| Field | Value |
|-------|-------|
| **Date** | 21.05.2026 |
| **Team** | NULLXES |
| **Scope** | ARACHNE-X only (repo + production runtime) |
| **Branch** | `arachne-last-patch` |
| **Related** | [`JOB_AI_AVATAR_RUNBOOK_2026-05-21.md`](JOB_AI_AVATAR_RUNBOOK_2026-05-21.md) · [`ARACHNE_X_SCALE_UP_TRAINING_ROADMAP_50B_2026-05-21.md`](ARACHNE_X_SCALE_UP_TRAINING_ROADMAP_50B_2026-05-21.md) · [`ARCHITECTURE.md`](../ARCHITECTURE.md) |

---

## Canonical name

**`NULLXES ARACHNE-X ULTRA · ACV-DiT-13.6B`**

| Segment | Meaning |
|---------|---------|
| **NULLXES** | Operational platform for realtime digital employees |
| **ARACHNE-X** | Stack codename (library + runtime + checkpoints) |
| **ULTRA** | Production weight tier (independently trained, Hub-published) |
| **ACV-DiT** | **A**udio-**C**onditioned **V**ideo **Di**ffusion **T**ransformer |
| **13.6B** | Active foundation DiT parameter count (~13.6×10⁹) |

Operational slug for logs, runbooks, and `.run.json` metadata:

```
arachne-x-ultra-acv-dit-13.6b-d4096-l48
```

---

## Why this naming (NULLXES doctrine)

1. **Class describes runtime role**, not marketing — ACV-DiT states the operational contract: diffusion over video latents with per-block audio cross-attention.
2. **Tier is separate from class** — ULTRA = frozen production weights; character adaptation = identity bank + attention LoRA, not a new base class.
3. **Dimensionality in the slug** — `d4096-l48` is debuggable under production pressure without opening config shards.
4. **Product vision vs class string** — V2 «Ночная Фурия» is the stateful-realtime product target; the class string is frozen tensor geometry.
5. **Infrastructure vocabulary** — modes, checkpoints, and lifecycle states are explicit and observable externally.

---

## Model class

| Field | Value |
|-------|-------|
| **Family** | ARACHNE-X |
| **Tier** | ULTRA (production) |
| **Class** | 3D Audio-Conditioned Video DiT (ACV-DiT) |
| **Paradigm** | Flow-match Euler diffusion → Wan VAE latents |
| **Ingress** | image · audio · text · (video for `avc`) |
| **Egress** | MP4 (30 fps mux) · latent stream (serving) |
| **Adaptation** | identity bank · attention-only LoRA · `enroll_identity` |
| **Frozen at train** | foundation DiT · UMT5 · wav2vec · VAE |
| **Runtime weights** | [ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) — **not** meituan-longcat Hub snapshots in prod |
| **Lineage** | 3D DiT architecture class per [LongCat-Video report](https://arxiv.org/abs/2510.22200); NULLXES independently trained ULTRA checkpoints |

**Deps / install:** [`Documentation/REQUIREMENTS.md`](REQUIREMENTS.md) · **Policy:** [`ARCHITECTURE.md`](../ARCHITECTURE.md)

---

## Dimensionality and parameters

### Foundation DiT (avatar path)

| Geometry | Value |
|----------|-------|
| Hidden **d** | **4096** |
| Depth **L** | **48** transformer blocks |
| Attention heads | 32 (head dim 128) |
| Latent channels | 16 in / 16 out |
| Patch size | `(1, 2, 2)` temporal × spatial |
| Active params | **~13.6B** (DiT core) |

Implementation: `arachne_x/modules/avatar/arachne_avatar_dit.py` — `LongCatVideoAvatarTransformer3DModel`.

### Satellite encoders (runtime bundle; not counted in 13.6B)

| Component | Spec |
|-----------|------|
| **UMT5-XXL** | 4096-d → text cross-attention every block |
| **Wav2Vec2** | 12 hidden layers stacked → `[T_emb, 12, 768]` |
| **AudioProjModel** | local window W=5 → **32 × 768** tokens per video frame |
| **Wan VAE** | z=16, temporal stride **4**, spatial stride **8** |
| **Identity bank** | **1024 slots × (4 tokens × 4096)** → tensor `[1024, 16384]` |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` (shift 12) |

### Character LoRA (per-avatar, optional)

| Param | Production default |
|-------|-------------------|
| Rank | 128 (smoke: 16) |
| Alpha | 64 (smoke: 8) |
| Scope | attention Q/K/V/Out only — self-attn, text cross-attn, **audio cross-attn** |
| Policy | FFN / `audio_proj` / `final_layer` LoRA **forbidden** |

Trainable delta: order 10⁷–10⁸ params (≪ 13.6B base).

### Operational footprint (disk / VRAM; not param count)

| Asset | Typical size |
|-------|--------------|
| ULTRA-AVATAR bundle | ~120 GB |
| ULTRA-VIDEO bundle | ~80 GB+ |
| Merged avatar runtime | `weights/arachne-avatar-runtime` (symlinks) |
| Enrolled identity bank `.pt` | ~321 MB / character |
| VRAM at infer (H200 class) | ~110–120 GB loaded |

---

## Checkpoint topology

```
NULLXES_CHECKPOINT_DIR = weights/arachne-avatar-runtime   ← production avatar infer
VIDEO_CKPT             = weights/ARACHNE-X-ULTRA-VIDEO      ← t2v / i2v / smoke
                         weights/ARACHNE-X-ULTRA-AVATAR     ← avatar weight source

Layout (WeightsLayout):
  tokenizer/  text_encoder/  vae/  scheduler/
  avatar_single/  avatar_multi/
  audio/wav2vec2/  audio/vocal_separator/
```

**Binding rule:** HR avatar production = **merged avatar runtime**, not raw VIDEO DiT alone.

Hub cards:

- [ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR)
- [ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO)

---

## Runtime modes (`scripts/infer.py`)

| Mode | Checkpoint | Input | Output | Job.ai |
|------|------------|-------|--------|--------|
| **`ai2v`** | avatar runtime | JPG + WAV + prompt | MP4 + mux audio | **Primary** |
| **`enroll_identity`** | avatar runtime | JPG | `.pt` identity bank | **Required** |
| `at2v` | avatar runtime | WAV + prompt | MP4 | Rare |
| `avc` | avatar runtime | video + WAV | MP4 | Re-dub / continuation |
| `streaming_ai2v` | avatar runtime | JPG + audio chunks | short MP4 | Realtime R&D |
| `t2v` / `i2v` | VIDEO | text / image | MP4 | Smoke only |
| `audio_i2v` / `imagine_i2v` | VIDEO | lab inputs | — | **Not HR prod** (OOM on H200) |

---

## Production inference parameters

Validated **21.05.2026** on RunPod H200 (`f04386344291`), Job.ai avatar bring-up.

| Parameter | Smoke | Production |
|-----------|-------|------------|
| `--resolution` | `480p` | **`720p`** |
| `--num_inference_steps` | 2–17 | **25** (budget) / **35** (max quality) |
| `--text_guidance_scale` | 3.0–4.0 | **4.0** |
| `--audio_guidance_scale` | 3.0 | **5.0–5.5** |
| `--num_frames_mode` | explicit `17` | **`sync`** (lipsync) / `duration` (full audio) |
| `--identity_strength` | — | **1.0** |
| `--identity_id` | — | **1** (per bank file) |
| Mux fps | — | **30** (fixed post) |
| Frame count rule | — | **4n+1** (17, 49, 97, 113, 165, 185…) |
| Default `embedding_fps` | — | **64** (= 16 × VAE temporal stride) |

Triple CFG at denoise: uncond (null text + zero audio) · text-only · full (text + audio).

Prompt contract: include **speaking, talking, lipsync, stable identity**.

---

## Measured wall-clock (sync, 25 steps, 720p)

| Avatar | Frames | Video duration | Denoise wall | s/it |
|--------|--------|----------------|--------------|------|
| Elena | 97 | 3.23 s | ~31 min | ~75 |
| Svetlana | 113 | 3.77 s | ~41 min | ~98 |

Deliverables from bring-up:

- `output/elenaV2_identity_bank.pt`
- `output/svetlanaV2_identity_bank.pt`
- `output/elena_ai2v_sync_v2.mp4`
- `output/svetlana_ai2v_sync_v2.mp4`

`sync` trades full audio length for stable lipsync and ~2× lower frame budget vs `duration`.

---

## NULLXES layer stack

```
Behavior        → gateway / turn-taking / session memory     [outside arachne_x]
Runtime         → inference_engine · avatar_serving · infer.py
Prompt Intel    → prompt_compiler (Gemma / OpenAI / off) → UMT5
Adaptation      → identity bank + attention-only LoRA       [trainable surface]
Motion Adapter  → reserved (future audio motion residual)
Foundation      → ACV-DiT 13.6B + Wan VAE + wav2vec        [frozen]
```

---

## External one-liner

> **NULLXES ARACHNE-X ULTRA** — infrastructure-grade **13.6B audio-conditioned 3D video diffusion runtime** (d4096, L48) for hyperrealistic digital HR employees: enroll-once identity bank, triple-CFG lipsync infer, merged avatar runtime on H200-class GPU.

---

## File map (classification-relevant)

| Path | Role |
|------|------|
| `arachne_x/modules/avatar/arachne_avatar_dit.py` | Foundation ACV-DiT |
| `arachne_x/pipeline_arachne_x_video_avatar.py` | Avatar pipeline, identity bank, ai2v |
| `arachne_x/inference_frames.py` | Frame budget (`sync` / `duration` / 4n+1) |
| `arachne_x/inference_audio.py` | Windowed wav2vec embeddings |
| `arachne_x/runtime/inference_engine.py` | CLI/runtime execution |
| `scripts/infer.py` | Thin infer CLI |
| `ARCHITECTURE.md` | Binding train/infer policy |
